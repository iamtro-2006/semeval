"""Portable teacher artifacts and strict student/answer formatting (no GPU imports)."""
import hashlib
import json
import re

from ..common import path, read_jsonl


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def parse_teacher(text, finish_reason):
    if finish_reason != 'stop':
        raise ValueError(f'Incomplete teacher output: {finish_reason}')
    # Thinking-model chat templates may already supply the opening <think> token.
    if '</think>' not in text:
        raise ValueError('Teacher did not finish its thinking section')
    body = text.rsplit('</think>', 1)[1].strip()
    if body.startswith('```json') and body.endswith('```'):
        body = body[7:-3].strip()
    obj = json.loads(body)
    if obj.get('supported') is not True:
        raise ValueError(f'Unsupported reference: {str(obj.get("issue", "unspecified"))[:500]}')
    steps = obj.get('steps')
    if not isinstance(steps, list) or not 3 <= len(steps) <= 6:
        raise ValueError('Expected 3–6 reasoning steps')
    if any(not isinstance(s, str) or not 12 <= len(s.strip()) <= 1500
           or re.search(r'</?(?:think|answer)>', s) for s in steps):
        raise ValueError('Invalid reasoning step')
    if sum(len(s.split()) for s in steps) > 350:
        raise ValueError('Reasoning exceeds the concise solution budget')
    return '\n'.join(f'{i}. {s.strip()}' for i, s in enumerate(steps, 1))


def assistant_target(row):
    answer = row['answer'].strip()
    if re.search(r'</?(?:think|answer)>', answer):
        raise ValueError('Gold answer contains reserved formatting tags')
    if row.get('reasoning'):
        return f'<think>\n{row["reasoning"]}\n</think>\n<answer>{answer}</answer>'
    # Held-out gold has no fabricated teacher reasoning: this loss is an answer-only proxy.
    return f'<answer>{answer}</answer>'


def final_answer(text):
    matches = list(re.finditer(r'<answer>\s*(.*?)\s*</answer>', text, re.DOTALL))
    if len(matches) != 1 or not matches[0].group(1).strip():
        raise ValueError('Missing or ambiguous final <answer>; increase generation budget or inspect the trace')
    match = matches[0]
    prefix, suffix = text[:match.start()].strip(), text[match.end():].strip()
    if not prefix.startswith('<think>') or not prefix.endswith('</think>') or suffix:
        raise ValueError('Incomplete reasoning/answer format; inspect the trace')
    answer = match.group(1).strip()
    if '<think>' in answer or '</think>' in answer:
        raise ValueError('Reasoning leaked into the final answer')
    return answer


def load_reasoned_train(cfg, train, validation):
    """Join by exact source rows and verify hashes, completeness, coverage and split isolation."""
    directory = path(cfg['data']['reasoning_dir'])
    report = json.loads((directory / 'dataset.json').read_text(encoding='utf-8'))
    if report.get('schema_version') != 1 or report.get('complete') is not True or report.get('smoke'):
        raise ValueError('Reasoning dataset is incomplete or smoke-only')
    if report['source_train_hash'] != digest(train) or report['source_validation_hash'] != digest(validation):
        raise ValueError('Teacher/student train or validation split mismatch')
    cfg.setdefault('reasoning', {})['dataset_report'] = report
    records = read_jsonl(directory / 'train_reasoning.jsonl')
    if digest(records) != report['accepted_hash']:
        raise ValueError('Reasoning dataset hash mismatch')
    sources = {r['id']: r for r in train}
    seen, result = set(), []
    for row in records:
        key = row['id']
        if key in seen or key not in sources:
            raise ValueError('Duplicate or non-training teacher ID')
        seen.add(key)
        original = sources[key]
        if any(row.get(k) != original[k] for k in ('question', 'image', 'answer')):
            raise ValueError(f'Teacher source row changed: {key}')
        if not isinstance(row.get('reasoning'), str) or not row['reasoning'].strip():
            raise ValueError(f'Missing reasoning: {key}')
        result.append({**original, 'reasoning': row['reasoning']})
    coverage = len(result) / len(train)
    minimum = cfg['data'].get('reasoning_min_coverage', 0.95)
    if not 0 < minimum <= 1 or coverage < minimum or len(result) != report['accepted']:
        raise ValueError(f'Reasoning coverage {coverage:.1%} below requirement {minimum:.1%}, or count mismatch')
    print(f'Reasoning train: {len(result)}/{len(train)} ({coverage:.1%}); rejected samples excluded', flush=True)
    return result
