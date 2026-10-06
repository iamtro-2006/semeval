"""One Transformers pipeline; reads supplied splits without creating new ones."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import time
import uuid

from metrics import IDENTITIES, evaluate_targets, parse_target

ROOT = Path(__file__).resolve().parents[1]
TEXT_FIELDS = ('yt_title', 'yt_description', 'yt_comment')
INPUT_VARIANTS = {
    'comment': ('yt_comment',),
    'comment_title': ('yt_comment', 'yt_title'),
    'comment_title_desc': ('yt_comment', 'yt_title', 'yt_description'),
}
ID_FIELD = 'StereoQueerEval_id'


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_tsv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream, delimiter='\t')
        if not {ID_FIELD, *TEXT_FIELDS} <= set(reader.fieldnames or []):
            raise ValueError('TSV must include ID, title, description, and comment.')
        rows = list(reader)
    if not rows:
        raise ValueError('Empty TSV.')
    seen = set()
    for row in rows:
        if None in row or any(row.get(k) is None for k in (ID_FIELD, *TEXT_FIELDS)):
            raise ValueError('Malformed TSV: check field quoting and column count.')
        if not row[ID_FIELD] or row[ID_FIELD] in seen:
            raise ValueError('Empty or duplicate ID.')
        seen.add(row[ID_FIELD])
        if row.get('target') and row['target'] != 'none':
            parse_target(row['target'])
    return rows


def load_prompt(path, mode):
    text = path.read_text(encoding='utf-8')
    blocks = dict(re.findall(r'^```(system|user|examples)\n(.*?)\n```', text, re.M | re.S))
    if not {'system', 'user'} <= set(blocks):
        raise ValueError('Prompt needs system and user blocks.')
    examples = json.loads(blocks['examples']) if mode == 'few_shot' else []
    return blocks, examples


def visible_input(row, variant):
    return {field: row[field] for field in INPUT_VARIANTS[variant]}


def validate_demonstrations(examples, train_rows, evaluation_rows, variant):
    """Verify actual train provenance, gold labels, visible fields and evidence."""
    train_by_id = {row[ID_FIELD]: row for row in train_rows}
    eval_ids = {row[ID_FIELD] for row in evaluation_rows}
    eval_videos = {(row['yt_title'].casefold().strip(), row['yt_description'].casefold().strip())
                   for row in evaluation_rows}
    eval_comments = {' '.join(row['yt_comment'].casefold().split()) for row in evaluation_rows}
    seen = set()
    for example in examples:
        source_id = example.get('source_id')
        if source_id in seen or source_id not in train_by_id:
            raise ValueError(f'Demo must match a unique training ID: {source_id}')
        seen.add(source_id)
        source = train_by_id[source_id]
        if example.get('source_split') != 'train':
            raise ValueError(f'Demo must declare source_split=train: {source_id}')
        if example['input'] != visible_input(source, variant):
            raise ValueError(f'Demo input differs from visible train fields: {source_id}')
        if example['output']['target'] != canonical_gold(source['target']):
            raise ValueError(f'Demo target differs from train gold: {source_id}')
        source_video = (source['yt_title'].casefold().strip(), source['yt_description'].casefold().strip())
        if (source_id in eval_ids or source_video in eval_videos or
                ' '.join(source['yt_comment'].casefold().split()) in eval_comments):
            raise ValueError(f'Demonstration overlaps evaluation data: {source_id}')
        _, _, error = validate_annotation(json.dumps(example['output']), example['input'])
        if error:
            raise ValueError(f'Invalid train evidence annotation {source_id}: {error}')


def make_messages(blocks, examples, row, language, variant):
    def user_message(sample):
        payload = json.dumps(visible_input(sample, variant), ensure_ascii=False)
        return blocks['user'].replace('<language>', language).replace('<input_json>', payload)
    messages = [{'role': 'system', 'content': blocks['system'].replace('<language>', language)}]
    for example in examples:
        messages.extend([
            {'role': 'user', 'content': user_message(example['input'])},
            {'role': 'assistant', 'content': json.dumps(example['output'], ensure_ascii=False)},
        ])
    messages.append({'role': 'user', 'content': user_message(row)})
    return messages


def validate_annotation(raw, row):
    """Validate exact source quotes and entity/label consistency; never repair."""
    def unique_keys(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError(f'Duplicate JSON key: {key}')
            obj[key] = value
        return obj

    def evidence(items, comment_only=False):
        if not isinstance(items, list) or not items:
            raise ValueError('Evidence must be a non-empty list.')
        for item in items:
            if not isinstance(item, dict) or set(item) != {'field', 'quote'}:
                raise ValueError('Evidence needs field and quote only.')
            field, quote = item['field'], item['quote']
            if field not in TEXT_FIELDS or field not in row or (comment_only and field != 'yt_comment'):
                raise ValueError('Invalid evidence source field.')
            if not isinstance(quote, str) or not quote.strip() or quote not in row[field]:
                raise ValueError('Evidence is not an exact source span.')

    try:
        obj = json.loads(raw.strip(), object_pairs_hook=unique_keys)
        if not isinstance(obj, dict) or set(obj) != {'entities', 'scope', 'identities', 'target'}:
            raise ValueError('Expected entities, scope, identities, target only.')
        entities, scope, identities = obj['entities'], obj['scope'], obj['identities']
        if not isinstance(entities, list):
            raise ValueError('entities must be a list.')
        if not isinstance(scope, dict) or set(scope) != {'label', 'pairs'}:
            raise ValueError('scope needs label and pairs.')
        if not isinstance(identities, dict) or set(identities) != set(IDENTITIES):
            raise ValueError('All nine identity keys are required.')
        if obj['target'] == 'none':
            if entities or scope != {'label': 'none', 'pairs': 'none'} or any(v != 'none' for v in identities.values()):
                raise ValueError('none requires empty entities and none scope/identities.')
            return 'none', obj, ''
        parsed_scope, codes = parse_target(obj['target'])
        names = set()
        for entity in entities:
            if not isinstance(entity, dict) or set(entity) != {'entity', 'reference_evidence', 'targeting_evidence'}:
                raise ValueError('Invalid entity structure.')
            name = entity['entity']
            if not isinstance(name, str) or not name.strip() or name in names:
                raise ValueError('Entity names must be non-empty and unique.')
            names.add(name)
            evidence(entity['reference_evidence'])
            evidence(entity['targeting_evidence'], comment_only=True)
        if not names or scope['label'] != parsed_scope:
            raise ValueError('Target scope must agree with extracted entity scope.')

        def pairs(items):
            if not isinstance(items, list) or not items:
                raise ValueError('Supported labels need non-empty entity/evidence pairs.')
            paired = set()
            for pair in items:
                if not isinstance(pair, dict) or set(pair) != {'entity', 'evidence'}:
                    raise ValueError('Pair needs entity and evidence.')
                if pair['entity'] not in names:
                    raise ValueError('Pair refers to an entity that was not extracted.')
                paired.add(pair['entity'])
                evidence(pair['evidence'])
            return paired

        scope_entities = pairs(scope['pairs'])
        supported, identity_entities = set(), set()
        for code, items in identities.items():
            if items != 'none':
                identity_entities.update(pairs(items))
                supported.add(code)
        if codes != supported or names != identity_entities or names != scope_entities:
            raise ValueError('Entities, scope pairs, identity pairs and target must agree.')
        canonical = parsed_scope + '_' + ','.join(k for k in IDENTITIES if k in codes)
        if canonical != obj['target']:
            raise ValueError('Target needs canonical order without duplicates or spaces.')
        return canonical, obj, ''
    except (ValueError, TypeError, AttributeError, KeyError) as exc:
        return '', None, str(exc)


def canonical_gold(value):
    if value == 'none':
        return value
    scope, codes = parse_target(value)
    return scope + '_' + ','.join(k for k in IDENTITIES if k in codes)


def evaluate(rows, predictions):
    if not all(r.get('target') for r in rows):
        return {'status': 'not_evaluated', 'reason': 'Gold target missing for one or more rows.', 'n': len(rows)}
    active = [i for i, r in enumerate(rows) if r['target'] != 'none']
    return {
        'status': 'evaluated', 'n': len(rows),
        'full_target_exact_match': sum(canonical_gold(r['target']) == p for r, p in zip(rows, predictions)) / len(rows),
        'invalid_count': sum(not p for p in predictions),
        'invalid_rate': sum(not p for p in predictions) / len(rows),
        'false_positive_target_on_gold_none': sum(r['target'] == 'none' and p not in ('none', '') for r, p in zip(rows, predictions)),
        'task_c_on_gold_active': evaluate_targets([rows[i]['target'] for i in active],
                                                 [predictions[i] for i in active]) if active else None,
        'note': 'Local metrics. Conditional Task C scores exclude gold=none. Literal evidence and consistency are checked; semantic evidence relevance still requires review.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='src/configs/inference.json')
    parser.add_argument('--input-variant', choices=INPUT_VARIANTS)
    parser.add_argument('--prompt-mode', choices=('zero_shot', 'few_shot'))
    parser.add_argument('--split', choices=('train', 'dev', 'test'))
    args = parser.parse_args()
    config = json.loads(project_path(args.config).read_text(encoding='utf-8'))
    if args.input_variant:
        config['input_variant'] = args.input_variant
    if args.prompt_mode:
        config['prompt']['mode'] = args.prompt_mode
    if args.split:
        config['data']['split'] = args.split
    variant = config['input_variant']
    if variant not in INPUT_VARIANTS:
        raise ValueError('input_variant must be comment, comment_title, or comment_title_desc.')
    data = config['data']
    path = project_path(data['files'][data['split']])
    if not path.is_file():
        raise FileNotFoundError(f'Provide the existing {data["split"]} TSV at {path}, or update its config path.')
    rows = read_tsv(path)
    mode = config['prompt']['mode']
    if mode not in ('zero_shot', 'few_shot'):
        raise ValueError('prompt.mode must be zero_shot or few_shot.')
    filename = 'zs_prompt.md' if mode == 'zero_shot' else 'fs_prompt.md'
    prompt_path = project_path(config['prompt']['directory']) / variant / filename
    blocks, examples = load_prompt(prompt_path, mode)
    train_path = None
    if mode == 'few_shot':
        shots = config['prompt']['shots']
        if not isinstance(shots, int) or shots < 1 or shots > len(examples):
            raise ValueError(f'prompt.shots must be between 1 and {len(examples)}.')
        examples = examples[:shots]
        train_path = project_path(data['files']['train'])
        validate_demonstrations(examples, read_tsv(train_path), rows, variant)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(config['generation']['seed'])
    torch.manual_seed(config['generation']['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['generation']['seed'])
    options = config['model']
    max_memory = options['max_memory']
    if max_memory is not None:
        # JSON object keys are strings; Accelerate expects integer GPU IDs.
        max_memory = {int(key) if key.isdigit() else key: value
                      for key, value in max_memory.items()}
    common = {'revision': options['revision'], 'local_files_only': options['local_files_only']}
    tokenizer = AutoTokenizer.from_pretrained(config['model_id'], **common)
    model = AutoModelForCausalLM.from_pretrained(
        config['model_id'], dtype=options['dtype'], device_map=options['device_map'],
        max_memory=max_memory, attn_implementation=options['attention_implementation'], **common)
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    generation = config['generation']
    if generation['max_input_tokens'] + generation['max_new_tokens'] > model.config.max_position_embeddings:
        raise ValueError('Configured input/output budget exceeds the checkpoint context length.')
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8]
    model_tag = re.sub(r'[^A-Za-z0-9._-]+', '_', config['model_id']).strip('_')
    output = (project_path(config['output_dir']) / model_tag / variant /
              mode / data['split'] / run_id)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'config': config, 'input_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'prompt_sha256': hashlib.sha256(prompt_path.read_bytes()).hexdigest(),
                'resolved_model_revision': getattr(model.config, '_commit_hash', None),
                'input_variant': variant, 'input_fields': list(INPUT_VARIANTS[variant]),
                'demo_source': 'actual train rows; evidence annotations manually curated' if examples else None,
                'demo_ids': [example['source_id'] for example in examples],
                'train_sha256': hashlib.sha256(train_path.read_bytes()).hexdigest() if train_path else None,
                'status': 'running', 'rows': len(rows)}
    def save_manifest():
        (output / 'run.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    save_manifest()
    predictions = []
    columns = [ID_FIELD, 'language', 'input_variant', 'prompt_mode', 'gold', 'pred_target', 'valid', 'error',
               'evidence_json', 'raw_output', 'input_tokens', 'output_tokens', 'latency_seconds']
    try:
        with (output / 'predictions.tsv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, delimiter='\t')
            writer.writeheader()
            for index, row in enumerate(rows):
                visible = visible_input(row, variant)
                messages = make_messages(blocks, examples, visible, data['language'], variant)
                inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                       return_dict=True, return_tensors='pt')
                count = inputs['input_ids'].shape[1]
                raw, target, annotation, error, elapsed, output_tokens = '', '', None, '', 0.0, 0
                if count > generation['max_input_tokens']:
                    error = 'Input exceeds configured budget; no text was truncated. Adjust config or prompt.'
                else:
                    inputs = inputs.to(model.get_input_embeddings().weight.device)
                    start = time.perf_counter()
                    with torch.inference_mode():
                        generated = model.generate(**inputs, do_sample=generation['do_sample'],
                                                   max_new_tokens=generation['max_new_tokens'],
                                                   pad_token_id=tokenizer.pad_token_id)[0, count:]
                    raw = tokenizer.decode(generated, skip_special_tokens=True)
                    elapsed, output_tokens = time.perf_counter() - start, len(generated)
                    target, annotation, error = validate_annotation(raw, visible)
                predictions.append(target)
                writer.writerow({ID_FIELD: row[ID_FIELD], 'language': data['language'],
                                 'input_variant': variant, 'prompt_mode': mode, 'gold': row.get('target', ''),
                                 'pred_target': target, 'valid': not bool(error), 'error': error,
                                 'evidence_json': json.dumps(annotation, ensure_ascii=False) if annotation else '',
                                 'raw_output': raw, 'input_tokens': count, 'output_tokens': output_tokens,
                                 'latency_seconds': elapsed})
                stream.flush()
                if (index + 1) % 25 == 0:
                    print(f'Annotated {index + 1}/{len(rows)}', flush=True)
        (output / 'metrics.json').write_text(json.dumps(evaluate(rows, predictions), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        manifest['status'] = 'complete'
    except Exception as exc:
        manifest.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        save_manifest()
    print(f'Output: {output}')


if __name__ == '__main__':
    main()
