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


def checkpoint_load_options(model_id, backend, common):
    """Choose a GPTQ runtime without changing checkpoint quantization settings."""
    if backend is None:
        return {}
    if not isinstance(backend, str) or not backend.strip():
        raise ValueError('model.gptq_backend must be null or a non-empty backend name.')
    from transformers import AutoConfig

    checkpoint = AutoConfig.from_pretrained(model_id, **common)
    quantization = getattr(checkpoint, 'quantization_config', None)
    if hasattr(quantization, 'to_dict'):
        quantization = quantization.to_dict()
    if not isinstance(quantization, dict) or quantization.get('quant_method') != 'gptq':
        raise ValueError('model.gptq_backend requires a GPTQ checkpoint; use null for other models.')
    checkpoint.quantization_config = {**quantization, 'backend': backend.strip().lower()}
    return {'config': checkpoint}


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
    video_exceptions = [example for example in examples if example.get('allow_video_overlap') is True]
    if len(video_exceptions) > 1:
        raise ValueError('Only one demonstration may allow video overlap.')
    max_identities = max((len(parse_target(row['target'])[1]) for row in train_rows
                          if row.get('target') and row['target'] != 'none'), default=0)
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
        allow_video_overlap = example.get('allow_video_overlap') is True
        if allow_video_overlap and (source['target'] == 'none' or max_identities < 2 or
                                    len(parse_target(source['target'])[1]) != max_identities):
            raise ValueError('Video overlap is allowed only for a maximum-identity train demonstration.')
        source_video = (source['yt_title'].casefold().strip(), source['yt_description'].casefold().strip())
        if (source_id in eval_ids or
                ' '.join(source['yt_comment'].casefold().split()) in eval_comments):
            raise ValueError(f'Demonstration overlaps evaluation data: {source_id}')
        if source_video in eval_videos and not allow_video_overlap:
            raise ValueError(f'Demonstration overlaps an evaluation video: {source_id}')
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
    """Check the compact annotation for diagnostics; target scoring is separate."""
    def unique_keys(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError(f'Duplicate JSON key: {key}')
            obj[key] = value
        return obj

    def evidence(items):
        if not isinstance(items, list) or not items:
            raise ValueError('Evidence must be a non-empty list.')
        for item in items:
            if not isinstance(item, dict) or set(item) != {'field', 'quote'}:
                raise ValueError('Evidence needs field and quote only.')
            field, quote = item['field'], item['quote']
            if field not in TEXT_FIELDS or field not in row:
                raise ValueError('Invalid evidence source field.')
            if not isinstance(quote, str) or not quote.strip() or quote not in row[field]:
                raise ValueError('Evidence is not an exact source span.')

    try:
        obj = json.loads(raw.strip(), object_pairs_hook=unique_keys)
        if not isinstance(obj, dict) or set(obj) != {'scope', 'identity', 'target', 'reasoning'}:
            raise ValueError('Expected scope, identity, target, reasoning only.')
        scope, identities = obj['scope'], obj['identity']
        if not isinstance(scope, dict) or set(scope) != {'label', 'evidence'}:
            raise ValueError('scope needs label and evidence.')
        if not isinstance(identities, list):
            raise ValueError('identity must be a JSON array.')
        if not isinstance(obj['reasoning'], str) or not obj['reasoning'].strip():
            raise ValueError('reasoning must be a non-empty string.')
        if obj['target'] == 'none':
            if scope != {'label': 'none', 'evidence': []} or identities:
                raise ValueError('none requires none scope with empty evidence and identity arrays.')
            return 'none', obj, ''
        parsed_scope, codes = parse_target(obj['target'])
        if scope['label'] != parsed_scope:
            raise ValueError('Target scope must agree with scope.label.')
        evidence(scope['evidence'])
        if not any(item['field'] == 'yt_comment' for item in scope['evidence']):
            raise ValueError('scope evidence must include a quote from yt_comment.')
        supported = []
        for identity in identities:
            if not isinstance(identity, dict) or set(identity) != {'label', 'evidence'}:
                raise ValueError('Each identity needs label and evidence.')
            code = identity['label']
            if code not in IDENTITIES or code in supported:
                raise ValueError('Identity labels must be supported codes without duplicates.')
            evidence(identity['evidence'])
            supported.append(code)
        if set(supported) != codes:
            raise ValueError('identity labels and target codes must agree.')
        if supported != [code for code in IDENTITIES if code in codes]:
            raise ValueError('identity labels need canonical order.')
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


def extract_target(raw):
    """Read the predicted label independently of evidence/schema validation."""
    try:
        obj = json.loads(raw.strip())
        if not isinstance(obj, dict) or 'target' not in obj:
            raise ValueError('Output must be a JSON object with a target field.')
        return canonical_gold(obj['target']), ''
    except (ValueError, TypeError) as exc:
        return '', str(exc)


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
        'note': 'Metrics score the target label only. Evidence/schema errors do not invalidate a valid target. Conditional Task C scores exclude gold=none.',
    }


def print_metrics(report):
    """Display saved target metrics in compact tables for copying results."""
    print(f'\nEvaluation: {report["n"]} samples')
    if report['status'] != 'evaluated':
        print(f'Not evaluated: {report["reason"]}')
        return
    print('Scores: 0 to 1. Only target labels are scored.')
    print(f'{"Metric":<38} {"Score":>10}')
    print(f'{"Full target exact match (all rows)":<38} {report["full_target_exact_match"]:>10.4f}')
    print(f'{"Invalid target rate (all rows)":<38} {report["invalid_rate"]:>10.4f}')
    print(f'Invalid targets: {report["invalid_count"]}/{report["n"]}')
    print(f'False positive targets on gold=none: {report["false_positive_target_on_gold_none"]}')
    task = report['task_c_on_gold_active']
    if task is None:
        print('Task C: no gold-active samples (all gold targets are none).')
        return
    print(f'\nTask C: {task["n"]} samples, excluding gold=none')
    print(f'{"Metric":<38} {"Score":>10}')
    for label, key in (
        ('Identity Macro-F1 (gold support)', 'identity_macro_f1_gold_support'),
        ('Identity Macro-F1 (fixed 9 labels)', 'identity_macro_f1_fixed_9'),
        ('Identity Micro-F1', 'identity_micro_f1'),
        ('Scope Macro-F1', 'scope_macro_f1'),
        ('Scope accuracy', 'scope_accuracy'),
        ('Target exact match', 'target_exact_match'),
    ):
        print(f'{label:<38} {task[key]:>10.4f}')
    for title, key in (('Scope', 'scope_per_class'), ('Identity', 'identity_per_class')):
        print(f'\n{title} per class (Task C)')
        print(f'{"Label":<12} {"Precision":>10} {"Recall":>10} {"F1":>10} {"Support":>8}')
        for label, values in task[key].items():
            print(f'{label:<12} {values["precision"]:>10.4f} {values["recall"]:>10.4f} '
                  f'{values["f1"]:>10.4f} {values["support"]:>8}')


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
    from tqdm import tqdm
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
    load_options = checkpoint_load_options(config['model_id'], options.get('gptq_backend'), common)
    tokenizer = AutoTokenizer.from_pretrained(config['model_id'], **common)
    model = AutoModelForCausalLM.from_pretrained(
        config['model_id'], dtype=options['dtype'], device_map=options['device_map'],
        max_memory=max_memory, attn_implementation=options['attention_implementation'], **load_options, **common)
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
               'annotation_valid', 'annotation_error', 'evidence_json', 'raw_output',
               'input_tokens', 'output_tokens', 'latency_seconds']
    try:
        with (output / 'predictions.tsv').open('w', encoding='utf-8', newline='') as stream, \
                tqdm(rows, desc=f'Inference {data["split"]}', unit='sample',
                     dynamic_ncols=True, miniters=1) as progress:
            writer = csv.DictWriter(stream, fieldnames=columns, delimiter='\t')
            writer.writeheader()
            for row in progress:
                visible = visible_input(row, variant)
                messages = make_messages(blocks, examples, visible, data['language'], variant)
                inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                       return_dict=True, return_tensors='pt')
                count = inputs['input_ids'].shape[1]
                raw, target, annotation, error, elapsed, output_tokens = '', '', None, '', 0.0, 0
                annotation_error = ''
                if count > generation['max_input_tokens']:
                    error = 'Input exceeds configured budget; no text was truncated. Adjust config or prompt.'
                    annotation_error = error
                else:
                    inputs = inputs.to(model.get_input_embeddings().weight.device)
                    start = time.perf_counter()
                    with torch.inference_mode():
                        generated = model.generate(**inputs, do_sample=generation['do_sample'],
                                                   max_new_tokens=generation['max_new_tokens'],
                                                   pad_token_id=tokenizer.pad_token_id)[0, count:]
                    raw = tokenizer.decode(generated, skip_special_tokens=True)
                    elapsed, output_tokens = time.perf_counter() - start, len(generated)
                    target, error = extract_target(raw)
                    _, annotation, annotation_error = validate_annotation(raw, visible)
                predictions.append(target)
                writer.writerow({ID_FIELD: row[ID_FIELD], 'language': data['language'],
                                 'input_variant': variant, 'prompt_mode': mode, 'gold': row.get('target', ''),
                                 'pred_target': target, 'valid': not bool(error), 'error': error,
                                 'annotation_valid': not bool(annotation_error), 'annotation_error': annotation_error,
                                 'evidence_json': json.dumps(annotation, ensure_ascii=False) if annotation else '',
                                 'raw_output': raw, 'input_tokens': count, 'output_tokens': output_tokens,
                                 'latency_seconds': elapsed})
                stream.flush()
        metrics = evaluate(rows, predictions)
        (output / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        manifest['status'] = 'complete'
    except Exception as exc:
        manifest.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        save_manifest()
    print(f'\nModel: {config["model_id"]} | Input: {variant} | Prompt: {mode} | Split: {data["split"]}')
    print_metrics(metrics)
    print(f'\nOutput: {output}')


if __name__ == '__main__':
    main()
