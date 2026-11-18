"""Score on official dev (or an explicitly saved legacy validation split)."""
import argparse
import hashlib
import json
import subprocess
import sys

from .common import ROOT, path, read_jsonl, write_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--limit', type=int, default=None, help='Omit to score all validation rows')
    parser.add_argument('--backend', choices=['vllm', 'transformers'], default=None)
    parser.add_argument('--prompt-mode', choices=['saved', 'cot'], default='saved')
    parser.add_argument('--cot-max-new-tokens', type=int, default=1024)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    if args.cot_max_new_tokens < 1:
        parser.error('--cot-max-new-tokens must be positive')
    directory = path(args.run_dir)
    config_file = directory / 'run.json'
    if not config_file.is_file():
        config_file = directory / 'best_model/run_config.json'
    cfg = json.loads(config_file.read_text(encoding='utf-8'))
    split = cfg['data']['eval_split']
    if split not in {'dev', 'validation'}:
        raise ValueError('Evaluation requires labelled dev (or legacy validation), never devtest/test')
    manifest = path(cfg['data']['manifest_dir'])
    for filename in (f'{split}.jsonl', f'{cfg["data"]["train_split"]}.jsonl'):
        expected = cfg.get('manifests', {}).get(filename)
        if expected and hashlib.sha256((manifest / filename).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Manifest changed since training: {filename}')
    validation = read_jsonl(manifest / f'{split}.jsonl')
    train = read_jsonl(manifest / f'{cfg["data"]["train_split"]}.jsonl')
    if {r['id'] for r in validation} & {r['id'] for r in train}:
        raise ValueError('Train and validation IDs overlap')
    if {r['image'] for r in validation} & {r['image'] for r in train}:
        raise ValueError('Train and validation images overlap')
    # Training smoke limits must never silently limit the final validation set.
    selected = validation[:args.limit] if args.limit else validation
    if not selected or any(not r.get('answer') for r in selected):
        raise ValueError('Validation must contain labelled rows')
    # Keep the report-directory API; evaluation.json records the actual dataset split.
    name = 'validation' if len(selected) == len(validation) else f'validation_subset_{len(selected)}'
    if args.prompt_mode == 'cot':
        name += '_cot'
    output = directory / name
    output.mkdir(parents=True, exist_ok=True)
    merged = directory / 'best_model'
    if merged.exists():
        from .models.artifacts import configure_merged_model
        cfg = configure_merged_model(cfg, merged, load_in_4bit=False)
    else:
        adapter = directory / 'final'
        if not (adapter / 'adapter_config.json').is_file():
            raise FileNotFoundError(f'Missing trained adapter: {adapter}')
        cfg['model'].update(adapter=str(adapter), adapter_revision=None,
                            processor=str(adapter), processor_revision=None)
    cfg['data']['limit'] = args.limit
    predictions = output / 'predictions.jsonl'
    import yaml
    defaults = yaml.safe_load((ROOT / 'configs/inference.yaml').read_text(encoding='utf-8'))['inference']
    cfg['inference'] = {**defaults, 'split': split,
                        'max_new_tokens': cfg.get('monitor', {}).get('max_new_tokens', 256),
                        'do_sample': False, 'output': str(predictions), 'resume': True}
    if args.prompt_mode == 'cot':
        cfg['prompt_text'] = (ROOT / 'prompts/qa_cot.txt').read_text(encoding='utf-8')
        cfg['inference'].update(prompt_mode='cot', prompt_file=str(output / 'prompt.txt'),
                                max_new_tokens=args.cot_max_new_tokens, vllm_max_model_len=4096,
                                on_format_error='raise', format_retries=1)
    if args.backend:
        cfg['inference']['backend'] = args.backend
    elif cfg['model'].get('load_in_4bit'):
        cfg['inference']['backend'] = 'transformers'
    # JSON is valid YAML. Reuse the exact saved training prompt and model settings.
    frozen_prompt = output / 'prompt.txt'
    frozen_prompt.write_text(cfg['prompt_text'], encoding='utf-8')
    cfg['prompt_file'] = str(frozen_prompt)
    config_file = output / 'inference_config.json'
    config_file.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding='utf-8')
    gold = output / 'gold_en.jsonl'
    write_jsonl(gold, selected)
    # Separate processes release the VLM's GPU memory before mmBERT is loaded.
    subprocess.run([sys.executable, '-m', 'mmcultureqa.inference.predict',
                    '--config', str(config_file)], check=True, cwd=ROOT)
    predicted = read_jsonl(predictions)
    if len(predicted) != len(selected) or {r['id'] for r in predicted} != {r['id'] for r in selected}:
        raise ValueError('Predictions must cover exactly the selected validation IDs')
    scores = output / 'scores.json'
    subprocess.run([sys.executable, '-m', 'mmcultureqa.score',
                    '--dtype', cfg.get('monitor', {}).get('dtype', 'float16'),
                    '--gold', str(gold), '--pred', str(predictions), '--lang', 'en',
                    '--out', str(scores)], check=True, cwd=ROOT)
    summary = {'split': split, 'evaluated_rows': len(selected), 'total_validation_rows': len(validation),
               'full_validation': len(selected) == len(validation), 'scores_file': str(scores),
               'model_artifact': 'merged_bf16' if merged.exists() else 'adapter',
               'load_in_4bit': cfg['model'].get('load_in_4bit', False),
               'prompt_mode': args.prompt_mode,
               'reasoning_enabled': args.prompt_mode == 'cot' or cfg.get('reasoning', {}).get('enabled', False),
               'gold_sha256': hashlib.sha256(json.dumps(selected, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
               'predictions_sha256': hashlib.sha256(predictions.read_bytes()).hexdigest(),
               'model_signature': (json.loads((merged / 'merge.json').read_text(encoding='utf-8')).get('signature')
                                   if merged.exists() else None),
               'scorer_dtype': cfg.get('monitor', {}).get('dtype', 'float16'),
               'seed': cfg['training']['seed'],
               'generation': cfg['inference'],
               'empty_answers': sum(not str(r.get('answer') or '').strip() for r in predicted)}
    (output / 'evaluation.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print(scores.read_text(encoding='utf-8'))


if __name__ == '__main__':
    main()
