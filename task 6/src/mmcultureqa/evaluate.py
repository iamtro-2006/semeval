"""Generate and officially score a completed adapter on its held-out validation set."""
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
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    directory = path(args.run_dir)
    cfg = json.loads((directory / 'run.json').read_text(encoding='utf-8'))
    split = cfg['data']['eval_split']
    if split != 'validation':
        raise ValueError('Final validation evaluation requires data.eval_split=validation')
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
    name = 'validation' if len(selected) == len(validation) else f'validation_subset_{len(selected)}'
    output = directory / name
    output.mkdir(parents=True, exist_ok=True)
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
    if args.backend:
        cfg['inference']['backend'] = args.backend
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
    subprocess.run([sys.executable, str(ROOT / 'vendor/organizer/scorer/score.py'),
                    '--gold', str(gold), '--pred', str(predictions), '--lang', 'en',
                    '--out', str(scores)], check=True, cwd=ROOT)
    summary = {'split': split, 'evaluated_rows': len(selected), 'total_validation_rows': len(validation),
               'full_validation': len(selected) == len(validation), 'scores_file': str(scores)}
    (output / 'evaluation.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print(scores.read_text(encoding='utf-8'))


if __name__ == '__main__':
    main()
