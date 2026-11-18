"""After training exits: merge, generate and score each saved checkpoint sequentially."""
import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

from .common import ROOT, path, read_jsonl
from .models.artifacts import require_merged_model


def fingerprint(cfg, adapter):
    files = [adapter / 'adapter_config.json'] + [adapter / name for name in
             ('adapter_model.safetensors', 'adapter_model.bin') if (adapter / name).is_file()]
    if len(files) < 2 or not files[0].is_file():
        raise FileNotFoundError(f'Incomplete checkpoint: {adapter}')
    manifest = path(cfg['data']['manifest_dir'])
    files += [manifest / (split + '.jsonl') for split in
              (cfg['data']['train_split'], cfg['data']['eval_split'])]
    files += [ROOT / 'configs/inference.yaml']
    digest = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode())
    for file in files:
        digest.update(file.name.encode())
        with file.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(chunk)
    return digest.hexdigest()


def f1_from_report(file):
    report = json.loads(file.read_text(encoding='utf-8'))
    values = [block['BERTScore-F1'] for group in report['result'] for block in group.values()
              if isinstance(block, dict) and 'BERTScore-F1' in block]
    if len(values) != 1 or not math.isfinite(float(values[0])):
        raise ValueError(f'Expected one finite validation F1: {file}')
    return float(values[0])


def select_checkpoint(run_dir):
    run = path(run_dir).resolve()
    cfg = json.loads((run / 'run.json').read_text(encoding='utf-8'))
    split = cfg['data']['eval_split']
    if split not in {'dev', 'validation'}:
        raise ValueError('Checkpoint selection requires dev (or legacy validation), never devtest/test')
    candidates = sorted([p for p in run.glob('checkpoint-*')
                         if p.is_dir() and re.fullmatch(r'checkpoint-\d+', p.name)],
                        key=lambda p: int(p.name.split('-')[-1]))
    if not candidates:
        raise FileNotFoundError('No checkpoint-N directories; retain checkpoints during training')
    total = len(read_jsonl(path(cfg['data']['manifest_dir']) / f'{split}.jsonl'))
    results = run / 'checkpoint_selection'
    results.mkdir(exist_ok=True)
    ranking = []

    def execute(*args):
        subprocess.run([sys.executable, '-u', '-m', *map(str, args)], cwd=ROOT, check=True)

    for index, adapter in enumerate(candidates, 1):
        if adapter.resolve().parent != run:
            raise ValueError('Checkpoint must be directly inside this run')
        identity = fingerprint(cfg, adapter)
        folder = results / adapter.name
        folder.mkdir(exist_ok=True)
        stamp = folder / 'candidate.json'
        scores = folder / 'validation/scores.json'
        evaluation = folder / 'validation/evaluation.json'
        cached = json.loads(stamp.read_text(encoding='utf-8')) if stamp.is_file() else {}
        print(f'[selection {index}/{len(candidates)}] {adapter.name}', flush=True)
        if not (cached.get('signature') == identity and scores.is_file() and evaluation.is_file()):
            (folder / 'run.json').write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding='utf-8')
            merged = folder / 'best_model'
            execute('mmcultureqa.export', '--run-dir', run, '--adapter', adapter, '--output', merged)
            require_merged_model(merged)
            try:
                # The SAME evaluate -> inference.predict -> load_vllm path/settings as notebook 04.
                # No training process, LoRA runtime or sleeping engine is present on the GPU.
                execute('mmcultureqa.evaluate', '--run-dir', folder, '--backend', 'vllm')
                summary = json.loads(evaluation.read_text(encoding='utf-8'))
                if (summary['split'] != split or not summary['full_validation'] or summary['evaluated_rows'] != total
                        or summary['model_artifact'] != 'merged_bf16'):
                    raise ValueError('Candidate was not evaluated as a merged model on full validation')
                f1_from_report(scores)
                stamp.write_text(json.dumps({'signature': identity}, indent=2), encoding='utf-8')
            finally:
                # Delete only this generated candidate model; keep all adapters and score reports.
                if merged.resolve().parent != folder.resolve() or not merged.resolve().is_relative_to(results):
                    raise ValueError('Unsafe temporary merged-model directory')
                shutil.rmtree(merged)
        f1 = f1_from_report(scores)
        ranking.append(dict(checkpoint=str(adapter), step=int(adapter.name.split('-')[-1]),
                            bertscore_f1=f1, scores_file=str(scores), signature=identity))
        (results / 'ranking.json').write_text(json.dumps(ranking, indent=2), encoding='utf-8')

    # Stable tie break: keep the earliest checkpoint at the same rounded organizer F1.
    winner = max(ranking, key=lambda item: (item['bertscore_f1'], -item['step']))
    selection = dict(stage='sft', eval_split=split, selection_method=f'post_training_merged_vllm_full_{split}',
        best_model_checkpoint=winner['checkpoint'], best_metric=winner['bertscore_f1'],
        metric_for_best_model=f'{split}_bertscore_f1', evaluated_rows=total, candidates=ranking)
    (run / 'selection.json').write_text(json.dumps(selection, indent=2), encoding='utf-8')
    # Only one final merged model remains on disk. Reload the winning adapter once more.
    execute('mmcultureqa.export', '--run-dir', run, '--adapter', winner['checkpoint'])
    require_merged_model(run / 'best_model')
    winning_report = results / Path(winner['checkpoint']).name / 'validation'
    shutil.copytree(winning_report, run / 'validation', dirs_exist_ok=True)
    print(f"Selected {winner['checkpoint']}: F1={winner['bertscore_f1']}", flush=True)
    return selection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    select_checkpoint(args.run_dir)


if __name__ == '__main__':
    main()
