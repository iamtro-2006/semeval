"""Precision-selectable runner around the unchanged organizer scorer."""
import argparse
import importlib.util
import json
import sys

from .common import ROOT


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--dtype', choices=['float16', 'float32'], default='float16')
    args, remaining = parser.parse_known_args()
    vendor = ROOT / 'vendor/organizer/scorer/score.py'
    spec = importlib.util.spec_from_file_location('mmcqa_organizer_score', vendor)
    organizer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(organizer)
    import torch
    if args.dtype == 'float16' and not torch.cuda.is_available():
        raise RuntimeError('FP16 scorer requires CUDA. Use --dtype float32 for CPU.')
    original = organizer.backbone.get_scorer

    def scorer_with_dtype():
        scorer = original()
        if not getattr(scorer, '_mmcqa_precision_applied', False):
            scorer._model.to(dtype=getattr(torch, args.dtype))
            scorer._model.eval()
            scorer._mmcqa_precision_applied = True
        return scorer

    organizer.backbone.get_scorer = scorer_with_dtype
    sys.argv = [str(vendor), *remaining]
    print(f'[mmcqa] Scorer precision: {args.dtype}', flush=True)
    organizer.main()
    if '--out' in remaining:
        from pathlib import Path
        output = Path(remaining[remaining.index('--out') + 1])
        report = json.loads(output.read_text(encoding='utf-8'))
        report['execution_precision'] = args.dtype
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
