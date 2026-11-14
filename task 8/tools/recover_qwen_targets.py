"""Recover Qwen target labels from saved model outputs and rescore without inference."""
import csv
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from inference import evaluate, extract_target  # noqa: E402

OUTPUT_DIR = ROOT / 'outputs' / 'Qwen2.5'
FILES = ('comment_title.tsv', 'comment_title_desc.tsv')


def process(path):
    backup_dir = OUTPUT_DIR / 'backups'
    backup = backup_dir / path.name
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream, delimiter='\t')
        columns = reader.fieldnames
        rows = list(reader)
    if backup.exists():
        with backup.open(encoding='utf-8-sig', newline='') as stream:
            original_rows = list(csv.DictReader(stream, delimiter='\t'))
        if len(original_rows) != len(rows) or any(
            old['StereoQueerEval_id'] != new['StereoQueerEval_id'] or
            old['raw_output'] != new['raw_output'] or old['gold'] != new['gold']
            for old, new in zip(original_rows, rows)
        ):
            raise ValueError(f'Current file differs from original model outputs: {path}')
    else:
        original_rows = [row.copy() for row in rows]
    if not rows or not columns or any(None in row for row in rows):
        raise ValueError(f'Malformed or empty TSV: {path}')
    required = {'StereoQueerEval_id', 'gold', 'pred_target', 'valid', 'error',
                'annotation_valid', 'annotation_error', 'evidence_json', 'raw_output'}
    if not required <= set(columns):
        raise ValueError(f'Missing columns in {path}: {sorted(required - set(columns))}')
    if len({row['StereoQueerEval_id'] for row in rows}) != len(rows):
        raise ValueError(f'Duplicate IDs in {path}')

    before = evaluate([{'target': row['gold']} for row in original_rows],
                      [row['pred_target'] for row in original_rows])
    recovered = 0
    for index, row in enumerate(rows):
        target, error = extract_target(row['raw_output'])
        if row['pred_target'] and row['pred_target'] != target:
            raise ValueError(f"Existing target differs from recovered output: {row['StereoQueerEval_id']}")
        if target and not original_rows[index]['pred_target']:
            recovered += 1
        row['pred_target'] = target
        row['valid'] = str(bool(target))
        row['error'] = error

    after = evaluate([{'target': row['gold']} for row in rows],
                     [row['pred_target'] for row in rows])
    backup_dir.mkdir(exist_ok=True)
    if not backup.exists():
        shutil.copy2(path, backup)
    with tempfile.NamedTemporaryFile('w', encoding='utf-8', newline='',
                                     dir=path.parent, prefix=path.stem + '.',
                                     suffix='.tmp', delete=False) as stream:
        temp_path = Path(stream.name)
        writer = csv.DictWriter(stream, fieldnames=columns, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(path)
    report = {'source': str(path), 'rows': len(rows), 'recovered_targets': recovered,
              'before': before, 'after': after}
    path.with_suffix('.metrics.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return report


def main():
    reports = [process(OUTPUT_DIR / name) for name in FILES]
    labels = (
        ('Rows', lambda r: str(r['rows'])),
        ('Recovered targets', lambda r: str(r['recovered_targets'])),
        ('Invalid targets before', lambda r: str(r['before']['invalid_count'])),
        ('Invalid targets after', lambda r: str(r['after']['invalid_count'])),
        ('Full target exact match', lambda r: f"{r['after']['full_target_exact_match']:.4f}"),
        ('Identity Macro-F1 (gold support)', lambda r: f"{r['after']['task_c_on_gold_active']['identity_macro_f1_gold_support']:.4f}"),
        ('Identity Macro-F1 (fixed 9)', lambda r: f"{r['after']['task_c_on_gold_active']['identity_macro_f1_fixed_9']:.4f}"),
        ('Identity Micro-F1', lambda r: f"{r['after']['task_c_on_gold_active']['identity_micro_f1']:.4f}"),
        ('Scope Macro-F1', lambda r: f"{r['after']['task_c_on_gold_active']['scope_macro_f1']:.4f}"),
        ('Scope accuracy', lambda r: f"{r['after']['task_c_on_gold_active']['scope_accuracy']:.4f}"),
        ('Task C target exact match', lambda r: f"{r['after']['task_c_on_gold_active']['target_exact_match']:.4f}"),
    )
    table = ['| Metric | comment_title | comment_title_desc |', '|---|---:|---:|']
    for label, getter in labels:
        table.append(f'| {label} | {getter(reports[0])} | {getter(reports[1])} |')
    text = '\n'.join(table) + '\n'
    (OUTPUT_DIR / 'scores.md').write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
