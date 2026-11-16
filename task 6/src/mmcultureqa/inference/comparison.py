"""Comparable, exportable zero-shot vs zero-shot CoT validation reports."""
import csv
import hashlib
import html
import json
import math

from ..common import path

METRICS = ('BERTScore-F1', 'BERTScore-P', 'BERTScore-R', 'BLEU', 'ROUGE-L', 'Coverage')


def build_comparison(run_dir, baseline_name='validation'):
    directory = path(run_dir)
    folders = [directory / baseline_name, directory / (baseline_name + '_cot')]
    evaluations, scores = [], []
    for folder in folders:
        evaluation = json.loads((folder / 'evaluation.json').read_text(encoding='utf-8'))
        predictions = folder / 'predictions.jsonl'
        if hashlib.sha256(predictions.read_bytes()).hexdigest() != evaluation.get('predictions_sha256'):
            raise ValueError(f'Predictions changed after evaluation: {folder}')
        evaluations.append(evaluation)
        scores.append(json.loads((folder / 'scores.json').read_text(encoding='utf-8')))
    first, second = evaluations
    if first.get('prompt_mode') != 'saved' or first.get('reasoning_enabled') is not False or second.get('prompt_mode') != 'cot':
        raise ValueError('Comparison requires direct-answer baseline and zero-shot CoT evaluation')
    for key in ('model_signature', 'gold_sha256', 'evaluated_rows', 'scorer_dtype', 'load_in_4bit', 'seed'):
        if key not in first or first[key] != second.get(key):
            raise ValueError(f'Evaluations are not comparable: {key}')
    if not first['model_signature'] or not first['gold_sha256']:
        raise ValueError('Missing checkpoint/data identity; rerun final validation with the updated evaluator')
    for key in ('backend', 'do_sample'):
        if first['generation'].get(key) != second['generation'].get(key):
            raise ValueError(f'Decoding differs between evaluations: {key}')
    rows = []
    for metric in METRICS:
        values = [float(score['submission_result'][metric]) for score in scores]
        if not all(math.isfinite(v) for v in values):
            raise ValueError(f'Invalid metric: {metric}')
        # BLEU is already 0–100; other organizer metrics are returned as fractions.
        scaled = [v if metric == 'BLEU' else v * 100 for v in values]
        rows.append({'metric': metric, 'zero_shot': scaled[0], 'zero_shot_cot': scaled[1],
                     'delta_points': scaled[1] - scaled[0]})
    report = {'rows': rows, 'evaluated_rows': first['evaluated_rows'],
              'model_signature': first['model_signature'], 'gold_sha256': first['gold_sha256'],
              'empty_answers': [e['empty_answers'] for e in evaluations],
              'max_new_tokens': [e['generation']['max_new_tokens'] for e in evaluations],
              'scorer_dtype': first['scorer_dtype'],
              'note': 'Same SFT checkpoint and held-out rows. Zero-shot means no in-context demonstrations. '
                      'BERTScore, ROUGE-L and Coverage multiplied by 100; BLEU unchanged. '
                      'Delta is CoT minus direct answer, in points. This is not a significance test.'}
    (directory / 'prompt_comparison.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    with (directory / 'prompt_comparison.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=('metric', 'zero_shot', 'zero_shot_cot', 'delta_points'))
        writer.writeheader()
        writer.writerows(rows)
    table_rows = []
    for row in rows:
        delta = row['delta_points']
        color = '#147d42' if delta > 0 else '#b42318' if delta < 0 else '#667085'
        cells = ''.join(f'<td>{row[key]:.3f}</td>' for key in ('zero_shot', 'zero_shot_cot'))
        table_rows.append(f'<tr><th>{html.escape(row["metric"])}</th>{cells}'
                          f'<td style="color:{color};font-weight:700">{delta:+.3f}</td></tr>')
    page = f'''<!doctype html><html><head><meta charset="utf-8"><title>Prompt comparison</title>
<style>.mmcqa-comparison{{font:14px system-ui,sans-serif;color:#172b4d;max-width:960px;padding:24px;
border:1px solid #e4e7ec;border-radius:14px;background:#fff}}.mmcqa-comparison h2{{margin:0 0 8px}}
.mmcqa-comparison table{{width:100%;border-collapse:collapse;margin:20px 0}}
.mmcqa-comparison td,.mmcqa-comparison th{{padding:12px 16px;border-bottom:1px solid #eaecf0;text-align:right}}
.mmcqa-comparison th:first-child{{text-align:left}}.mmcqa-comparison thead{{background:#f2f4f7}}
.mmcqa-comparison small{{color:#475467;line-height:1.6}}</style></head><body>
<section class="mmcqa-comparison"><h2>Zero-shot vs Zero-shot CoT</h2>
<small>Same checkpoint · {report['evaluated_rows']:,} validation questions · {html.escape(report['scorer_dtype'])} scorer</small>
<table><thead><tr><th>Metric ↑</th><th>Zero-shot</th><th>Zero-shot CoT</th><th>Δ points</th></tr></thead>
<tbody>{''.join(table_rows)}</tbody></table>
<small>All metrics shown on a 0–100 scale. Δ = CoT − Zero-shot; higher is better.<br>
Empty answers: {report['empty_answers'][0]} / {report['empty_answers'][1]} (direct / CoT). CoT retries format errors once with sampling; persistent errors stop evaluation.<br>
Generation token budgets: {report['max_new_tokens'][0]} / {report['max_new_tokens'][1]} (direct / CoT).<br>
Zero-shot refers to prompting without demonstrations on this fine-tuned checkpoint. Differences are descriptive, not statistical significance.</small>
</section></body></html>'''
    output = directory / 'prompt_comparison.html'
    output.write_text(page, encoding='utf-8')
    return output
