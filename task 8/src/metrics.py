import argparse
import csv
import json
from pathlib import Path

IDENTITIES = ('l', 'g', 'b', 't', 'q', 'i', 'a', 'nb', 'lgbtqia+')
SCOPES = ('individual', 'group')


def parse_target(value):
    """Parse a Task C label strictly. 'none' is not a Task C gold label.

    Identity ordering and duplicate codes do not change the identity set.
    Does not map aliases, repair model JSON, or silently remove unknown codes.
    """
    if not isinstance(value, str):
        raise ValueError(f'Target must be a string: {value!r}')
    parts = value.strip().split('_', 1)
    if len(parts) != 2 or parts[0] not in SCOPES:
        raise ValueError(f'Invalid target: {value!r}')
    identities = frozenset(v.strip() for v in parts[1].split(','))
    if not identities or not identities <= set(IDENTITIES):
        raise ValueError(f'Invalid identities: {value!r}')
    return parts[0], identities


def _ratio(numerator, denominator):
    # Fixed, explicit zero_division=0 convention.
    return numerator / denominator if denominator else 0.0


def _class_metrics(tp, fp, fn):
    return {
        'tp': tp, 'fp': fp, 'fn': fn,
        'precision': _ratio(tp, tp + fp),
        'recall': _ratio(tp, tp + fn),
        'f1': _ratio(2 * tp, 2 * tp + fp + fn),
        'support': tp + fn,
        'predicted_count': tp + fp,
        'f1_defined': bool(2 * tp + fp + fn),
    }


def _prepare_targets(gold_targets, predicted_targets):
    """Shared validation; invalid predictions become (None, empty set)."""
    gold_values, pred_values = list(gold_targets), list(predicted_targets)
    if len(gold_values) != len(pred_values):
        raise ValueError('Gold and prediction lengths differ.')
    if not gold_values:
        raise ValueError('Cannot evaluate an empty dataset.')
    gold, pred, invalid_indices = [], [], []
    for i, (g, p) in enumerate(zip(gold_values, pred_values)):
        try:
            gold.append(parse_target(g))
        except ValueError as exc:
            raise ValueError(f'Invalid gold at index {i}: {g!r}') from exc
        try:
            pred.append(parse_target(p))
        except ValueError:
            pred.append((None, frozenset()))
            invalid_indices.append(i)
    return gold, pred, invalid_indices


def _identity_counts(gold, pred, identity):
    return (
        sum(identity in gi and identity in pi for (_, gi), (_, pi) in zip(gold, pred)),
        sum(identity not in gi and identity in pi for (_, gi), (_, pi) in zip(gold, pred)),
        sum(identity in gi and identity not in pi for (_, gi), (_, pi) in zip(gold, pred)),
    )


def _check_identity(identity):
    if identity not in IDENTITIES:
        raise ValueError(f'Unknown identity: {identity!r}')


def identity_precision(gold_targets, predicted_targets, identity):
    """F: precision for one identity; zero denominator -> 0."""
    _check_identity(identity)
    gold, pred, _ = _prepare_targets(gold_targets, predicted_targets)
    tp, fp, _ = _identity_counts(gold, pred, identity)
    return _ratio(tp, tp + fp)


def identity_recall(gold_targets, predicted_targets, identity):
    """F: recall for one identity; zero denominator -> 0."""
    _check_identity(identity)
    gold, pred, _ = _prepare_targets(gold_targets, predicted_targets)
    tp, _, fn = _identity_counts(gold, pred, identity)
    return _ratio(tp, tp + fn)


def identity_f1(gold_targets, predicted_targets, identity):
    """F: F1 for one identity; undefined F1 -> 0."""
    _check_identity(identity)
    gold, pred, _ = _prepare_targets(gold_targets, predicted_targets)
    tp, fp, fn = _identity_counts(gold, pred, identity)
    return _ratio(2 * tp, 2 * tp + fp + fn)


def identity_support(gold_targets, predicted_targets, identity):
    """F: number of gold samples containing this identity."""
    _check_identity(identity)
    gold, pred, _ = _prepare_targets(gold_targets, predicted_targets)
    tp, _, fn = _identity_counts(gold, pred, identity)
    return tp + fn


def identity_predicted_count(gold_targets, predicted_targets, identity):
    """F: number of valid predictions containing this identity."""
    _check_identity(identity)
    gold, pred, _ = _prepare_targets(gold_targets, predicted_targets)
    tp, fp, _ = _identity_counts(gold, pred, identity)
    return tp + fp


def identity_per_class(gold_targets, predicted_targets):
    """F: complete per-identity report, including TP/FP/FN and support."""
    gold, pred, _ = _prepare_targets(gold_targets, predicted_targets)
    return {k: _class_metrics(*_identity_counts(gold, pred, k)) for k in IDENTITIES}


def identity_macro_f1_gold_support(gold_targets, predicted_targets):
    """A: average F1 only over identities with gold support > 0.

    Absent-gold false positives are not included here; report Micro-F1 too.
    """
    report = identity_per_class(gold_targets, predicted_targets)
    scores = [v['f1'] for v in report.values() if v['support'] > 0]
    return sum(scores) / len(scores)


def identity_macro_f1_fixed_9(gold_targets, predicted_targets):
    """B: average over all nine fixed identities; undefined F1 -> 0."""
    report = identity_per_class(gold_targets, predicted_targets)
    return sum(v['f1'] for v in report.values()) / len(IDENTITIES)


def identity_micro_f1(gold_targets, predicted_targets):
    """C: pooled TP/FP/FN across all nine identities."""
    report = identity_per_class(gold_targets, predicted_targets)
    tp = sum(v['tp'] for v in report.values())
    fp = sum(v['fp'] for v in report.values())
    fn = sum(v['fn'] for v in report.values())
    return _ratio(2 * tp, 2 * tp + fp + fn)


def scope_per_class(gold_targets, predicted_targets):
    """Per-scope precision/recall/F1; invalid predictions cause a false negative."""
    gold, pred, _ = _prepare_targets(gold_targets, predicted_targets)
    report = {}
    for scope in SCOPES:
        tp = sum(gs == scope and ps == scope for (gs, _), (ps, _) in zip(gold, pred))
        fp = sum(gs != scope and ps == scope for (gs, _), (ps, _) in zip(gold, pred))
        fn = sum(gs == scope and ps != scope for (gs, _), (ps, _) in zip(gold, pred))
        report[scope] = _class_metrics(tp, fp, fn)
    return report


def scope_macro_f1(gold_targets, predicted_targets):
    """D: average F1 over fixed scopes individual and group."""
    report = scope_per_class(gold_targets, predicted_targets)
    return sum(v['f1'] for v in report.values()) / len(SCOPES)


def target_exact_match(gold_targets, predicted_targets):
    """E: fraction with correct scope AND the entire identity set (order ignored)."""
    gold, pred, _ = _prepare_targets(gold_targets, predicted_targets)
    return sum(g == p for g, p in zip(gold, pred)) / len(gold)


def scope_accuracy(gold_targets, predicted_targets):
    """Fraction with correct scope, irrespective of identities."""
    gold, pred, _ = _prepare_targets(gold_targets, predicted_targets)
    return sum(g[0] == p[0] for g, p in zip(gold, pred)) / len(gold)


def invalid_rate(gold_targets, predicted_targets):
    """Fraction of predictions that are not valid Task C target strings."""
    gold, _, invalid_indices = _prepare_targets(gold_targets, predicted_targets)
    return len(invalid_indices) / len(gold)


def evaluate_targets(gold_targets, predicted_targets):
    """Call the independent metric functions and retain the original report schema.

    Each public metric accepts aligned sequences of target strings. All scores
    are in [0,1]. Invalid gold raises; invalid predictions are never dropped.
    Inputs can be lists, pandas Series, or generators. Materialize generators
    before calling multiple independent metrics on the same data.
    """
    gold_values, pred_values = list(gold_targets), list(predicted_targets)
    gold, pred, invalid_indices = _prepare_targets(gold_values, pred_values)
    report = identity_per_class(gold_values, pred_values)
    supported = [k for k in IDENTITIES if report[k]['support'] > 0]
    return {
        'n': len(gold),
        'identity_macro_f1_gold_support': identity_macro_f1_gold_support(gold_values, pred_values),
        'identity_macro_f1_fixed_9': identity_macro_f1_fixed_9(gold_values, pred_values),
        'identity_micro_f1': identity_micro_f1(gold_values, pred_values),
        'scope_macro_f1': scope_macro_f1(gold_values, pred_values),
        'target_exact_match': target_exact_match(gold_values, pred_values),
        'identity_per_class': report,
        'scope_per_class': scope_per_class(gold_values, pred_values),
        'gold_supported_identities': supported,
        'gold_supported_identity_count': len(supported),
        'scope_accuracy': scope_accuracy(gold_values, pred_values),
        'exact_match_count': sum(g == p for g, p in zip(gold, pred)),
        'invalid_count': len(invalid_indices),
        'invalid_rate': invalid_rate(gold_values, pred_values),
        'invalid_indices': invalid_indices,
        'identity_totals': {key: sum(v[key] for v in report.values()) for key in ('tp', 'fp', 'fn')},
        'zero_division': 0,
    }


def evaluate_csv(path, gold_col='gold', pred_col='pred_target',
                 lang_col='lang', delimiter=None, exclude_none=False):
    """Read aligned rows from one CSV/TSV; report pooled and per-language scores.

    exclude_none defaults False to prevent silently dropping non-Task-C rows.
    Turn it on explicitly when input contains gold='none'; exclusion uses only
    gold and the excluded row count is recorded. No predictions are generated.
    """
    path = Path(path)
    delimiter = delimiter or ('\t' if path.suffix.lower() == '.tsv' else ',')
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        columns = reader.fieldnames or []
        missing = {gold_col, pred_col} - set(columns)
        if missing:
            raise ValueError(f'Missing columns {sorted(missing)}; found {columns}')
        rows = list(reader)
    original_n = len(rows)
    if exclude_none:
        rows = [r for r in rows if r[gold_col].strip() != 'none']
    overall = evaluate_targets([r[gold_col] for r in rows], [r[pred_col] for r in rows])
    per_language = {}
    if lang_col in columns:
        for lang in sorted({r[lang_col] for r in rows}):
            subset = [r for r in rows if r[lang_col] == lang]
            per_language[lang] = evaluate_targets([r[gold_col] for r in subset], [r[pred_col] for r in subset])
    return {'source': str(path), 'input_rows': original_n,
            'excluded_gold_none_rows': original_n - len(rows),
            'overall': overall, 'per_language': per_language}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('predictions', help='CSV/TSV containing aligned gold/prediction columns')
    parser.add_argument('--gold-col', default='gold')
    parser.add_argument('--pred-col', default='pred_target', help='Use strict_target/normalized_target to evaluate first outputs separately')
    parser.add_argument('--lang-col', default='lang')
    parser.add_argument('--exclude-none', action='store_true', help='Explicitly select Task C rows by excluding gold=none')
    parser.add_argument('--output', help='Optional JSON report path')
    args = parser.parse_args()
    report = evaluate_csv(args.predictions, args.gold_col, args.pred_col, args.lang_col,
                          exclude_none=args.exclude_none)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(payload + '\n', encoding='utf-8')
    print(payload)


if __name__ == '__main__':
    main()
