"""Temporary EN split, preserving video/duplicate-comment groups and label balance.

Uses only stdlib + numpy; never loads a language model. Existing TSVs are refused.
"""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/raw/StereoQueerEval_EN_training.tsv'
DEST = ROOT / 'data/split/en'
REPORT = ROOT / 'outputs/en_split'
IDENTITIES = ('l', 'g', 'b', 't', 'q', 'i', 'a', 'nb', 'lgbtqia+')
SPLITS = ('train', 'dev', 'test')
RATIOS = np.array([.8, .1, .1])


def norm(text):
    return ' '.join(text.casefold().split())


def video(row):
    return norm(row['yt_title']) + '\x1f' + norm(row['yt_description'])


def groups_from_rows(rows):
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen_video, seen_comment = {}, {}
    for i, row in enumerate(rows):
        for mapping, key in ((seen_video, video(row)), (seen_comment, norm(row['yt_comment']))):
            if key:
                if key in mapping:
                    parent[find(i)] = find(mapping[key])
                else:
                    mapping[key] = i
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[find(i)].append(row)
    return sorted(groups.values(), key=lambda g: min(r['StereoQueerEval_id'] for r in g))


def row_features(row):
    result = Counter({'size': 1, 'target:' + row['target']: 1,
                      'hate:' + row['hate_speech']: 1, 'stereotype:' + row['stereotype']: 1})
    if row['target'] == 'none':
        result['scope:none'] = 1
    else:
        scope, identities = row['target'].split('_', 1)
        result['scope:' + scope] = 1
        for identity in set(identities.split(',')):
            result['identity:' + identity] = 1
    return result


def balanced_assignment(groups):
    counters = [sum((row_features(r) for r in g), Counter()) for g in groups]
    keys = ['size'] + sorted(set().union(*(c.keys() for c in counters)) - {'size'})
    matrix = np.array([[c[k] for k in keys] for c in counters], dtype=float)
    totals = matrix.sum(axis=0)
    desired = RATIOS[:, None] * totals[None, :]
    desired[:, 0] = [int(totals[0]) - 2 * round(totals[0] * .1),
                     round(totals[0] * .1), round(totals[0] * .1)]
    locked = set()
    rare_notes = []
    for identity in IDENTITIES:
        column = keys.index('identity:' + identity)
        present = np.flatnonzero(matrix[:, column])
        if len(present) == 1:
            locked.add(int(present[0]))
            rare_notes.append({'identity': identity, 'support': int(totals[column]),
                               'independent_groups': 1, 'assigned_to': 'train'})
    # Labels present exclusively in locked components cannot be split proportionally.
    for column in range(1, len(keys)):
        present = set(map(int, np.flatnonzero(matrix[:, column])))
        if present and present <= locked:
            desired[:, column] = [totals[column], 0, 0]
    weights = np.array([150.0 if k == 'size' else
                        4.0 if k.startswith('scope:') else
                        3.0 if k.startswith('identity:') else
                        1.5 if k.startswith('hate:') else .6 for k in keys])
    scale = weights / np.maximum(totals, 8)
    coverage_columns = [keys.index('identity:' + identity) for identity in IDENTITIES
                        if sum(matrix[:, keys.index('identity:' + identity)] > 0) >= 3]
    def objective(counts):
        return float(np.sum((counts - desired) ** 2 * scale) +
                     12 * np.sum(counts[:, coverage_columns] == 0))
    movable = [i for i in range(len(groups)) if i not in locked]
    best = None
    for restart in range(12):
        rng = random.Random(42 + restart)
        assignment = np.zeros(len(groups), dtype=int)
        for i in movable:
            x = rng.random()
            assignment[i] = 0 if x < .8 else 1 if x < .9 else 2
        counts = np.stack([matrix[assignment == split].sum(axis=0) for split in range(3)])
        current = objective(counts)
        for step in range(16000):
            i = rng.choice(movable)
            source = int(assignment[i])
            destination = rng.choice([s for s in range(3) if s != source])
            swap = rng.random() < .7
            j = rng.choice(movable) if swap else None
            if swap and int(assignment[j]) != destination:
                continue
            delta = matrix[i] - (matrix[j] if swap else 0)
            trial = counts.copy()
            trial[source] -= delta
            trial[destination] += delta
            score = objective(trial)
            temperature = .12 * max(0, 1 - step / 11000)
            if score < current or (temperature and rng.random() < np.exp(min(0, (current - score) / temperature))):
                counts, current = trial, score
                assignment[i] = destination
                if swap:
                    assignment[j] = source
        if best is None or current < best[0]:
            best = (current, assignment.copy(), counts.copy())
    score, assignment, counts = best
    return assignment, {'seed': 42, 'ratios': dict(zip(SPLITS, map(float, RATIOS))),
                        'target_sizes': dict(zip(SPLITS, map(int, desired[:, 0]))),
                        'objective': score, 'group_count': len(groups),
                        'largest_group_rows': max(map(len, groups)),
                        'rare_identity_constraints': rare_notes,
                        'method': 'Connected video-proxy/duplicate-comment groups; deterministic multi-start label-balance optimization; singleton identity groups retained in train.'}


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    if any((DEST / f'{s}.tsv').exists() for s in SPLITS):
        raise FileExistsError('Existing split TSVs will not be overwritten.')
    with SOURCE.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream, delimiter='\t')
        fields, rows = reader.fieldnames, list(reader)
    groups = groups_from_rows(rows)
    assignment, info = balanced_assignment(groups)
    subsets = {s: [] for s in SPLITS}
    for index, group in enumerate(groups):
        subsets[SPLITS[int(assignment[index])]].extend(group)
    for subset in subsets.values():
        subset.sort(key=lambda r: r['StereoQueerEval_id'])
    assert len({r['StereoQueerEval_id'] for subset in subsets.values() for r in subset}) == len(rows)
    assert sum(map(len, subsets.values())) == len(rows)
    overlap = {}
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1:]:
            result = {
                'ids': len({r['StereoQueerEval_id'] for r in subsets[left]} & {r['StereoQueerEval_id'] for r in subsets[right]}),
                'video_proxies': len({video(r) for r in subsets[left]} & {video(r) for r in subsets[right]}),
                'normalized_comments': len({norm(r['yt_comment']) for r in subsets[left]} & {norm(r['yt_comment']) for r in subsets[right]}),
            }
            assert not any(result.values()), result
            overlap[left + '/' + right] = result
    for split, subset in subsets.items():
        with (DEST / f'{split}.tsv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, delimiter='\t')
            writer.writeheader()
            writer.writerows(subset)
    distributions = {}
    for split, subset in subsets.items():
        distributions[split] = {'rows': len(subset), 'features': dict(sum((row_features(r) for r in subset), Counter()))}
    info.update({'source': str(SOURCE), 'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                 'temporary_local_split': True, 'rows': len(rows),
                 'checks': {'all_rows_preserved': True, 'all_columns_preserved': True,
                            'overlaps': overlap}, 'distributions': distributions,
                 'video_proxy_caveat': 'Title+description proxy cannot detect renamed videos or near-duplicates.'})
    REPORT.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(info, ensure_ascii=False, indent=2) + '\n'
    (DEST / 'split_manifest.json').write_text(payload, encoding='utf-8')
    (REPORT / 'distribution.json').write_text(payload, encoding='utf-8')
    with (REPORT / 'label_distribution.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['label_type', 'label', 'train_count', 'dev_count', 'test_count',
                         'train_pct_of_rows', 'dev_pct_of_rows', 'test_pct_of_rows'])
        keys = sorted(set().union(*(v['features'] for v in distributions.values())))
        for key in keys:
            if key == 'size':
                continue
            counts = [distributions[s]['features'].get(key, 0) for s in SPLITS]
            percents = [round(100 * count / distributions[s]['rows'], 4) for count, s in zip(counts, SPLITS)]
            kind, label = key.split(':', 1)
            writer.writerow([kind, label, *counts, *percents])
    print(json.dumps({'sizes': {s: len(r) for s, r in subsets.items()},
                      'identity_counts': {s: {i: distributions[s]['features'].get('identity:' + i, 0) for i in IDENTITIES} for s in SPLITS},
                      'constraints': info['rare_identity_constraints'], 'overlaps': overlap}, indent=2))


if __name__ == '__main__':
    main()
