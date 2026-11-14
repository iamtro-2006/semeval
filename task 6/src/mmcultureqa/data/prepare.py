"""Download only English textual QA and shared MENA images; prepare portable manifests."""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath

from ..common import ROOT, path, read_jsonl, write_jsonl

REPO = 'QCRI/MMCQA-SemEval27'
REVISION = 'e850ede73e2f1ef3dde12e0210de67d107247ef0'
SPLITS = ('train', 'dev', 'devtest')


def download(root, revision):
    from huggingface_hub import snapshot_download
    patterns = ['README.md'] + [f'qa/mena/{s}_en.{ext}' for s in SPLITS for ext in ('jsonl', 'parquet')]
    patterns += [f'archives/images_mena_{s}.zip' for s in SPLITS]
    snapshot_download(REPO, repo_type='dataset', revision=revision, local_dir=root, allow_patterns=patterns)
    (root / 'REVISION').write_text(revision, encoding='utf-8')


def prepare(root, destination, validation_size, seed):
    from PIL import Image
    import random
    destination.mkdir(parents=True, exist_ok=True)
    report = {'source': REPO, 'revision': (root / 'REVISION').read_text(encoding='utf-8-sig').strip(),
              'seed': seed, 'splits': {}}
    datasets = {}
    all_ids, all_images = set(), set()
    for split in SPLITS:
        rows = read_jsonl(root / 'qa' / 'mena' / f'{split}_en.jsonl')
        ids = [r['id'] for r in rows]
        if len(ids) != len(set(ids)) or all_ids.intersection(ids):
            raise ValueError(f'Duplicate IDs within/across {split}')
        all_ids.update(ids)
        members = {r['image'] for r in rows}
        if all_images.intersection(members):
            raise ValueError(f'Images overlap across source splits: {split}')
        all_images.update(members)
        archive = root / 'archives' / f'images_mena_{split}.zip'
        sizes = []
        with zipfile.ZipFile(archive) as zf:
            missing = members - set(zf.namelist())
            if missing:
                raise ValueError(f'{archive}: missing {len(missing)} images')
            for member in sorted(members):
                rel = PurePosixPath(member)
                if rel.is_absolute() or '..' in rel.parts or rel.parts[0] != 'images':
                    raise ValueError(f'Unsafe image path: {member}')
                target = root / split / Path(*rel.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                payload = zf.read(member)  # checks CRC, including on subsequent runs
                if not target.exists() or target.stat().st_size != len(payload):
                    target.write_bytes(payload)
                with Image.open(target) as im:
                    sizes.append(im.size)
                    im.verify()
        converted = []
        for row in rows:
            if not isinstance(row.get('question'), str) or not row['question'].strip():
                raise ValueError(f'Missing question: {row["id"]}')
            if split != 'devtest' and not str(row.get('answer') or '').strip():
                raise ValueError(f'Missing label: {row["id"]}')
            target = root / split / row['image']
            converted.append({**row, 'image': target.relative_to(ROOT).as_posix()})
        datasets[split] = converted
        report['splits'][split] = {'rows': len(rows), 'fields': sorted(set().union(*(r.keys() for r in rows))),
                                  'labelled': sum(bool(r.get('answer')) for r in rows),
                                  'unique_images': len(members), 'min_width': min(w for w, h in sizes),
                                  'max_width': max(w for w, h in sizes), 'archive_bytes': archive.stat().st_size}
    train = datasets.pop('train')
    if not 0 < validation_size < len(train):
        raise ValueError('validation-size must be between 1 and train size - 1')
    # Group by image, so repeated views/questions on the same image cannot leak.
    groups = sorted({r['image'] for r in train})
    random.Random(seed).shuffle(groups)
    holdout = set(groups[:validation_size])
    datasets['train'] = [r for r in train if r['image'] not in holdout]
    datasets['validation'] = [r for r in train if r['image'] in holdout]
    for name, rows in datasets.items():
        write_jsonl(destination / f'{name}.jsonl', rows)
    report['prepared_counts'] = {k: len(v) for k, v in datasets.items()}
    report['manifest_sha256'] = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in destination.glob('*.jsonl')}
    (destination / 'inspection.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['download', 'prepare'])
    parser.add_argument('--root', default='data/raw')
    parser.add_argument('--destination', default='data/prepared')
    parser.add_argument('--revision', default=REVISION)
    parser.add_argument('--validation-size', type=int, default=1000, help='Number of image groups held out from train')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    root = path(args.root)
    if args.command == 'download':
        download(root, args.revision)
    else:
        prepare(root, path(args.destination), args.validation_size, args.seed)


if __name__ == '__main__':
    main()
