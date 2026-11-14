from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def path(value):
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def merge(left, right):
    out = copy.deepcopy(left)
    for key, value in right.items():
        out[key] = merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


def config():
    import yaml
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--set', action='append', default=[], metavar='KEY=YAML_VALUE')
    args = parser.parse_args()

    def read(filename):
        obj = yaml.safe_load(path(filename).read_text(encoding='utf-8'))
        parent = obj.pop('extends', None)
        return merge(read(parent), obj) if parent else obj

    cfg = read(args.config)
    for item in args.set:
        key, value = item.split('=', 1)
        target = cfg
        parts = key.split('.')
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = yaml.safe_load(value)
    return cfg


def read_jsonl(filename):
    with path(filename).open(encoding='utf-8-sig') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(filename, rows):
    target = path(filename)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def rows_for(cfg, split):
    rows = read_jsonl(path(cfg['data']['manifest_dir']) / f'{split}.jsonl')
    limit = cfg['data'].get('limit')
    return rows[:limit] if limit else rows


def messages(cfg, question):
    text = path(cfg['prompt_file']).read_text(encoding='utf-8').format(question=question)
    result = []
    if cfg.get('system_prompt'):
        result.append({'role': 'system', 'content': cfg['system_prompt']})
    result.append({'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': text}]})
    return result


def image(row):
    from PIL import Image, ImageOps
    with Image.open(path(row['image'])) as im:
        return ImageOps.exif_transpose(im).convert('RGB')


def provenance(cfg, folder):
    import importlib.metadata
    out = path(folder)
    out.mkdir(parents=True, exist_ok=True)
    report = copy.deepcopy(cfg)
    report['prompt_text'] = path(cfg['prompt_file']).read_text(encoding='utf-8')
    report['versions'] = {k: importlib.metadata.version(k) for k in ['torch', 'transformers', 'trl', 'peft', 'ms-swift']}
    try:
        report['versions']['vllm'] = importlib.metadata.version('vllm')
    except importlib.metadata.PackageNotFoundError:
        pass
    report['manifests'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in path(cfg['data']['manifest_dir']).glob('*.jsonl')}
    (out / 'run.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
