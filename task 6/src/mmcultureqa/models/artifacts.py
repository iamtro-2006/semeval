"""Validate self-contained merged checkpoints without consulting Hugging Face."""
import copy
import json
from pathlib import Path

from ..common import path


def require_merged_model(folder):
    folder = path(folder).resolve()
    for name in ('config.json', 'tokenizer_config.json', 'preprocessor_config.json',
                 'run_config.json', 'prompt.txt', 'merge.json'):
        if not (folder / name).is_file():
            raise FileNotFoundError(f'Merged model is incomplete: {folder / name}')
    if not any((folder / name).is_file() for name in ('tokenizer.json', 'vocab.json', 'tokenizer.model')):
        raise FileNotFoundError(f'Tokenizer vocabulary missing: {folder}')
    index = folder / 'model.safetensors.index.json'
    if index.is_file():
        weights = set(json.loads(index.read_text(encoding='utf-8'))['weight_map'].values())
    else:
        weights = {'model.safetensors'}
    if not weights:
        raise ValueError('Merged checkpoint has an empty weight map')
    for name in weights:
        file = (folder / name).resolve()
        if not file.is_relative_to(folder) or not file.is_file() or file.stat().st_size == 0:
            raise FileNotFoundError(f'Merged weight shard missing or invalid: {name}')
    config = json.loads((folder / 'config.json').read_text(encoding='utf-8'))
    if config.get('quantization_config'):
        raise ValueError('best_model must contain full BF16 weights, not a quantized base')
    return folder


def configure_merged_model(cfg, folder, load_in_4bit=False):
    folder = require_merged_model(folder)
    cfg = copy.deepcopy(cfg)
    saved = json.loads((folder / 'run_config.json').read_text(encoding='utf-8'))
    cfg['model'].update(name=str(folder), revision=None, adapter=None, adapter_revision=None,
                        processor=str(folder), processor_revision=None, dtype='bfloat16',
                        load_in_4bit=load_in_4bit, local_files_only=True, merged_model=True)
    cfg['prompt_file'] = str(folder / 'prompt.txt')
    cfg['system_prompt'] = saved.get('system_prompt')
    for key in ('model_type', 'enable_thinking'):
        if key in saved['model']:
            cfg['model'][key] = saved['model'][key]
        else:
            cfg['model'].pop(key, None)
    # A distilled model must retain its output contract when notebook 04 reloads it.
    cfg['reasoning'] = saved.get('reasoning', {'enabled': False})
    # Use the same image preprocessing as the training run.
    for key in ('min_pixels', 'max_pixels'):
        cfg['model'][key] = saved['model'][key]
    return cfg
