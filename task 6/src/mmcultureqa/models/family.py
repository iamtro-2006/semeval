"""Model-family metadata shared by training, export and inference."""
import json

from ..common import path


def model_type(cfg):
    mc = cfg['model']
    if mc.get('model_type'):
        result = mc['model_type']
    elif (path(mc['name']) / 'config.json').is_file():
        result = json.loads((path(mc['name']) / 'config.json').read_text(encoding='utf-8'))['model_type']
    else:
        result = 'qwen3_5' if 'qwen3.5' in mc['name'].lower() else 'qwen3_vl'
    if result not in {'qwen3_vl', 'qwen3_5'}:
        raise ValueError(f'Unsupported model family: {result}')
    return result
