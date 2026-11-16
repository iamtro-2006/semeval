"""Merge a selected local adapter into a fresh BF16 base after training exits."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile

from .common import path
from .models.artifacts import require_merged_model


def export_merged(run_dir, device='cuda:0'):
    run = path(run_dir).resolve()
    cfg = json.loads((run / 'run.json').read_text(encoding='utf-8'))
    adapter = run / 'final'
    weights = [adapter / name for name in ('adapter_model.safetensors', 'adapter_model.bin')
               if (adapter / name).is_file()]
    if not weights or not (adapter / 'adapter_config.json').is_file():
        raise FileNotFoundError(f'Selected adapter is incomplete: {adapter}')
    adapter_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in [adapter / 'adapter_config.json', *weights]}
    mc = cfg['model']
    identity = {'base': mc['name'], 'revision': mc.get('revision'),
                'adapter_sha256': adapter_hashes, 'dtype': 'bfloat16',
                'prompt_sha256': hashlib.sha256(cfg['prompt_text'].encode()).hexdigest(),
                'system_prompt': cfg.get('system_prompt')}
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    destination = run / 'best_model'
    if destination.exists():
        require_merged_model(destination)
        previous = json.loads((destination / 'merge.json').read_text(encoding='utf-8'))
        if previous.get('signature') != signature:
            raise FileExistsError('best_model belongs to a different adapter/base; choose a new run directory')
        return destination

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from peft import PeftModel
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('BF16 merge requires a CUDA device, or explicitly select --device cpu')
    base = path(mc['name'])
    local = base.is_dir()
    if mc.get('local_files_only') and not local:
        raise FileNotFoundError(f'Local base model missing: {base}')
    # Never merge into the quantized training instance. Reload the original BF16 base.
    model = AutoModelForImageTextToText.from_pretrained(
        str(base) if local else mc['name'], revision=None if local else mc.get('revision'),
        dtype=torch.bfloat16, device_map={'': device}, local_files_only=local,
        attn_implementation=mc.get('attn_implementation', 'sdpa'))
    if getattr(model, 'is_loaded_in_4bit', False) or getattr(model, 'is_loaded_in_8bit', False):
        raise ValueError('Refusing to export from a quantized base')
    processor = AutoProcessor.from_pretrained(str(adapter), local_files_only=True)
    peft_model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    merged = peft_model.merge_and_unload(safe_merge=True)
    merged.to(dtype=torch.bfloat16)
    merged.config.use_cache = True
    selection_file = run / 'selection.json'
    selection = json.loads(selection_file.read_text(encoding='utf-8')) if selection_file.is_file() else {}
    report = {**identity, 'signature': signature, 'selection': selection,
              'note': 'Selected during NF4 training; merged into the original BF16 base; score this export separately.'}
    # Publish only a complete model; interrupted exports never masquerade as best_model.
    with tempfile.TemporaryDirectory(prefix='.merge-', dir=run) as temporary:
        staged = Path(temporary) / 'best_model'
        merged.save_pretrained(str(staged), safe_serialization=True, max_shard_size='2GB')
        processor.save_pretrained(str(staged))
        (staged / 'run_config.json').write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding='utf-8')
        (staged / 'prompt.txt').write_text(cfg['prompt_text'], encoding='utf-8')
        (staged / 'merge.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        require_merged_model(staged)
        staged.rename(destination)
    return destination


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    print(f'Merged BF16 model: {export_merged(args.run_dir, args.device)}', flush=True)


if __name__ == '__main__':
    main()
