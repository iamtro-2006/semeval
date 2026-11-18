"""Sequential GPU ownership for single-GPU BF16 LoRA validation."""
import copy
import gc
import json
import os
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from types import MethodType

from ..common import ROOT, path, read_jsonl, write_jsonl
from .vllm_worker import PersistentValidationWorker


@contextmanager
def training_on_cpu(trainer):
    """Restore each parameter, buffer, gradient and optimizer tensor to its device."""
    import torch
    entries = []
    seen = set()

    def move(tensor):
        if id(tensor) in seen or tensor.device.type != 'cuda':
            return
        seen.add(id(tensor))
        entries.append((tensor, tensor.device))
        tensor.data = tensor.data.to('cpu')

    def nested(value):
        if torch.is_tensor(value):
            move(value)
        elif isinstance(value, dict):
            for item in value.values():
                nested(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                nested(item)

    try:
        torch.cuda.synchronize()
        for tensor in trainer.model.parameters():
            if tensor.grad is not None:
                move(tensor.grad)
            move(tensor)
        for tensor in trainer.model.buffers():
            move(tensor)
        if trainer.optimizer is not None:
            nested(trainer.optimizer.state)
        gc.collect()
        torch.cuda.empty_cache()
        yield
    finally:
        for tensor, device in entries:
            tensor.data = tensor.data.to(device)
        gc.collect()
        torch.cuda.empty_cache()


def install_validation(trainer, cfg, processor):
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('Sequential vLLM SFT validation supports one GPU/process only')
    if cfg['model'].get('load_in_4bit'):
        raise ValueError('CPU-swapped SFT validation requires BF16 LoRA, not a quantized training base')
    if cfg['peft'].get('train_projector'):
        raise ValueError('vLLM adapter validation requires language-only LoRA; disable train_projector')
    original = trainer.evaluate
    worker = PersistentValidationWorker()

    def snapshot(self, work, sample_limit=None):
        from .swift_backend import canonical_adapter
        adapter = work / 'adapter'
        self.save_model(str(adapter))
        processor.save_pretrained(adapter)
        canonical_adapter(adapter, cfg['model']['name'])
        validation = read_jsonl(path(cfg['data']['manifest_dir']) / f'{cfg["data"]["eval_split"]}.jsonl')
        limit = cfg['monitor'].get('limit')
        if limit is not None and limit < len(validation):
            import random
            validation = random.Random(42).sample(validation, limit)
        if sample_limit is not None:
            validation = validation[:sample_limit]
        write_jsonl(work / 'validation.jsonl', validation)
        child = copy.deepcopy(cfg)
        child.pop('extends', None)
        child['data'].update(manifest_dir=str(work), limit=None)
        child['model'].update(adapter=str(adapter), adapter_revision=None, merged_model=False,
                              load_in_4bit=False)
        (work / 'prompt.txt').write_text(path(cfg['prompt_file']).read_text(encoding='utf-8'), encoding='utf-8')
        child['prompt_file'] = str(work / 'prompt.txt')
        child['inference'] = dict(backend='vllm', split='validation', batch_size=8,
            max_new_tokens=cfg['monitor']['max_new_tokens'], do_sample=False, resume=False,
            output=str(work / 'predictions.jsonl'), **cfg['monitor'].get('vllm', {}))
        config = work / 'config.json'
        config.write_text(json.dumps(child, ensure_ascii=False), encoding='utf-8')
        return config, validation

    def preflight():
        # Exercise the exact base + LoRA image path twice BEFORE any full validation pass.
        # The engine survives this test and is reused for the full training run.
        print('[validation] Preflight: 2 images x 2 snapshots, including sleep/wake', flush=True)
        with tempfile.TemporaryDirectory(prefix='mmcqa-preflight-', dir=path(cfg['training']['output_dir'])) as root:
            with training_on_cpu(trainer):
                try:
                    for index in range(2):
                        work = Path(root) / str(index)
                        work.mkdir()
                        config, rows = snapshot(trainer, work, sample_limit=2)
                        worker.predict(config)
                        predictions = read_jsonl(work / 'predictions.jsonl')
                        if len(predictions) != len(rows) or {r['id'] for r in predictions} != {r['id'] for r in rows}:
                            raise ValueError('vLLM preflight coverage mismatch')
                except BaseException:
                    worker.close()
                    raise
        print('[validation] Preflight passed; proceeding to full validation/training', flush=True)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix='eval', **kwargs):
        # Native teacher-forced validation retains loss/token accuracy and their progress bar.
        metrics = original(eval_dataset=eval_dataset, ignore_keys=ignore_keys,
                           metric_key_prefix=metric_key_prefix, **kwargs)
        with tempfile.TemporaryDirectory(prefix='mmcqa-val-', dir=path(cfg['training']['output_dir'])) as work:
            work = Path(work)
            config, validation = snapshot(self, work)
            # Worker acknowledges only once generation finishes and vLLM is sleeping in RAM.
            with training_on_cpu(self):
                try:
                    worker.predict(config)
                    predicted = read_jsonl(work / 'predictions.jsonl')
                    if len(predicted) != len(validation) or {r['id'] for r in predicted} != {r['id'] for r in validation}:
                        raise ValueError('Periodic vLLM predictions do not cover validation exactly')
                    subprocess.run([sys.executable, '-u', '-m', 'mmcultureqa.score',
                        '--dtype', cfg['monitor']['dtype'], '--gold', str(work / 'validation.jsonl'),
                        '--pred', str(work / 'predictions.jsonl'), '--lang', 'en',
                        '--out', str(work / 'scores.json')], cwd=ROOT, check=True)
                except BaseException:
                    worker.close()
                    raise
            report = json.loads((work / 'scores.json').read_text(encoding='utf-8'))
            blocks = [scores for block in report['result'] for scores in block.values()
                      if isinstance(scores, dict) and 'BERTScore-F1' in scores]
            if len(blocks) != 1:
                raise ValueError('Expected exactly one validation F1 result')
            key = metric_key_prefix + '_bertscore_f1'
            metrics[key] = float(blocks[0]['BERTScore-F1'])
            self.log({key: metrics[key]})
        return metrics

    trainer.evaluate = MethodType(evaluate, trainer)
    worker.preflight = preflight
    return worker
