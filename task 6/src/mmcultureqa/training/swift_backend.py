"""Configuration adapter for the native ms-swift SFT and GRPO pipelines."""
import json
import os
from pathlib import Path

from ..common import messages, path, provenance, read_jsonl, rows_for, write_jsonl


def export_dataset(cfg, rows, destination, stage):
    records = []
    for row in rows:
        if not row.get('answer'):
            raise ValueError('Training/validation requires labels')
        msg = messages(cfg, row['question'])
        msg[-1]['content'] = '<image>\n' + msg[-1]['content'][-1]['text']
        if stage == 'sft':
            msg.append({'role': 'assistant', 'content': row['answer']})
        record = {'messages': msg, 'images': [str(path(row['image']))]}
        if stage == 'grpo':
            record.update(solution=row['answer'], id=row['id'])
        records.append(record)
    write_jsonl(destination, records)


def build_arguments(cfg, stage):
    tc, mc = cfg['training'], cfg['model']
    out = path(tc['output_dir'])
    data = out / 'swift_data'
    train = rows_for(cfg, cfg['data']['train_split'])
    validation = read_jsonl(path(cfg['data']['manifest_dir']) / f'{cfg["data"]["eval_split"]}.jsonl')
    if {r['id'] for r in train} & {r['id'] for r in validation}:
        raise ValueError('Train/validation IDs overlap')
    if {r['image'] for r in train} & {r['image'] for r in validation}:
        raise ValueError('Train/validation images overlap')
    if cfg['monitor'].get('limit') is not None and cfg['monitor']['limit'] < 1:
        raise ValueError('monitor.limit must be null or positive')
    if cfg['monitor'].get('limit'):
        validation = validation[:cfg['monitor']['limit']]
    if not train or not validation:
        raise ValueError('Train and validation must be non-empty')
    export_dataset(cfg, train, data / 'train.jsonl', stage)
    export_dataset(cfg, validation, data / 'validation.jsonl', stage)
    allowed = ('num_train_epochs', 'learning_rate', 'per_device_train_batch_size', 'per_device_eval_batch_size',
               'gradient_accumulation_steps', 'gradient_checkpointing', 'gradient_checkpointing_kwargs',
               'optim', 'lr_scheduler_type', 'warmup_ratio', 'weight_decay', 'max_grad_norm', 'logging_steps',
               'eval_strategy', 'eval_steps', 'save_steps', 'save_total_limit', 'load_best_model_at_end',
               'metric_for_best_model', 'greater_is_better', 'dataloader_num_workers', 'dataloader_pin_memory',
               'report_to', 'seed', 'max_steps', 'acc_strategy')
    args = {k: tc[k] for k in allowed if k in tc}
    generate = bool(tc.get('predict_with_generate', False))
    args.update(model=mc['name'], model_revision=mc.get('revision'), model_type='qwen3_vl', use_hf=True,
                train_type='lora', tuner_backend='peft', output_dir=str(out), add_version=False,
                create_checkpoint_symlink=False, dataset=[str(data / 'train.jsonl')],
                val_dataset=[str(data / 'validation.jsonl')], split_dataset_ratio=0., dataset_shuffle=False,
                val_dataset_shuffle=False, remove_unused_columns=False, strict=True, lazy_tokenize=True,
                max_length=None, truncation_strategy='delete', packing=False, padding_free=False,
                loss_scale='default', attn_impl=mc.get('attn_implementation', 'sdpa'),
                torch_dtype=mc.get('dtype', 'auto'), target_regex=cfg['peft']['target_modules'],
                lora_rank=cfg['peft']['r'], lora_alpha=cfg['peft']['lora_alpha'],
                lora_dropout=cfg['peft']['lora_dropout'], lora_bias=cfg['peft']['bias'],
                freeze_vit=True, freeze_aligner=True)
    if mc.get('load_in_4bit'):
        args.update(quant_method='bnb', quant_bits=4, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True)
    if mc.get('adapter'):
        adapter = path(mc['adapter'])
        if not adapter.is_dir():
            from huggingface_hub import snapshot_download
            adapter = Path(snapshot_download(mc['adapter'], revision=mc.get('adapter_revision')))
        args['adapters'] = [str(adapter)]
        adapter_config = json.loads((adapter / 'adapter_config.json').read_text())
        args['lora_rank'] = max(cfg['peft']['r'], adapter_config['r'])
    if cfg.get('resume_from_checkpoint'):
        args['resume_from_checkpoint'] = str(path(cfg['resume_from_checkpoint']))
    if stage == 'sft':
        args.update(predict_with_generate=generate, metric='mmcqa_bertscore' if generate else None,
                    max_new_tokens=cfg['monitor']['max_new_tokens'], temperature=0.,
                    metric_for_best_model='bertscore_f1' if generate else 'loss', greater_is_better=generate)
    else:
        if not mc.get('adapter'):
            raise ValueError('GRPO continuation requires model.adapter')
        if cfg['reward']['type'] not in {'mmbert', 'exact'}:
            raise ValueError('reward.type must be mmbert or exact')
        for key in ('num_generations', 'max_completion_length', 'temperature', 'top_p', 'beta', 'loss_type', 'use_vllm'):
            args[key] = tc[key]
        for key in ('vllm_mode', 'vllm_enable_lora', 'vllm_gpu_memory_utilization', 'vllm_max_model_len',
                    'vllm_max_num_seqs', 'vllm_enable_prefix_caching', 'vllm_limit_mm_per_prompt',
                    'sleep_level', 'offload_model', 'offload_optimizer'):
            if key in tc:
                args[key] = tc[key]
        args.update(rlhf_type='grpo', reward_funcs=['mmcqa_bertscore' if cfg['reward']['type'] == 'mmbert' else 'mmcqa_exact'],
                    overlong_filter=tc.get('mask_truncated_completions', True),
                    load_best_model_at_end=False, metric_for_best_model='reward', greater_is_better=True)
        args['eval_strategy'] = (tc['eval_strategy'] if tc['eval_strategy'] != 'no' else 'steps') if generate else 'no'
        if args.get('beta', 0) > 0:
            args['ref_adapters'] = args['adapters']
    args.update(cfg.get('swift', {}))
    return args


def canonical_adapter(folder, model_name):
    """Replace machine-specific base snapshot path with the known model identifier."""
    file = Path(folder) / 'adapter_config.json'
    if file.exists():
        obj = json.loads(file.read_text())
        obj['base_model_name_or_path'] = model_name
        file.write_text(json.dumps(obj, indent=2), encoding='utf-8')


def train(cfg, stage):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Use a CUDA GPU / Colab for ms-swift training')
    dtype = cfg['model'].get('dtype', 'auto')
    if dtype == 'auto':
        dtype = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'
    if dtype == 'bfloat16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 is required by this configuration; select a BF16-capable GPU such as L4')
    cfg['_dtype'] = cfg['model']['dtype'] = dtype
    cfg['_framework'] = 'ms-swift==3.11.3'
    os.environ.update(MAX_PIXELS=str(cfg['model']['max_pixels']), MIN_PIXELS=str(cfg['model']['min_pixels']))
    from swift.llm import TrainArguments, RLHFArguments
    from swift.llm.train.sft import SwiftSft
    from swift.llm.train.rlhf import SwiftRLHF
    from transformers import TrainerCallback
    from .swift_metrics import register_plugins
    from ..storage import checkpoint_callbacks
    register_plugins(cfg)
    # Reserve scorer memory before vLLM profiles the GPU. Reuse this cached encoder
    # for every reward/metric call using the configured scoring precision.
    from .swift_metrics import preload_scorer
    if stage == 'grpo' and cfg['reward']['type'] == 'mmbert':
        preload_scorer(cfg['reward'])
    elif stage == 'sft' and cfg['training'].get('predict_with_generate', False):
        preload_scorer(cfg['monitor'])
    native = build_arguments(cfg, stage)
    if cfg['model'].get('load_in_4bit'):
        native['bnb_4bit_compute_dtype'] = dtype
    cls = SwiftSft if stage == 'sft' else SwiftRLHF
    args_cls = TrainArguments if stage == 'sft' else RLHFArguments

    class AdapterMetadata(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            if state.is_world_process_zero:
                canonical_adapter(Path(args.output_dir) / f'checkpoint-{state.global_step}', cfg['model']['name'])

    class Pipeline(cls):
        def train(self, trainer):
            trainer.add_callback(AdapterMetadata())
            for callback in checkpoint_callbacks():
                trainer.add_callback(callback)
            cfg['_trainable_parameters'] = [n for n, p in trainer.model.named_parameters() if p.requires_grad]
            if not cfg['_trainable_parameters']:
                raise RuntimeError('No trainable LoRA parameters')
            provenance(cfg, native['output_dir'])
            result = super().train(trainer)
            final = Path(native['output_dir']) / 'final'
            trainer.save_model(str(final))
            if trainer.is_world_process_zero():
                self.processor.save_pretrained(final)
                canonical_adapter(final, cfg['model']['name'])
            return result

    out = path(cfg['training']['output_dir'])
    out.mkdir(parents=True, exist_ok=True)
    (out / 'swift_arguments.json').write_text(json.dumps(native, indent=2), encoding='utf-8')
    Pipeline(args_cls(**native)).main()
