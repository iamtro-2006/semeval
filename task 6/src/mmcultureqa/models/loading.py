from ..common import path


def source(value):
    local = path(value)
    return str(local) if local.exists() else value


def load(cfg, train=False):
    import os
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

    if not torch.cuda.is_available():
        raise RuntimeError('This entry point requires a CUDA GPU. Use the Colab notebook; data preparation runs on CPU.')
    mc = cfg['model']
    dtype = mc.get('dtype', 'auto')
    if dtype == 'auto':
        dtype = 'bfloat16' if torch.cuda.is_bf16_supported() else 'float16'
    if dtype == 'bfloat16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 not supported on this GPU; set model.dtype=float16')
    cfg['_dtype'] = dtype
    base = source(mc['name'])
    proc = mc.get('processor') or base
    processor = AutoProcessor.from_pretrained(source(proc), revision=mc.get('processor_revision', mc.get('revision')),
                                            min_pixels=mc['min_pixels'], max_pixels=mc['max_pixels'])
    processor.tokenizer.padding_side = 'right' if train else 'left'
    kwargs = {'revision': mc.get('revision'), 'dtype': getattr(torch, dtype),
              'attn_implementation': mc.get('attn_implementation', 'sdpa')}
    if train:
        # Never shard a training model with device_map='auto'. One process owns one GPU.
        if mc.get('load_in_4bit'):
            kwargs['device_map'] = {'': int(os.environ.get('LOCAL_RANK', 0))}
    else:
        kwargs['device_map'] = 'auto'
    if mc.get('load_in_4bit'):
        kwargs['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                                          bnb_4bit_use_double_quant=True,
                                                          bnb_4bit_compute_dtype=getattr(torch, dtype))
    model = AutoModelForImageTextToText.from_pretrained(base, **kwargs)
    if train and mc.get('load_in_4bit'):
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=cfg['training']['gradient_checkpointing'])
    if mc.get('adapter'):
        from peft import PeftConfig
        adapter_source = source(mc['adapter'])
        ac = PeftConfig.from_pretrained(adapter_source, revision=mc.get('adapter_revision'))
        if ac.base_model_name_or_path.rstrip('/') != mc['name'].rstrip('/'):
            raise ValueError(f'Adapter base {ac.base_model_name_or_path} does not match {mc["name"]}')
        model = PeftModel.from_pretrained(model, adapter_source, revision=mc.get('adapter_revision'), is_trainable=train)
    elif train:
        model = get_peft_model(model, LoraConfig(**cfg['peft']))
    model.config.use_cache = not train
    if train:
        model.print_trainable_parameters()
        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError('No trainable parameters')
        cfg['_trainable_parameters'] = trainable
    return model, processor
