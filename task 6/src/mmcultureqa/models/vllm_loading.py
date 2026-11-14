"""Native ms-swift vLLM engine for batched, adapter-aware QA inference."""
import json
import os

from ..common import image, messages, path


def load_vllm(cfg):
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('The default BF16 vLLM inference configuration requires a BF16-capable CUDA GPU')
    if cfg['model'].get('load_in_4bit'):
        raise ValueError('This vLLM recipe uses BF16 weights. For 4-bit select inference.backend=transformers')
    if cfg['model'].get('dtype') not in {'bfloat16', 'auto'}:
        raise ValueError('This vLLM recipe requires model.dtype=bfloat16; use transformers for a different dtype')
    os.environ.update(MAX_PIXELS=str(cfg['model']['max_pixels']), MIN_PIXELS=str(cfg['model']['min_pixels']))
    from swift.llm import VllmEngine, InferRequest, RequestConfig
    mc, ic = cfg['model'], cfg['inference']
    adapters, rank = [], cfg['peft']['r']
    if mc.get('adapter'):
        adapter = path(mc['adapter'])
        if not adapter.is_dir():
            from huggingface_hub import snapshot_download
            adapter = path(snapshot_download(mc['adapter'], revision=mc.get('adapter_revision')))
        adapter_config = json.loads((adapter / 'adapter_config.json').read_text())
        if adapter_config['base_model_name_or_path'].rstrip('/') != mc['name'].rstrip('/'):
            raise ValueError('Adapter base model does not match the configured base model')
        rank = adapter_config['r']
        adapters = [str(adapter)]
    engine = VllmEngine(mc['name'], torch.bfloat16, model_type='qwen3_vl', use_hf=True,
                        revision=mc.get('revision'), adapters=adapters, max_lora_rank=rank,
                        max_model_len=ic['vllm_max_model_len'], max_num_seqs=ic['vllm_max_num_seqs'],
                        gpu_memory_utilization=ic['vllm_gpu_memory_utilization'],
                        enable_prefix_caching=True, limit_mm_per_prompt={'image': 1, 'video': 0},
                        seed=cfg['training']['seed'])
    cfg['_dtype'] = 'bfloat16'

    def generate(rows):
        requests = []
        for row in rows:
            msg = messages(cfg, row['question'])
            msg[-1]['content'] = '<image>\n' + msg[-1]['content'][-1]['text']
            requests.append(InferRequest(messages=msg, images=[image(row)]))
        responses = engine.infer(requests, RequestConfig(max_tokens=ic['max_new_tokens'],
            temperature=ic.get('temperature', 0.8) if ic['do_sample'] else 0.0,
            seed=cfg['training']['seed']), use_tqdm=False)
        answers = []
        for response in responses:
            if isinstance(response, Exception):
                raise response
            answers.append(response.choices[0].message.content)
        return answers

    return generate
