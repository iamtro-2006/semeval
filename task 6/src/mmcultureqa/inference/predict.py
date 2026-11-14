import hashlib
import json

from ..common import config, image, messages, path, provenance, read_jsonl, rows_for


def main():
    import torch
    from transformers import set_seed
    from ..models.loading import load
    cfg = config()
    set_seed(cfg['training']['seed'])
    ic = cfg['inference']
    rows = rows_for(cfg, ic['split'])
    output = path(ic['output'])
    output.parent.mkdir(parents=True, exist_ok=True)
    # Fingerprint BEFORE loading/mutating runtime config; prevents mixed model/prompt outputs.
    signature = hashlib.sha256(json.dumps({'model': cfg['model'], 'prompt': path(cfg['prompt_file']).read_text(encoding='utf-8'),
        'system': cfg.get('system_prompt'), 'generation': {k: v for k, v in ic.items() if k not in ('output', 'resume')},
        'rows': rows}, sort_keys=True).encode()).hexdigest()
    stamp = output.with_suffix('.signature')
    done = set()
    if output.exists():
        if not ic.get('resume'):
            raise FileExistsError(f'{output}: choose a new output filename')
        if not stamp.exists() or stamp.read_text() != signature:
            raise ValueError('Resume model/prompt/data mismatch; choose a new output filename')
        previous = read_jsonl(output)
        done = {r['id'] for r in previous}
        if len(done) != len(previous):
            raise ValueError('Duplicate prediction IDs')
    pending = [r for r in rows if r['id'] not in done]
    if not pending:
        print('All predictions already complete')
        return
    stamp.write_text(signature)
    backend = ic.get('backend', 'transformers')
    if backend == 'vllm':
        from ..models.vllm_loading import load_vllm
        generate = load_vllm(cfg)
    elif backend == 'transformers':
        model, processor = load(cfg)
        model.eval()

        def generate(batch):
            texts = [processor.apply_chat_template(messages(cfg, r['question']), tokenize=False,
                     add_generation_prompt=True) for r in batch]
            inputs = processor(text=texts, images=[image(r) for r in batch], padding=True,
                               truncation=False, return_tensors='pt').to(model.device)
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=ic['max_new_tokens'],
                                           do_sample=ic['do_sample'], use_cache=True)
            return processor.batch_decode(generated[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    else:
        raise ValueError('inference.backend must be vllm or transformers')
    provenance(cfg, output.parent / output.stem)
    with output.open('a', encoding='utf-8') as handle:
        for offset in range(0, len(pending), ic['batch_size']):
            batch = pending[offset:offset + ic['batch_size']]
            answers = generate(batch)
            if len(answers) != len(batch):
                raise ValueError('Inference backend returned an incorrect number of answers')
            for row, answer in zip(batch, answers):
                handle.write(json.dumps({'id': row['id'], 'answer': answer}, ensure_ascii=False) + '\n')
            handle.flush()
            print(f'{len(done) + min(offset + len(batch), len(pending))}/{len(rows)}', flush=True)


if __name__ == '__main__':
    main()
