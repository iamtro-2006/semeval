import hashlib
import json
import sys

from ..common import config, image, messages, path, provenance, read_jsonl, rows_for


def main():
    import torch
    from transformers import set_seed
    from ..models.loading import load
    cfg = config()
    if cfg['model'].get('merged_model'):
        from ..models.artifacts import configure_merged_model
        cfg = configure_merged_model(cfg, cfg['model']['name'], cfg['model'].get('load_in_4bit', False))
    set_seed(cfg['training']['seed'])
    ic = cfg['inference']
    if ic.get('prompt_mode', 'saved') == 'cot':
        # Apply after merged-model restoration; this changes inference only, not training artifacts.
        cfg['prompt_file'] = ic['prompt_file']
        cfg['reasoning'] = {'enabled': True, 'max_new_tokens': ic['max_new_tokens'],
                            'vllm_max_model_len': ic['vllm_max_model_len']}
    reasoning = cfg.get('reasoning', {})
    if reasoning.get('enabled'):
        ic['max_new_tokens'] = max(ic['max_new_tokens'], reasoning.get('max_new_tokens', 1024))
        ic['vllm_max_model_len'] = max(ic['vllm_max_model_len'], reasoning.get('vllm_max_model_len', 4096))
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
    from tqdm import tqdm
    with output.open('a', encoding='utf-8') as handle, tqdm(
            total=len(rows), initial=len(done), desc='Validation' if ic['split'] == 'validation' else 'Inference',
            unit='question', dynamic_ncols=True, mininterval=1., file=sys.stdout) as progress:
        for offset in range(0, len(pending), ic['batch_size']):
            batch = pending[offset:offset + ic['batch_size']]
            answers = generate(batch)
            if len(answers) != len(batch):
                raise ValueError('Inference backend returned an incorrect number of answers')
            for row, answer in zip(batch, answers):
                if reasoning.get('enabled'):
                    # Keep raw generations for diagnostics, but score/submit the final answer only.
                    trace = output.with_suffix('.traces.jsonl')
                    record = {'id': row['id'], 'generation': answer}
                    from ..data.distillation import final_answer
                    retries = ic.get('format_retries', 0)
                    if retries not in (0, 1):
                        raise ValueError('inference.format_retries must be 0 or 1')
                    for attempt in range(retries + 1):
                        record = {'id': row['id'], 'generation': answer, 'attempt': attempt + 1}
                        error = None
                        try:
                            answer = final_answer(answer)
                        except ValueError as exc:
                            error = exc
                            record['format_error'] = str(exc)
                        with trace.open('a', encoding='utf-8') as traces:
                            traces.write(json.dumps(record, ensure_ascii=False) + '\n')
                        if error is None:
                            break
                        if attempt < retries:
                            # Retry only this row; use sampling to avoid replaying a greedy loop.
                            original = dict(ic)
                            seed = cfg['training']['seed']
                            try:
                                ic.update(do_sample=True, temperature=0.7)
                                cfg['training']['seed'] = seed + 1
                                retry = generate([row])
                                if len(retry) != 1:
                                    raise ValueError('Retry returned an incorrect number of answers')
                                answer = retry[0]
                            finally:
                                ic.clear()
                                ic.update(original)
                                cfg['training']['seed'] = seed
                    if error is not None:
                        if retries:
                            raise ValueError(f'Invalid final answer for {row["id"]} after 2 attempts; see {trace}') from error
                        answer = ''
                    if error is not None and ic.get('on_format_error', 'raise') != 'empty':
                        raise error
                handle.write(json.dumps({'id': row['id'], 'answer': answer}, ensure_ascii=False) + '\n')
            handle.flush()
            progress.update(len(batch))


if __name__ == '__main__':
    main()
