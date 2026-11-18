"""Offline, resumable vLLM synthesis on TRAIN only; no validation answers enter teacher requests."""
import json
import os
import random

from ..common import config, image, path, read_jsonl, write_jsonl
from .distillation import digest, parse_teacher, sample_rows


def atomic_json(file, value):
    temporary = file.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(file)


def read_journal(file):
    """Recover only an interrupted final append; never silently discard earlier corruption."""
    if not file.exists():
        return []
    data = file.read_bytes()
    end = data.rfind(b'\n') + 1
    records = [json.loads(line) for line in data[:end].splitlines() if line.strip()]
    if end != len(data):
        with file.open('r+b') as handle:
            handle.truncate(end)
    return records


def generate_dataset(cfg):
    tc = cfg['teacher']
    if cfg['data']['train_split'] != 'train' or cfg['data']['eval_split'] not in {'dev', 'validation'}:
        raise ValueError('Teacher synthesis is restricted to the prepared train split')
    train = read_jsonl(path(cfg['data']['manifest_dir']) / 'train.jsonl')
    validation = read_jsonl(path(cfg['data']['manifest_dir']) / f'{cfg["data"]["eval_split"]}.jsonl')
    if not train or len({r['id'] for r in train}) != len(train) or any(not r.get('answer') for r in train):
        raise ValueError('Expected unique, labelled training rows')
    if ({r['id'] for r in train} & {r['id'] for r in validation}
            or {r['image'] for r in train} & {r['image'] for r in validation}):
        raise ValueError('Train/validation overlap')
    for key in ('batch_size', 'max_num_seqs', 'max_new_tokens', 'attempts'):
        if tc[key] < 1:
            raise ValueError(f'teacher.{key} must be positive')
    if tc.get('limit') is not None and tc['limit'] < 1:
        raise ValueError('teacher.limit must be null or positive')
    selected = sample_rows(train, tc.get('limit'), tc['seed'])
    prompt = path(cfg['prompt_file']).read_text(encoding='utf-8')
    spec = {'teacher': tc, 'system_prompt': cfg['system_prompt'], 'prompt': prompt,
            'pixels': {k: cfg['model'][k] for k in ('min_pixels', 'max_pixels')},
            'source_train_hash': digest(train), 'source_validation_hash': digest(validation)}
    if tc.get('limit') is not None:
        spec['sampling'] = 'random_without_replacement'
    fingerprint = digest(spec)
    output = path(tc['output_dir'])
    output.mkdir(parents=True, exist_ok=True)
    metadata_file, journal_file = output / 'generation.json', output / 'attempts.jsonl'
    if metadata_file.exists():
        metadata = json.loads(metadata_file.read_text(encoding='utf-8'))
        if metadata['fingerprint'] != fingerprint:
            raise ValueError('Resume teacher/prompt/data mismatch: choose a new teacher output directory')
    else:
        if any(output.iterdir()):
            raise ValueError('Non-empty teacher directory without generation.json')
        from huggingface_hub import HfApi
        revision = HfApi().model_info(tc['name'], revision=tc['revision']).sha
        metadata = {'schema_version': 1, 'fingerprint': fingerprint, 'resolved_revision': revision, 'spec': spec}
        atomic_json(metadata_file, metadata)
    journal = read_journal(journal_file)
    done = {r['id'] for r in journal}
    if len(done) != len(journal) or not done <= {r['id'] for r in selected}:
        raise ValueError('Duplicate or foreign IDs in teacher journal')
    pending = [r for r in selected if r['id'] not in done]
    if pending:
        import torch
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError('Teacher requires a BF16-capable CUDA GPU')
        if torch.cuda.get_device_capability() < (8, 9):
            raise RuntimeError('This block-FP8 teacher recipe targets L4/Ada, Hopper or newer; select a suitable GPU')
        os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams
        revision = metadata['resolved_revision']
        processor = AutoProcessor.from_pretrained(tc['name'], revision=revision,
            min_pixels=cfg['model']['min_pixels'], max_pixels=cfg['model']['max_pixels'])
        # Prequantized official FP8 checkpoint; no online quantization and no second teacher copy.
        llm = LLM(model=tc['name'], revision=revision, tokenizer_revision=revision, dtype='bfloat16',
            max_model_len=tc['max_model_len'], max_num_seqs=tc['max_num_seqs'],
            gpu_memory_utilization=tc['gpu_memory_utilization'], enable_prefix_caching=True,
            limit_mm_per_prompt={'image': 1, 'video': 0}, seed=tc['seed'],
            mm_processor_kwargs={'min_pixels': cfg['model']['min_pixels'],
                                 'max_pixels': cfg['model']['max_pixels']})
        with journal_file.open('a', encoding='utf-8') as handle:
            for offset in range(0, len(pending), tc['batch_size']):
                batch = pending[offset:offset + tc['batch_size']]
                unresolved = list(batch)
                results = {}
                for attempt in range(tc['attempts']):
                    requests = []
                    for row in unresolved:
                        msg = [{'role': 'system', 'content': cfg['system_prompt']},
                               {'role': 'user', 'content': [{'type': 'image'}, {'type': 'text',
                                'text': prompt.format(question=row['question'], answer=row['answer'])}]}]
                        requests.append({'prompt': processor.apply_chat_template(msg, tokenize=False,
                            add_generation_prompt=True), 'multi_modal_data': {'image': image(row)}})
                    params = SamplingParams(max_tokens=tc['max_new_tokens'], temperature=tc['temperature'],
                        top_p=tc['top_p'], top_k=tc['top_k'], seed=tc['seed'] + offset + attempt)
                    responses = llm.generate(requests, params, use_tqdm=True)
                    if len(responses) != len(unresolved):
                        raise ValueError('Teacher returned an incorrect number of responses')
                    retry = []
                    for row, response in zip(unresolved, responses):
                        completion = response.outputs[0]
                        item = results.setdefault(row['id'], {'id': row['id'], 'attempts': []})
                        trace = {'text': completion.text, 'finish_reason': completion.finish_reason}
                        try:
                            item['reasoning'] = parse_teacher(completion.text, completion.finish_reason)
                        except (ValueError, TypeError, AttributeError) as error:
                            trace['error'] = str(error)
                            retry.append(row)
                        item['attempts'].append(trace)
                    unresolved = retry
                    if not unresolved:
                        break
                for row in batch:
                    item = results[row['id']]
                    handle.write(json.dumps(item, ensure_ascii=False) + '\n')
                    journal.append(item)
                handle.flush()
                os.fsync(handle.fileno())
                print(f'Teacher completed {len(journal)}/{len(selected)}; '
                      f'accepted {sum(bool(r.get("reasoning")) for r in journal)}', flush=True)
    by_id = {r['id']: r for r in journal}
    accepted = [{**r, 'reasoning': by_id[r['id']]['reasoning']} for r in selected
                if by_id.get(r['id'], {}).get('reasoning')]
    rejected = [{**r, 'attempts': by_id[r['id']]['attempts']} for r in selected
                if r['id'] in by_id and not by_id[r['id']].get('reasoning')]
    write_jsonl(output / 'train_reasoning.jsonl', accepted)
    write_jsonl(output / 'rejected.jsonl', rejected)
    write_jsonl(output / 'audit_sample.jsonl', random.Random(42).sample(accepted, min(32, len(accepted))))
    report = {'schema_version': 1, 'complete': len(journal) == len(selected), 'smoke': tc.get('limit') is not None,
              'total_train': len(train), 'attempted': len(journal), 'accepted': len(accepted),
              'rejected': len(rejected), 'accepted_hash': digest(accepted),
              'source_train_hash': digest(train), 'source_validation_hash': digest(validation),
              'teacher': tc['name'], 'teacher_revision': metadata['resolved_revision'],
              'sample_size': tc.get('limit'), 'sample_seed': tc['seed'],
              'selected_ids': [r['id'] for r in selected],
              'note': 'Answer-conditioned synthetic solutions; structural checks are not factual verification.'}
    atomic_json(output / 'dataset.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    generate_dataset(config())
