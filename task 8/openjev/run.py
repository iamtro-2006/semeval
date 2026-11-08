"""Shared OpenJev workflow for train.ipynb and inference.ipynb on Colab.

Only final reports, an adapter and one resumable checkpoint are written.
Task prompts follow OpenJev's letter-option layout; probabilities here are
restricted A/B softmax scores, not the vendor's calibrated API probabilities.
"""
from array import array
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'src'))
from inference import ID_FIELD, canonical_gold, evaluate, read_tsv
from metrics import IDENTITIES, parse_target

DEFAULTS = dict(model='openjev/OpenJev-Flash-9B', revision='main',
                train='', dev='', test='', output='/content/results',
                samples=True, epochs=2, lr=1e-4, rank=8, max_length=2048,
                accumulation=16, save_steps=100, seed=42)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                         encoding='utf-8')
    temporary.replace(path)


def load_samples(path=HERE / 'samples.json'):
    rows = json.loads(Path(path).read_text(encoding='utf-8'))['examples']
    seen = set()
    for row in rows:
        if row['origin'] != 'synthetic' or row['split'] not in ('train', 'dev'):
            raise ValueError('Synthetic examples must declare train/dev provenance.')
        if row[ID_FIELD] in seen or not row[ID_FIELD].startswith('synthetic_a_'):
            raise ValueError('Synthetic IDs must be unique and explicitly marked.')
        seen.add(row[ID_FIELD])
        if row['target'] not in ('none', 'group_a', 'individual_a'):
            raise ValueError('Unexpected synthetic target.')
        if (row['target'] == 'none') != (row['hate_speech'] == 'no'):
            raise ValueError('Synthetic target and hate_speech disagree.')
        if any(not row.get(k, '').strip() for k in ('yt_title', 'yt_description', 'yt_comment')):
            raise ValueError('Empty synthetic text field.')
    for split in ('train', 'dev'):
        if sum(r['split'] == split and r['target'] != 'none' for r in rows) < 4:
            raise ValueError('At least four a-positive samples are required per split.')
    return rows


def prepare(config):
    """Read caller-supplied TSV paths; augment only in memory, without resplitting."""
    config = {**DEFAULTS, **config}
    unknown = set(config) - set(DEFAULTS)
    if unknown:
        raise ValueError(f'Unknown settings: {sorted(unknown)}')
    for key in ('epochs', 'rank', 'max_length', 'accumulation', 'save_steps'):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f'{key} must be a positive integer.')
    if type(config['seed']) is not int or not math.isfinite(config['lr']) or config['lr'] <= 0:
        raise ValueError('Invalid seed or learning rate.')
    groups, paths = {}, {}
    for split in ('train', 'dev', 'test'):
        if split == 'test' and not config[split]:
            continue
        path = Path(config[split]).expanduser().resolve()
        if not config[split] or not path.is_file():
            raise FileNotFoundError(f'Set {split.upper()} to an existing TSV file: {path}')
        paths[split] = path
        groups[split] = []
        for row in read_tsv(path):
            if not row.get('target'):
                raise ValueError(f'{split} must contain target labels: {row[ID_FIELD]}')
            canonical_gold(row['target'])
            groups[split].append({**row, 'split': split, 'origin': 'original'})
    # Original split boundaries are checked before synthetic augmentation.
    seen_ids, seen_comments = {}, {}
    videos = {}
    for split, rows in groups.items():
        videos[split] = {(r['yt_title'].casefold().strip(), r['yt_description'].casefold().strip())
                         for r in rows}
        for row in rows:
            normalized = ' '.join(row['yt_comment'].casefold().split())
            for key, seen, label in ((row[ID_FIELD], seen_ids, 'ID'),
                                     (normalized, seen_comments, 'comment')):
                if key in seen and seen[key] != split:
                    raise ValueError(f'Cross-split {label} overlap: {row[ID_FIELD]}')
                seen[key] = split
    groups['dev_synthetic'] = []
    if config['samples']:
        for row in load_samples():
            normalized = ' '.join(row['yt_comment'].casefold().split())
            if row[ID_FIELD] in seen_ids or normalized in seen_comments:
                raise ValueError(f'Synthetic example overlaps another row: {row[ID_FIELD]}')
            seen_ids[row[ID_FIELD]] = row['split']
            seen_comments[normalized] = row['split']
            dest = 'train' if row['split'] == 'train' else 'dev_synthetic'
            groups[dest].append(row)
    questions = json.loads((HERE / 'questions.json').read_text(encoding='utf-8'))
    if tuple(questions['identities']) != IDENTITIES:
        raise ValueError('Questions must cover all nine identities in canonical order.')
    summary = {}
    for split, rows in groups.items():
        support = {code: sum(r['target'] != 'none' and code in parse_target(r['target'])[1]
                             for r in rows) for code in IDENTITIES}
        summary[split] = dict(rows=len(rows), synthetic=sum(r['origin'] == 'synthetic' for r in rows),
                              none=sum(r['target'] == 'none' for r in rows), support=support)
    overlap = {f'{a}/{b}': len(videos[a] & videos[b])
               for a in videos for b in videos if a < b}
    return config, groups, questions, paths, dict(splits=summary, shared_videos=overlap)


def decisions(rows, questions, inference=False):
    """One source comment yields nine binary tasks and one conditional scope task."""
    for row in rows:
        label = row.get('target', '')
        scope, codes = (None, set()) if label in ('', 'none') else parse_target(label)
        state = {k: row[k] for k in questions['state_fields']}
        common = dict(source_id=row[ID_FIELD], state=state, origin=row['origin'], split=row['split'])
        for code, question in questions['identities'].items():
            yield {**common, 'key': code, 'question': question['instructions'],
                   'options': [('yes', 'The statement is true.'), ('no', 'The statement is false.')],
                   'answer': 0 if code in codes else 1, 'supervised': bool(label)}
        if scope is not None or inference:
            yield {**common, 'key': 'scope', 'question': questions['scope']['instructions'],
                   'options': list(questions['scope']['criteria'].items()),
                   'answer': 0 if scope == 'individual' else 1, 'supervised': scope is not None}


def prompt(task, questions, reverse=False):
    """Keep gold/provenance out of model inputs; letters refer to option positions."""
    options = task['options'][::-1] if reverse else task['options']
    lines = '\n'.join(f'[{letter}] {key}: {description}'
                      for letter, (key, description) in zip('AB', options))
    instructions = questions['instructions'] + ' ' + questions['identity_rules']
    return (f'State:\n{json.dumps(task["state"], ensure_ascii=False)}\n\n'
            f'Question: {instructions} {task["question"]}\nOptions:\n{lines}\n\n'
            'Answer with the letter of the best option only.')


def order_for(seed, epoch, task):
    identity = f'{seed}:{epoch}:{task["source_id"]}:{task["key"]}'
    return bool(hashlib.sha256(identity.encode()).digest()[0] % 2)


def decode(probabilities, threshold):
    codes = [k for k in IDENTITIES if probabilities[k] >= threshold]
    if not codes:
        return '', 'abstain'  # No forced label; insufficient evidence is not gold none.
    scope = 'individual' if probabilities['scope'] >= 0.5 else 'group'
    return scope + '_' + ','.join(codes), 'predicted'


def report(rows, scores, threshold):
    if not rows:
        return {'n': 0}
    if len(rows) != len(scores):
        raise ValueError('Rows and predictions must align.')
    targets = [decode(p, threshold)[0] for p in scores]
    result = evaluate(rows, targets)
    if result['status'] != 'evaluated':
        result.update(abstain_count=sum(not value for value in targets),
                      coverage=sum(bool(value) for value in targets) / len(rows))
        return result
    tp = fp = fn = 0
    brier = 0.0
    for row, p in zip(rows, scores):
        truth = set() if row['target'] == 'none' else parse_target(row['target'])[1]
        for code in IDENTITIES:
            y, predicted = code in truth, p[code] >= threshold
            tp += y and predicted
            fp += not y and predicted
            fn += y and not predicted
            brier += (p[code] - int(y)) ** 2
    result['binary_identity_micro_f1'] = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    result['binary_identity_brier'] = brier / (len(rows) * len(IDENTITIES))
    result['abstain_count'] = sum(not value for value in targets)
    result['coverage'] = 1 - result['abstain_count'] / len(rows)
    result['note'] = ('Empty identity selections are abstentions and count as invalid targets. '
                      'Binary scores include all rows; Task C metrics use gold-active rows. '
                      'Synthetic none rows are negative controls, not a trained none detector.')
    return result


def choose_threshold(rows, scores):
    """Tune one global threshold on original dev only, never synthetic dev/test."""
    if any(row['origin'] != 'original' or row['split'] != 'dev' for row in rows):
        raise ValueError('Threshold selection requires original dev rows only.')
    candidates = [i / 100 for i in range(10, 91, 5)]
    best = max(candidates, key=lambda t: (report(rows, scores, t)['binary_identity_micro_f1'],
                                          -abs(t - 0.5)))
    return best


class Run:
    def __init__(self, config):
        self.config, self.groups, self.questions, self.paths, self.summary = prepare(config)
        self.output = Path(self.config['output']).expanduser().resolve()
        if any(p == self.output or self.output in p.parents for p in self.paths.values()):
            raise ValueError('Keep OUTPUT separate from dataset files.')
        self.tasks = {k: list(decisions(rows, self.questions, inference=k != 'train'))
                      for k, rows in self.groups.items()}
        self.model = None
        self.tokens = {}
        self.checked = False
        print(json.dumps(self.summary, indent=2))
        print('No dataset copies will be written. Shared videos retain the existing split protocol.')

    def check(self):
        """Check GPU, tokenizer and every prompt length before loading model weights."""
        import torch
        from transformers import AutoConfig, AutoTokenizer
        self.checked = False
        self.tokens.clear()
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError('Select a CUDA GPU supporting BF16 in the Colab runtime (A100 recommended).')
        self.torch = torch
        torch.manual_seed(self.config['seed'])
        torch.cuda.manual_seed_all(self.config['seed'])
        info = torch.cuda.get_device_properties(0)
        print(f'GPU: {info.name}; total VRAM: {info.total_memory / 2**30:.1f} GiB')
        metadata = AutoConfig.from_pretrained(self.config['model'], revision=self.config['revision'])
        if getattr(metadata, 'quantization_config', None):
            raise ValueError('Use the original BF16 checkpoint, not an FP8/4-bit checkpoint.')
        self.revision = getattr(metadata, '_commit_hash', None) or self.config['revision']
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model'], revision=self.revision)
        self.letters = [self.tokenizer.encode(letter, add_special_tokens=False) for letter in 'AB']
        if any(len(ids) != 1 for ids in self.letters) or self.letters[0] == self.letters[1]:
            raise ValueError('A/B must be distinct single-token options for this tokenizer.')
        self.letters = [ids[0] for ids in self.letters]
        longest = 0
        too_long = []
        for split, tasks in self.tasks.items():
            for i, task in enumerate(tasks):
                for reverse in ((False, True) if split == 'train' else (False,)):
                    text = self.tokenizer.apply_chat_template(
                        [{'role': 'user', 'content': prompt(task, self.questions, reverse)}],
                        tokenize=False, add_generation_prompt=True, enable_thinking=False)
                    ids = self.tokenizer(text, add_special_tokens=False, truncation=False)['input_ids']
                    longest = max(longest, len(ids))
                    if len(ids) > self.config['max_length']:
                        too_long.append((split, task['source_id'], len(ids)))
                    self.tokens[(split, i, reverse)] = array('i', ids)
        print(f'Longest prompt: {longest} tokens; configured limit: {self.config["max_length"]}')
        if too_long:
            raise ValueError(f'{len(too_long)} prompts exceed MAX_LENGTH. No text was truncated. '
                             f'Increase MAX_LENGTH to at least {longest}. Examples: {too_long[:3]}')
        max_context = getattr(getattr(metadata, 'text_config', metadata), 'max_position_embeddings', None)
        if max_context and longest > max_context:
            raise ValueError('Data exceeds the checkpoint context length.')
        self.checked = True
        print('Input checks passed. No weights loaded yet.')

    def _base(self):
        """Load the published BF16 model without creating or attaching an adapter."""
        if not self.checked:
            self.check()
        from transformers import AutoModelForImageTextToText
        torch = self.torch
        base = AutoModelForImageTextToText.from_pretrained(
            self.config['model'], revision=self.revision, dtype=torch.bfloat16,
            device_map={'': 0}, attn_implementation='sdpa')
        base.config.use_cache = False
        if hasattr(base.config, 'text_config'):
            base.config.text_config.use_cache = False
        return base

    def _load(self):
        if self.model is not None:
            return
        from peft import LoraConfig, TaskType, get_peft_model
        base = self._base()
        torch = self.torch
        allowed = {'q_proj', 'v_proj', 'gate_proj', 'up_proj', 'down_proj'}
        modules = [name for name, layer in base.named_modules()
                   if name.startswith('model.language_model.')
                   and name.rsplit('.', 1)[-1] in allowed and isinstance(layer, torch.nn.Linear)]
        if not modules:
            raise ValueError('No supported text-backbone LoRA modules found in this checkpoint.')
        self.model = get_peft_model(base, LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=self.config['rank'], lora_alpha=2*self.config['rank'],
            lora_dropout=0.05, bias='none', target_modules=modules, revision=self.revision))
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        self.model.enable_input_require_grads()
        self.model.print_trainable_parameters()
        if any('visual' in n and p.requires_grad for n, p in self.model.named_parameters()):
            raise ValueError('Vision weights must remain frozen.')

    def _logits(self, split, index, reverse=False):
        torch = self.torch
        ids = torch.tensor(list(self.tokens[(split, index, reverse)]), device='cuda', dtype=torch.long)[None]
        result = self.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                            use_cache=False, logits_to_keep=1)
        logits = result.logits[0, -1, self.letters].float()
        if not torch.isfinite(logits).all():
            raise FloatingPointError('Non-finite decision logits.')
        return logits

    def _loss(self, split, index, reverse=False):
        answer = self.tasks[split][index]['answer']
        gold = 1 - answer if reverse else answer
        return self.torch.nn.functional.cross_entropy(
            self._logits(split, index, reverse)[None], self.torch.tensor([gold], device='cuda'))

    def _weights(self):
        from peft import get_peft_model_state_dict
        return {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(self.model).items()}

    def _set_weights(self, state):
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(self.model, state)

    def _identity(self):
        import importlib.metadata
        config = {k: v for k, v in self.config.items() if k not in ('train', 'dev', 'test', 'output')}
        return dict(config=config, revision=self.revision,
                    inputs={k: digest(v) for k, v in self.paths.items() if k != 'test'},
                    files={name: digest(HERE / name) for name in ('run.py', 'questions.json', 'samples.json')},
                    metrics=digest(HERE.parent / 'src/metrics.py'),
                    versions={p: importlib.metadata.version(p) for p in ('torch', 'transformers', 'peft')})

    def _checkpoint(self, optimizer, identity, progress, best, history):
        torch = self.torch
        folder = self.output / 'checkpoint'
        folder.mkdir(exist_ok=True)
        state = dict(identity=identity, progress=progress, weights=self._weights(),
                     optimizer=optimizer.state_dict(), best=best, history=history,
                     rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())
        torch.save(state, folder / 'state.tmp')
        (folder / 'state.tmp').replace(folder / 'state.pt')

    def _history(self, rows):
        with (self.output / 'history.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['epoch', 'step', 'train_loss', 'dev_loss', 'peak_gib'])
            writer.writeheader()
            writer.writerows(rows)

    def train(self, resume=False):
        self._load()
        torch = self.torch
        from tqdm.auto import tqdm
        identity = self._identity()
        state_path = self.output / 'checkpoint/state.pt'
        if resume:
            if not state_path.is_file():
                raise FileNotFoundError(f'No checkpoint at {state_path}. Restore a saved results ZIP first.')
        elif self.output.exists() and any(self.output.iterdir()):
            raise FileExistsError('OUTPUT is not empty. Set RESUME=True or choose a new output folder.')
        self.output.mkdir(parents=True, exist_ok=True)
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=self.config['lr'], weight_decay=0.01)
        progress = dict(epoch=0, offset=0, step=0, loss_sum=0.0, loss_n=0)
        best, history = None, []
        # Smoke the longest train prompt, with gradients, before the full run.
        self.model.train()
        torch.cuda.reset_peak_memory_stats()
        longest = max(range(len(self.tasks['train'])), key=lambda i: len(self.tokens[('train', i, False)]))
        smoke = self._loss('train', longest)
        smoke.backward()
        if not any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().max() > 0 for p in params):
            raise RuntimeError('Smoke check found no finite, nonzero adapter gradients.')
        optimizer.zero_grad(set_to_none=True)
        print(f'Forward/backward passed. Peak VRAM: {torch.cuda.max_memory_allocated()/2**30:.2f} GiB')
        if resume:
            state = torch.load(state_path, map_location='cpu', weights_only=False)
            if state['identity'] != identity:
                raise ValueError('Checkpoint settings/data/code/model versions differ. Use the original setup.')
            self._set_weights(state['weights'])
            optimizer.load_state_dict(state['optimizer'])
            progress, best, history = state['progress'], state['best'], state['history']
            torch.set_rng_state(state['rng'])
            torch.cuda.set_rng_state_all(state['cuda_rng'])
            del state
        else:
            torch.manual_seed(self.config['seed'])
            torch.cuda.manual_seed_all(self.config['seed'])
        settings = dict(config=self.config, identity=identity, questions=self.questions,
                        summary=self.summary, status='training', threshold=None,
                        probability='Restricted A/B softmax; no claim of domain calibration.',
                        empty_selection='abstain, retained as an invalid target in metrics')
        if resume and (self.output / 'settings.json').is_file():
            previous = json.loads((self.output / 'settings.json').read_text(encoding='utf-8'))
            for key in ('threshold', 'test_sha256'):
                if key in previous:
                    settings[key] = previous[key]
        save_json(self.output / 'settings.json', settings)
        n = len(self.tasks['train'])
        for epoch in range(progress['epoch'], self.config['epochs']):
            order = list(range(n))
            random.Random(self.config['seed'] + epoch).shuffle(order)
            start = progress['offset'] if progress['epoch'] == epoch else 0
            self.model.train()
            for offset in tqdm(range(start, n), desc=f'Epoch {epoch+1}/{self.config["epochs"]}', unit='question'):
                i = order[offset]
                reverse = order_for(self.config['seed'], epoch, self.tasks['train'][i])
                loss = self._loss('train', i, reverse)
                group_start = offset // self.config['accumulation'] * self.config['accumulation']
                group_size = min(self.config['accumulation'], n - group_start)
                (loss / group_size).backward()
                progress['loss_sum'] += loss.detach().item()
                progress['loss_n'] += 1
                if (offset + 1) % self.config['accumulation'] == 0 or offset + 1 == n:
                    torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    progress.update(epoch=epoch, offset=offset+1, step=progress['step']+1)
                    if progress['step'] % self.config['save_steps'] == 0:
                        self._checkpoint(optimizer, identity, progress, best, history)
            self.model.eval()
            with torch.inference_mode():
                # None rows have no scope label: do not validate that undefined label.
                valid = [i for i, t in enumerate(self.tasks['dev']) if t['supervised']]
                dev_loss = sum(self._loss('dev', i).item() for i in tqdm(valid, desc='Dev loss')) / len(valid)
            if best is None or dev_loss < best['loss']:
                best = dict(loss=dev_loss, epoch=epoch+1, weights=self._weights())
            history.append(dict(epoch=epoch+1, step=progress['step'],
                                train_loss=progress['loss_sum']/progress['loss_n'], dev_loss=dev_loss,
                                peak_gib=torch.cuda.max_memory_allocated()/2**30))
            self._history(history)
            print(history[-1])
            progress.update(epoch=epoch+1, offset=0, loss_sum=0.0, loss_n=0)
            self._checkpoint(optimizer, identity, progress, best, history)
        self._set_weights(best['weights'])
        self.model.save_pretrained(self.output / 'adapter', safe_serialization=True)
        self.tokenizer.save_pretrained(self.output / 'adapter')
        # Verify that the saved adapter can be reloaded and reproduces decision logits.
        from peft.utils.save_and_load import load_peft_weights
        self.model.eval()
        with torch.inference_mode():
            before = self._logits('dev', 0).detach().clone()
            self._set_weights(load_peft_weights(str(self.output / 'adapter'), device='cpu'))
            after = self._logits('dev', 0)
        if not torch.allclose(before, after, rtol=1e-3, atol=1e-3):
            raise RuntimeError('Saved adapter reload changed the smoke prediction.')
        settings.update(status='trained', best_epoch=best['epoch'], adapter_reload_checked=True)
        save_json(self.output / 'settings.json', settings)
        self._history(history)
        print(f'Adapter saved to {self.output / "adapter"}. Call job.evaluate() next.')

    def _predict(self, split):
        from tqdm.auto import tqdm
        scores = {r[ID_FIELD]: {} for r in self.groups[split]}
        self.model.eval()
        with self.torch.inference_mode():
            for i, task in enumerate(tqdm(self.tasks[split], desc=f'Predict {split}', unit='question')):
                probability = self._logits(split, i).softmax(-1)[0].item()
                scores[task['source_id']][task['key']] = probability
        return [scores[r[ID_FIELD]] for r in self.groups[split]]

    def _write_predictions(self, scores, threshold):
        fields = ['id', 'split', 'origin', 'gold', 'target', 'status', 'p_individual',
                  *['p_' + k for k in IDENTITIES]]
        with (self.output / 'predictions.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for split, values in scores.items():
                for row, p in zip(self.groups[split], values):
                    target, status = decode(p, threshold)
                    writer.writerow(dict(id=row[ID_FIELD], split=split, origin=row['origin'],
                                         gold=canonical_gold(row['target']) if row.get('target') else '',
                                         target=target, status=status,
                                         p_individual=p['scope'], **{'p_'+k: p[k] for k in IDENTITIES}))

    def evaluate(self):
        """Freeze the global threshold on real dev before reading test predictions."""
        self._load()
        from peft.utils.save_and_load import load_peft_weights
        settings_path = self.output / 'settings.json'
        settings = json.loads(settings_path.read_text(encoding='utf-8'))
        if settings['identity'] != self._identity():
            raise ValueError('Evaluation setup differs from the saved training run.')
        self._set_weights(load_peft_weights(str(self.output / 'adapter'), device='cpu'))
        dev_scores = self._predict('dev')
        threshold = settings.get('threshold')
        if threshold is None:
            threshold = choose_threshold(self.groups['dev'], dev_scores)
            settings['threshold'] = threshold
            save_json(settings_path, settings)
        if 'test' in self.paths:
            test_hash = digest(self.paths['test'])
            if settings.get('test_sha256', test_hash) != test_hash:
                raise ValueError('Test data differs from the previous evaluation of this run.')
            settings['test_sha256'] = test_hash
            save_json(settings_path, settings)
        scores = {'dev': dev_scores}
        for split in ('dev_synthetic', 'test'):
            if self.groups.get(split):
                scores[split] = self._predict(split)
        self._write_predictions(scores, threshold)
        metrics = {split: report(self.groups[split], values, threshold) for split, values in scores.items()}
        metrics['protocol'] = dict(threshold=threshold, selected_on='original dev',
                                   checkpoint_selected_on='original dev question loss',
                                   synthetic_dev='diagnostic only; not included in threshold selection',
                                   calibration='none; threshold tuning does not calibrate probabilities')
        save_json(self.output / 'metrics.json', metrics)
        settings['status'] = 'complete'
        save_json(settings_path, settings)
        print(json.dumps({k: {m: v for m, v in r.items() if m in (
            'n', 'full_target_exact_match', 'binary_identity_micro_f1', 'coverage')}
                          for k, r in metrics.items() if k != 'protocol'}, indent=2))
        print(f'Threshold: {threshold:.2f}; results: {self.output}')
        return metrics


class ZeroShot(Run):
    """Published checkpoint only: no training data, synthetic samples or adapters."""
    def __init__(self, data, output='/content/zero-shot', dev='', **options):
        allowed = {'model', 'revision', 'max_length', 'seed'}
        if set(options) - allowed:
            raise ValueError(f'Unknown inference settings: {sorted(set(options)-allowed)}')
        self.config = {key: options.get(key, DEFAULTS[key]) for key in allowed}
        if type(self.config['max_length']) is not int or self.config['max_length'] < 1:
            raise ValueError('max_length must be a positive integer.')
        if type(self.config['seed']) is not int:
            raise ValueError('seed must be an integer.')
        self.output = Path(output).expanduser().resolve()
        self.paths, self.groups = {}, {}
        for split, value in [('data', data), ('dev', dev)]:
            if split == 'dev' and not value:
                continue
            path = Path(value).expanduser().resolve()
            if not value or not path.is_file():
                raise FileNotFoundError(f'Set {split.upper()} to an existing TSV file: {path}')
            if path == self.output or self.output in path.parents:
                raise ValueError('Keep OUTPUT separate from dataset files.')
            rows = [{**row, 'target': row.get('target', ''), 'split': split, 'origin': 'original'}
                    for row in read_tsv(path)]
            if split == 'dev' and not all(row['target'] for row in rows):
                raise ValueError('Optional DEV must have gold targets for threshold selection.')
            self.paths[split], self.groups[split] = path, rows
        if 'dev' in self.groups:
            ids = {row[ID_FIELD] for row in self.groups['data']}
            comments = {' '.join(row['yt_comment'].casefold().split()) for row in self.groups['data']}
            if any(row[ID_FIELD] in ids or ' '.join(row['yt_comment'].casefold().split()) in comments
                   for row in self.groups['dev']):
                raise ValueError('DEV for threshold selection must be separate from DATA (IDs/comments).')
        self.questions = json.loads((HERE / 'questions.json').read_text(encoding='utf-8'))
        if tuple(self.questions['identities']) != IDENTITIES:
            raise ValueError('Questions must cover all nine identities in canonical order.')
        self.tasks = {split: list(decisions(rows, self.questions, inference=True))
                      for split, rows in self.groups.items()}
        self.model, self.tokens, self.checked = None, {}, False
        print({split: len(rows) for split, rows in self.groups.items()})
        print('Zero-shot: published BF16 weights; no task training, examples or adapter.')

    def _load(self):
        if self.model is None:
            self.model = self._base()
            self.model.requires_grad_(False)
            self.model.eval()

    def train(self, *args, **kwargs):
        raise RuntimeError('ZeroShot does not train. Use Run with train.ipynb for fine-tuning.')

    def evaluate(self):
        raise RuntimeError('Use job.infer() with inference.ipynb for zero-shot inference.')

    def infer(self, threshold=0.5, tune_on_dev=False):
        if not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 < threshold < 1:
            raise ValueError('threshold must be between 0 and 1.')
        if tune_on_dev and 'dev' not in self.groups:
            raise ValueError('Set a separate labeled DEV file to tune the threshold.')
        if self.output.exists() and any(self.output.iterdir()):
            raise FileExistsError('OUTPUT is not empty. Choose a new folder for each comparison.')
        self._load()
        self.output.mkdir(parents=True, exist_ok=True)
        import importlib.metadata
        settings = dict(mode='zero_shot_dev_threshold' if tune_on_dev else 'zero_shot',
                        config=self.config, resolved_revision=self.revision, threshold=threshold,
                        threshold_source='original dev' if tune_on_dev else 'fixed before inference',
                        questions=self.questions, adapters=False, training=False, synthetic_samples=False,
                        inputs={k: digest(v) for k, v in self.paths.items()},
                        code_sha256=digest(HERE / 'run.py'),
                        versions={p: importlib.metadata.version(p) for p in ('torch', 'transformers')},
                        probability='Restricted A/B softmax; no domain calibration.', status='running')
        save_json(self.output / 'settings.json', settings)
        try:
            scores = {}
            if tune_on_dev:
                scores['dev'] = self._predict('dev')
                threshold = choose_threshold(self.groups['dev'], scores['dev'])
                settings['threshold'] = threshold
                save_json(self.output / 'settings.json', settings)
            scores['data'] = self._predict('data')
            self._write_predictions(scores, threshold)
            metrics = {split: report(self.groups[split], values, threshold) for split, values in scores.items()}
            metrics['protocol'] = dict(mode=settings['mode'], threshold=threshold,
                                       selected_on=settings['threshold_source'], calibration='none',
                                       task_training=False, adapter=False, synthetic_samples=False)
            save_json(self.output / 'metrics.json', metrics)
            settings['status'] = 'complete'
            print(json.dumps(metrics['data'], indent=2))
            print(f'Threshold: {threshold:.2f}; zero-shot results: {self.output}')
            return metrics
        except Exception as exc:
            settings.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            save_json(self.output / 'settings.json', settings)
