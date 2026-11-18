"""One isolated, sleeping vLLM engine for the lifetime of a training run."""
import copy
import hashlib
import json
import multiprocessing
import os
import signal
import time
import traceback
import uuid

from ..common import path, read_jsonl, write_jsonl


def engine_signature(cfg):
    model = {k: v for k, v in cfg['model'].items()
             if k not in {'adapter', 'adapter_revision'}}
    inference = {k: v for k, v in cfg['inference'].items() if k not in {'output', 'resume'}}
    return hashlib.sha256(json.dumps(dict(model=model, inference=inference,
        prompt=path(cfg['prompt_file']).read_text(encoding='utf-8'),
        system=cfg.get('system_prompt'), peft=cfg['peft'], seed=cfg['training']['seed']),
        sort_keys=True).encode()).hexdigest()


class ValidationSession:
    def __init__(self, cfg):
        from ..models.vllm_loading import load_vllm
        self.signature = engine_signature(cfg)
        self.cfg = copy.deepcopy(cfg)
        if self.cfg['inference'].get('vllm_cpu_offload_gb', 0):
            raise ValueError('Persistent validation uses no CPU weight offload during generation')
        print('[validation] Initializing base + LoRA vLLM engine (once per training run)', flush=True)
        self.generate = load_vllm(self.cfg, persistent=True)
        self.engine = self.generate.engine
        self.sleeping = False

    def predict(self, cfg):
        from swift.llm import AdapterRequest
        from tqdm import tqdm
        if engine_signature(cfg) != self.signature:
            raise ValueError('Persistent vLLM base/prompt/generation changed during training')
        adapter = path(cfg['model']['adapter'])
        metadata = json.loads((adapter / 'adapter_config.json').read_text(encoding='utf-8'))
        if metadata['base_model_name_or_path'].rstrip('/') != cfg['model']['name'].rstrip('/'):
            raise ValueError('Validation adapter does not match the frozen vLLM base')
        if metadata['r'] > cfg['peft']['r']:
            raise ValueError('Validation adapter exceeds engine max_lora_rank')
        # Update paths to the NEW snapshot/prompt before the preceding temp directory disappears.
        self.cfg.clear()
        self.cfg.update(copy.deepcopy(cfg))
        if self.sleeping:
            print('[validation] Waking vLLM; loading current LoRA snapshot', flush=True)
            self.engine.wake_up()
            self.sleeping = False
        try:
            self.engine.reset_prefix_cache()
            # A new name means a new LoRA ID in Swift; never reuse a cached adapter identity.
            request = AdapterRequest('validation-' + uuid.uuid4().hex, str(adapter))
            print(f'[validation] Current LoRA: {request.name} -> {adapter}', flush=True)
            split = cfg['inference'].get('split', cfg['data']['eval_split'])
            rows = read_jsonl(path(cfg['data']['manifest_dir']) / f'{split}.jsonl')
            output = []
            batch_size = cfg['inference']['batch_size']
            with tqdm(total=len(rows), desc='Validation vLLM', unit='question', dynamic_ncols=True) as bar:
                for offset in range(0, len(rows), batch_size):
                    batch = rows[offset:offset + batch_size]
                    answers = self.generate(batch, adapter_request=request)
                    if len(answers) != len(batch):
                        raise ValueError('vLLM returned an incorrect number of answers')
                    output.extend({'id': row['id'], 'answer': answer} for row, answer in zip(batch, answers))
                    bar.update(len(batch))
            write_jsonl(cfg['inference']['output'], output)
        finally:
            # Remove the snapshot before its directory is deleted; keep only the frozen base.
            for adapter_id in list(self.engine.list_loras()):
                self.engine.remove_lora(adapter_id)
            self.engine.reset_prefix_cache()
            self.engine.sleep(level=1)
            self.sleeping = True
            print('[validation] vLLM sleeping; GPU available for scoring/training', flush=True)

    def close(self):
        # Swift 3.11.3 / vLLM 0.11.0 V1 engine exposes this synchronous shutdown API.
        self.engine.engine_core.shutdown()


def serve(connection):
    session = None
    try:
        while True:
            message = connection.recv()
            if message['command'] == 'stop':
                break
            if message['command'] != 'predict':
                raise ValueError('Unknown validation worker command')
            cfg = json.loads(path(message['config']).read_text(encoding='utf-8'))
            if session is None:
                session = ValidationSession(cfg)
            session.predict(cfg)
            # Acknowledgement is sent ONLY after the engine is asleep.
            connection.send({'ok': True})
    except EOFError:
        pass
    except BaseException:
        error = traceback.format_exc()
        if session is not None:
            session.close()
            session = None
        connection.send({'ok': False, 'error': error})
    finally:
        if session is not None:
            session.close()
        connection.close()


def worker_entry(connection):
    if os.name == 'posix':
        os.setsid()  # Isolate worker + engine children for cleanup after interruption.
    serve(connection)


class PersistentValidationWorker:
    def __init__(self, timeout=3600):
        self.timeout = timeout
        self.process = None
        self.connection = None

    def predict(self, config):
        if self.process is None:
            context = multiprocessing.get_context('spawn')
            self.connection, child = context.Pipe()
            self.process = context.Process(target=worker_entry, args=(child,))
            self.process.start()
            child.close()
        try:
            self.connection.send({'command': 'predict', 'config': str(config)})
            deadline = time.monotonic() + self.timeout
            while not self.connection.poll(0.25):
                if not self.process.is_alive():
                    raise RuntimeError(f'vLLM validation worker exited: {self.process.exitcode}')
                if time.monotonic() > deadline:
                    raise TimeoutError('vLLM validation worker timed out')
            result = self.connection.recv()
            if not result['ok']:
                raise RuntimeError(result['error'])
        except BaseException:
            # Stop the whole GPU worker group BEFORE restoring training tensors to GPU.
            self.close()
            raise

    def close(self):
        if self.process is None:
            return
        try:
            if self.process.is_alive():
                try:
                    self.connection.send({'command': 'stop'})
                except (BrokenPipeError, EOFError, OSError):
                    pass
                self.process.join(timeout=15)
            if self.process.is_alive():
                if os.name == 'posix' and os.getpgid(self.process.pid) == self.process.pid:
                    os.killpg(self.process.pid, signal.SIGTERM)
                else:
                    self.process.terminate()
                self.process.join(timeout=5)
                if self.process.is_alive():
                    if os.name == 'posix' and os.getpgid(self.process.pid) == self.process.pid:
                        os.killpg(self.process.pid, signal.SIGKILL)
                    else:
                        self.process.kill()
                    self.process.join(timeout=5)
        finally:
            self.connection.close()
            self.process = None
            self.connection = None
