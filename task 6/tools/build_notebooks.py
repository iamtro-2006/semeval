"""Generate clean Colab notebooks using only the Python standard library."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def md(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': text.splitlines(True)}


def code(text):
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': text.splitlines(True)}


setup = [md('''# Task 6 · English QA
Clone the shared semeval repository and install the package in `task 6/`.
No code ZIP or local dataset upload. Edit REPO_URL / GIT_REF in the cell below.
Use Runtime → Change runtime type → GPU for training. The transfer notebook can run on CPU.
'''), code('''# Repository settings: editable in the notebook.
REPO_URL = 'https://github.com/iamtro-2006/semeval.git'
GIT_REF = 'main'  # branch, tag, or commit
CHECKOUT = '/content/semeval'
PROJECT_SUBDIR = 'task 6'

from pathlib import Path
import os, subprocess, sys
import codecs

def stream_run(cmd, check=True, cwd=None):
    # Route child output through IPython stdout, rather than its hidden OS file descriptor.
    env = dict(os.environ, PYTHONUNBUFFERED='1')
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                               cwd=cwd if cwd is not None else globals().get('PROJECT_ROOT'))
    decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
    try:
        while True:
            chunk = os.read(process.stdout.fileno(), 4096)
            if not chunk:
                break
            sys.stdout.write(decoder.decode(chunk))
            sys.stdout.flush()
        sys.stdout.write(decoder.decode(b'', final=True))
        sys.stdout.flush()
        result = subprocess.CompletedProcess(cmd, process.wait())
    except BaseException:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise
    finally:
        process.stdout.close()
    if check:
        result.check_returncode()
    return result

checkout = Path(CHECKOUT).expanduser()
if not checkout.is_absolute():
    raise ValueError('CHECKOUT must be absolute, e.g. /content/semeval')
checkout = checkout.resolve()
if not checkout.exists():
    checkout.parent.mkdir(parents=True, exist_ok=True)
    stream_run(['git', 'clone', REPO_URL, str(checkout)], check=True, cwd=checkout.parent)
else:
    actual = subprocess.check_output(['git', '-C', str(checkout), 'remote', 'get-url', 'origin'], text=True).strip()
    if actual != REPO_URL:
        raise ValueError('CHECKOUT belongs to a different repository; choose another CHECKOUT')
    dirty = subprocess.check_output(['git', '-C', str(checkout), 'status', '--porcelain'], text=True)
    if dirty.strip():
        raise RuntimeError('Checkout has edits; preserve your changes or choose a fresh CHECKOUT before switching revision')
stream_run(['git', '-C', str(checkout), 'fetch', 'origin', GIT_REF], check=True, cwd=checkout.parent)
stream_run(['git', '-C', str(checkout), 'checkout', '--detach', 'FETCH_HEAD'], check=True, cwd=checkout.parent)
PROJECT_ROOT = (checkout / PROJECT_SUBDIR).resolve()
SRC = PROJECT_ROOT / 'src'
if not (SRC / 'mmcultureqa/__init__.py').is_file():
    raise FileNotFoundError(f'Project package not found: {SRC / "mmcultureqa"}')
os.chdir(PROJECT_ROOT)
# Editable installation alone does not refresh sys.path in an already running kernel.
# Export PYTHONPATH too, so child Python processes and workers use this same checkout.
sys.path[:] = [str(SRC)] + [p for p in sys.path if p != str(SRC)]
python_paths = [p for p in os.environ.get('PYTHONPATH', '').split(os.pathsep) if p and p != str(SRC)]
os.environ['PYTHONPATH'] = os.pathsep.join([str(SRC), *python_paths])
import importlib
importlib.invalidate_caches()
for name in list(sys.modules):
    if name == 'mmcultureqa' or name.startswith('mmcultureqa.'):
        del sys.modules[name]
stream_run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--no-build-isolation', '-e', str(PROJECT_ROOT)], check=True)
from mmcultureqa.common import ROOT as PACKAGE_ROOT
if PACKAGE_ROOT.resolve() != PROJECT_ROOT:
    raise ImportError(f'mmcultureqa loaded from {PACKAGE_ROOT}, expected {PROJECT_ROOT}')
stream_run([sys.executable, '-c', 'import mmcultureqa; print("Child package:", mmcultureqa.__file__)'])
print('Project:', Path.cwd())
print('Commit:', subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip())
''')]

# Shared storage setup: every notebook uses the same folder ID and relative layout.
storage = [md('''## MMCQA folder and authentication
Only ROOT_FOLDER_ID selects storage. Paste the ID or URL of your MMCQA folder.
RCLONE_CONFIG_PATH points to your authenticated rclone.conf; this is not a storage root.
All data, adapters, checkpoints and results are addressed relative to the selected ID.
If your config is on mounted Drive, run the optional mount cell first.
'''), code('''# Optional, only to access a config file stored on Drive:
# from google.colab import drive
# drive.mount('/content/drive')
'''), code('''ROOT_FOLDER_ID = '1kcdByZiSJzGSoEtY9q0OEhsZwiTZkxpV'  # ID or URL of the MMCQA folder
RCLONE_CONFIG_PATH = '/content/rclone.conf'
REMOTE = 'iamtro'  # Google Drive remote in rclone.conf
'''), code('''stream_run(['sudo', 'apt-get', '-qq', 'update'])
stream_run(['sudo', 'apt-get', '-qq', 'install', '-y', 'rclone'])
import configparser, re, tempfile
from urllib.parse import urlparse
config_source = Path(RCLONE_CONFIG_PATH).expanduser()
if not config_source.is_absolute():
    raise ValueError('RCLONE_CONFIG_PATH must be absolute, e.g. /content/rclone.conf')
if not config_source.is_file():
    raise FileNotFoundError(f'Upload your authenticated rclone.conf or mount Drive: {config_source}')
sections = configparser.RawConfigParser()
sections.read(config_source, encoding='utf-8-sig')
remotes = [name for name in sections.sections() if sections[name].get('type') == 'drive']
if not REMOTE:
    if len(remotes) != 1:
        raise ValueError(f'Select REMOTE from Google Drive sections: {remotes}')
    REMOTE = remotes[0]
if REMOTE not in remotes:
    raise ValueError('REMOTE must be a Google Drive section')
root_id = ROOT_FOLDER_ID.strip()
if root_id.startswith(('https://', 'http://')):
    match = re.search(r'/folders/([^/?]+)', urlparse(root_id).path)
    root_id = match.group(1) if match else ''
if not re.fullmatch(r'[A-Za-z0-9_-]+', root_id):
    raise ValueError('Enter the ID or folder URL of MMCQA in ROOT_FOLDER_ID')
temporary_config = tempfile.TemporaryDirectory(prefix='mmcqa-rclone-')
config_path = Path(temporary_config.name) / 'rclone.conf'
config_path.write_bytes(config_source.read_bytes())
config_path.chmod(0o600)
rclone_base = ['rclone', '--config', str(config_path), '--drive-root-folder-id', root_id]
def cloud(relative):
    if relative.startswith('/') or '..' in Path(relative).parts:
        raise ValueError('Storage paths must be relative to MMCQA')
    return f'{REMOTE}:{relative}'
def copy(source, destination, *extra):
    stream_run(rclone_base + ['copy', str(source), str(destination),
                   '--transfers', '4', '--checkers', '8', '--progress', *extra], check=True)
stream_run(rclone_base + ['lsd', cloud(''), '--max-depth', '1'], check=True)
print('All storage is relative to MMCQA folder ID:', root_id)
''')]

data_setup = [md('''## Prepare data on Colab SSD
The download location is automatically derived from ROOT_FOLDER_ID.
'''), code('''USE_DRIVE_DATA = True
if USE_DRIVE_DATA:
    copy(cloud('task6-data/raw'), PROJECT_ROOT / 'data/raw')
else:
    stream_run([sys.executable, '-m', 'mmcultureqa.data.prepare', 'download'], check=True)
stream_run([sys.executable, '-m', 'mmcultureqa.data.prepare', 'prepare'], check=True)
''')]

gpu_setup = setup + [md('''## Install GPU environment
Restart the runtime if requested, then rerun clone/setup. GPU training has not been run on the CPU-only development machine.
'''), code('''INSTALL_VLLM = False  # enable only for optional BF16/vLLM inference
requirements = 'requirements-vllm.txt' if INSTALL_VLLM else 'requirements.txt'
stream_run([sys.executable, '-m', 'pip', 'install', '-r', str(PROJECT_ROOT / requirements)])
# Colab may have unrelated preinstalled-package conflicts; print the full report.
stream_run([sys.executable, '-m', 'pip', 'check'], check=False)
stream_run(['nvidia-smi'])
'''), code('''# Fail early on dependency/import errors before downloading data or starting training.
import subprocess, sys
stream_run([sys.executable, '-c', """
import sys
from importlib.metadata import version
for name in ('numpy', 'torch', 'transformers', 'peft', 'trl', 'ms-swift', 'bitsandbytes', 'msgspec'):
    print(name, version(name), flush=True)
from transformers.modeling_utils import PreTrainedModel
from peft import PeftModel
from swift.llm import TrainArguments, RLHFArguments
import qwen_vl_utils
import decord
from bitsandbytes.nn import Linear4bit
from swift.llm import PtEngine
import mmcultureqa.common
import mmcultureqa.data.prepare
import mmcultureqa.training.sft
import mmcultureqa.training.grpo
import mmcultureqa.training.curves
import mmcultureqa.inference.predict
import mmcultureqa.evaluate
from swift.llm.train.sft import SwiftSft
from swift.llm.train.rlhf import SwiftRLHF
# Import the concrete trainer too: SwiftRLHF alone leaves its dependencies lazy.
from swift.trainers.rlhf_trainer.grpo_trainer import GRPOTrainer
from msgspec import field
from swift.llm import InferRequest, RequestConfig
if sys.argv[1] == '1':
    print('vllm', version('vllm'), flush=True)
    from vllm import LLM
    from swift.llm import VllmEngine
from swift.plugin import metric_mapping, orms, ORM
from swift.utils import Serializer
print('Training imports OK', flush=True)
""", '1' if INSTALL_VLLM else '0'])
''')] + storage + data_setup + [md('''## Experiment settings
Only ROOT_FOLDER_ID determines cloud storage. RUN_NAME groups your experiments under it.
Intermediate checkpoints stay on local SSD; only the selected final adapter uploads after successful training.
Uploads use explicit allowlists: adapter/processor, run settings, learning reports and validation results.
Optimizer states, exported training data, caches and temporary evaluation files stay local.
Local SSD is used for training and inference. If the runtime is lost before the final upload, local checkpoints cannot be recovered from Drive.
SFT and GRPO default to NF4 QLoRA with BF16 compute and native Transformers generation.
GRPO reuses the training Qwen model for generation, with no second vLLM Qwen and no CPU offload.
Both stages retain CUDA FP16 mmBERT scoring.
SFT's optional in-training generation monitor uses the native ms-swift Transformers path.
Final validation uses Transformers when USE_4BIT=True, otherwise vLLM.
These are unbenchmarked GPU starting configurations; measure throughput and peak memory on Colab.
'''), code('''import json
USE_4BIT = True
PREDICT_WITH_GENERATE = True  # SFT generation + mmBERT; False selects teacher-forced validation
RUN_FULL = True  # full training from YAML; False explicitly selects a one-step smoke run
RUN_NAME = 'experiment01'
if not re.fullmatch(r'[A-Za-z0-9_-]+', RUN_NAME):
    raise ValueError('RUN_NAME must contain only letters, numbers, underscores or hyphens')
RUNS = PROJECT_ROOT / 'outputs/runs' / RUN_NAME
RUNS.mkdir(parents=True, exist_ok=True)
SFT_STAGE = 'sft' if RUN_FULL else 'sft-smoke'
GRPO_STAGE = 'grpo' if RUN_FULL else 'grpo-smoke'
SFT = str(RUNS / SFT_STAGE)
GRPO = str(RUNS / GRPO_STAGE)
common = {'model.dtype': 'bfloat16', 'model.load_in_4bit': USE_4BIT,
          'training.predict_with_generate': PREDICT_WITH_GENERATE}

def upload_run(local_output, cloud_subdir):
    from mmcultureqa.storage import upload_filters
    copy(local_output, cloud(cloud_subdir), *upload_filters('run'))

def upload_predictions(local_output, cloud_subdir):
    from mmcultureqa.storage import upload_filters
    copy(local_output, cloud(cloud_subdir), *upload_filters('predictions'))

def run_command(module, config, overrides):
    config = Path(config)
    if not config.is_absolute():
        config = PROJECT_ROOT / config
    if not config.is_file():
        raise FileNotFoundError(f'Configuration not found: {config}')
    cmd = [sys.executable, '-m', module, '--config', str(config)]
    for key, value in overrides.items():
        cmd += ['--set', f'{key}={json.dumps(value)}']
    stream_run(cmd, check=True)

def require_adapter(folder):
    folder = Path(folder)
    if not (folder / 'adapter_config.json').is_file():
        raise FileNotFoundError(f'Adapter config missing: {folder}')
    if not any((folder / name).is_file() for name in ('adapter_model.safetensors', 'adapter_model.bin')):
        raise FileNotFoundError(f'Adapter weights missing: {folder}')

def run(module, config, overrides, local_output, cloud_subdir):
    run_command(module, config, overrides)
    require_adapter(Path(local_output) / 'final')
    upload_run(local_output, cloud_subdir)
''')]

transfer = setup + storage + [md('''## Download English QA and upload automatically below MMCQA
One shared data backup: train/dev/devtest English JSONL, image ZIPs and pinned REVISION.
Parquet duplicates, extracted images and download caches stay local.
All later notebooks use the same relative data path; no dataset copy per experiment.
'''), code('''stream_run([sys.executable, '-m', 'pip', 'install', 'huggingface-hub==0.36.0', 'Pillow==11.3.0'])
stream_run([sys.executable, '-m', 'mmcultureqa.data.prepare', 'download'])
stream_run([sys.executable, '-m', 'mmcultureqa.data.prepare', 'prepare'])
'''), code('''from mmcultureqa.storage import upload_filters
filters = upload_filters('data')
copy('data/raw', cloud('task6-data/raw'), *filters)
stream_run(rclone_base + ['check', 'data/raw', cloud('task6-data/raw'),
                '--one-way', '--size-only', *filters], check=True)
print('Dataset ready under MMCQA/task6-data/raw')
''')]

def experiment03_profile(cells, inference=False):
    result = []
    for cell in cells:
        source = ''.join(cell['source'])
        if cell['cell_type'] == 'code':
            source = source.replace("RUN_NAME = 'experiment01'", "RUN_NAME = 'experiment03'", 1)
            source = source.replace('INSTALL_VLLM = False  # enable only for optional BF16/vLLM inference',
                                    'INSTALL_VLLM = True  # final generation uses vLLM after training exits')
            source = source.replace('PREDICT_WITH_GENERATE = True', 'PREDICT_WITH_GENERATE = False', 1)
            source = source.replace('USE_4BIT = True', 'USE_4BIT = False', 1)
            source = source.replace("upload_filters('run')", "upload_filters('merged_run')")
            if inference:
                source = source.replace('USE_4BIT = True', 'USE_4BIT = False', 1)
            # This profile exports a complete BF16 model; legacy adapter launcher belongs to GRPO.
            source = source.partition('\ndef run(module, config, overrides, local_output, cloud_subdir):')[0]
            cell = code(source)
        elif source.startswith('## Experiment settings'):
            cell = md('''## Experiment 03 settings
SFT trains BF16 LoRA with microbatch 8 and accumulation 1.
Training logs loss/token accuracy only; it NEVER starts vLLM or mmBERT.
Retain every checkpoint. After training exits, merge each checkpoint and generate full validation.
Selection uses the same merged-model vLLM path/settings as notebook 04, with CUDA graphs enabled.
No runtime LoRA, sleep/wake or CPU weight offload during generation.
Only one temporary merged candidate is kept at a time; export the F1 winner at the end.
BF16 8B training may require more VRAM than an L4.
Default 1 epoch, peak learning rate 5e-5 (3% warmup), eval/save every 125 optimizer steps.
After training exits, merge the selected local adapter into a fresh BF16 base and upload best_model/.
Final validation and test inference use that complete BF16 model via vLLM in a separate process.
Drive uploads exclude local adapters, intermediate checkpoints, optimizer and temporary files.
Use the same ROOT_FOLDER_ID and RUN_NAME for SFT and inference. GRPO is unchanged and deferred.
''')
        result.append(cell)
    return result


sft_notebook = experiment03_profile(gpu_setup) + [md('''## SFT only · experiment03
BF16 LoRA; training uses teacher-forced loss/token accuracy only. No vLLM during training.
BF16 base; microbatch 8 x accumulation 1 preserves effective batch 8 on one GPU.
Save every 125 steps; retain ALL checkpoints. Periodic loss/token accuracy uses full validation.
After 1 training epoch, merge and score each checkpoint on all 1,000 validation questions.
Select the highest merged-model F1, then export that winner as best_model/.
Language LoRA already covers q/k/v/o plus gate/up/down projections, not only key/value.
ViT stays frozen; final inference uses a merged model without runtime LoRA.
best_model/ is a complete BF16 checkpoint, not a 4-bit checkpoint or a standalone adapter.
Edit SFT_OVERRIDES for prompt and training parameters. Resume uses a full LOCAL checkpoint.
Automatic uploads contain no resume state.
'''), code('''TRAIN_PROJECTOR = False  # Optional projector LoRA; all trained modules merge before generation
BASE_MODEL_RUN_NAME = ''  # empty: original Qwen; otherwise start a NEW adapter on that run's merged SFT model
SFT_OVERRIDES = {}  # optional native project config overrides
RESUME_CHECKPOINT = ''  # local checkpoint-N; final/ is not a resumable checkpoint
from mmcultureqa.models.artifacts import require_merged_model
MODEL_SIZE = '4B'  # '4B' or '8B'; both use configs/sft.yaml
MODEL_PROFILES = {
    '4B': ('Qwen/Qwen3-VL-4B-Instruct', 'ebb281ec70b05090aa6165b016eac8ec08e71b17'),
    '8B': ('Qwen/Qwen3-VL-8B-Instruct', '0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'),
}
model_name, model_revision = MODEL_PROFILES[MODEL_SIZE]
sft_args = {**common, 'model.name': model_name, 'model.revision': model_revision,
            'peft.train_projector': TRAIN_PROJECTOR, 'training.output_dir': SFT}
if BASE_MODEL_RUN_NAME:
    if not re.fullmatch(r'[A-Za-z0-9_-]+', BASE_MODEL_RUN_NAME):
        raise ValueError('BASE_MODEL_RUN_NAME must be a run name')
    base = PROJECT_ROOT / 'outputs/inputs/bases' / BASE_MODEL_RUN_NAME / 'best_model'
    copy(cloud(f'task6-runs/{BASE_MODEL_RUN_NAME}/sft/best_model'), base)
    require_merged_model(base)
    sft_args.update({'model.name': str(base), 'model.revision': None,
                     'model.adapter': None, 'model.adapter_revision': None,
                     'model.local_files_only': True})
if not RUN_FULL:
    sft_args.update({'data.limit': 8, 'training.max_steps': 1, 'training.eval_strategy': 'no',
                     'training.load_best_model_at_end': False, 'training.save_steps': 1})
if RESUME_CHECKPOINT:
    if not (Path(SFT) / RESUME_CHECKPOINT / 'trainer_state.json').is_file():
        raise FileNotFoundError('Local checkpoint missing; restore a full checkpoint before resuming')
    sft_args['resume_from_checkpoint'] = str(Path(SFT) / RESUME_CHECKPOINT)
sft_args.update(SFT_OVERRIDES)
run_command('mmcultureqa.training.sft', 'configs/sft.yaml', sft_args)
require_adapter(Path(SFT) / 'final')
print('Training finished; all checkpoints remain local for merged-model F1 selection.')
'''), code('''# Training has exited: merge and score ALL saved checkpoints sequentially.
# Completed candidate scores are reused if this cell is interrupted and rerun.
stream_run([sys.executable, '-m', 'mmcultureqa.select_checkpoint', '--run-dir', SFT])
require_merged_model(Path(SFT) / 'best_model')
upload_run(SFT, f'task6-runs/{RUN_NAME}/{SFT_STAGE}')
print('Merged BF16 model on Drive:', cloud(f'task6-runs/{RUN_NAME}/{SFT_STAGE}/best_model'))
print('Notebook 04: select the same ROOT_FOLDER_ID, INPUT_RUN_NAME and MODEL_STAGE.')
''')]

def grpo_profile(cells):
    result = []
    for cell in cells:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            source = source.replace("RUN_NAME = 'experiment01'", "RUN_NAME = 'experiment01_qlora'", 1)
            cell = code(source)
        result.append(cell)
    return result


grpo_notebook = grpo_profile(gpu_setup) + [md('''## GRPO only · ms-swift + PEFT QLoRA
Base weights: bitsandbytes NF4 + double quantization; BF16 compute, only LoRA is trained.
USE_4BIT=True; training.use_vllm=False. Native PtEngine shares the training model.
Single GPU default: train microbatch 4, accumulation 4, rollout 16 completions (4 questions), eval batch 8.
Keep 4 completions/question and the same mmBERT-F1 reward. No model/optimizer offload or vLLM sleep.
If GPU memory is insufficient, try train batch 2 and eval batch 4 in GRPO_OVERRIDES.
Use a new RUN_NAME for this recipe; SFT_RUN_NAME separately selects the existing SFT adapter.
Full GRPO training defaults to 1 epoch, with validation and checkpoint saving every 125 optimizer steps.
Validation rollouts are enabled independently of the SFT flag. Increasing epochs preserves best selection.
Select the highest validation reward checkpoint and export it as final/. Inspect
eval_rewards/MMCQABERTScore/mean for raw mmBERT F1 (sampled responses, not teacher-forced accuracy).
GRPO does not have the supervised token-loss/accuracy semantics of SFT.
The explicit RUN_FULL=False smoke path disables periodic validation and best selection.
Paths are automatically derived from MMCQA root. This notebook never trains SFT.
A full GRPO experiment should use a full SFT checkpoint, not a smoke adapter.
'''), code('''SFT_RUN_NAME = 'experiment01'  # existing SFT run; independent of the new QLoRA output RUN_NAME
INPUT_SFT_STAGE = 'sft' if RUN_FULL else 'sft-smoke'
SFT_CHECKPOINT = 'final'  # or checkpoint-N
GRPO_OVERRIDES = {}  # OOM alternative: {'training.per_device_train_batch_size': 2, 'training.per_device_eval_batch_size': 4}
RESUME_CHECKPOINT = ''  # manually backed-up GRPO checkpoint-N only
local_adapter = PROJECT_ROOT / 'outputs/inputs/sft' / SFT_RUN_NAME / INPUT_SFT_STAGE / SFT_CHECKPOINT
copy(cloud(f'task6-runs/{SFT_RUN_NAME}/{INPUT_SFT_STAGE}/{SFT_CHECKPOINT}'), local_adapter)
require_adapter(local_adapter)
grpo_args = {**common, 'training.predict_with_generate': True,
             'model.adapter': str(local_adapter), 'model.adapter_revision': None,
             'training.output_dir': GRPO}
if not RUN_FULL:
    grpo_args.update({'data.limit': 8, 'training.max_steps': 1, 'training.save_steps': 1,
                      'training.eval_strategy': 'no', 'training.predict_with_generate': False,
                      'training.load_best_model_at_end': False})
if RESUME_CHECKPOINT:
    copy(cloud(f'task6-runs/{RUN_NAME}/{GRPO_STAGE}'), GRPO)
    grpo_args['resume_from_checkpoint'] = str(Path(GRPO) / RESUME_CHECKPOINT)
grpo_args.update(GRPO_OVERRIDES)
run('mmcultureqa.training.grpo', 'configs/grpo.yaml', grpo_args, GRPO, f'task6-runs/{RUN_NAME}/{GRPO_STAGE}')
'''), code('''if not (Path(GRPO) / 'final/adapter_config.json').is_file():
    raise FileNotFoundError('No final GRPO adapter was saved')
print('GRPO adapter on Drive:', cloud(f'task6-runs/{RUN_NAME}/{GRPO_STAGE}/final'))
''')]

def final_validation(run_variable, stage_variable, merged=False):
    result = [md('''## Final held-out validation score
Generate answers with the saved final adapter and score with the unchanged official scorer.
Default: ALL 1,000 validation examples, even after a one-step training smoke run.
Set VAL_LIMIT=4 only to smoke-test evaluation; subset results are explicitly marked.
BERTScore-F1 is the ranking metric, not exact-match accuracy. No weights are updated.
'''), code('''VAL_LIMIT = None  # full validation; optionally 4 for an evaluation smoke test
validation_run = Path(__RUN_DIR__)
cmd = [sys.executable, '-m', 'mmcultureqa.evaluate', '--run-dir', str(validation_run)]
cmd += ['--backend', 'transformers' if USE_4BIT else 'vllm']
if VAL_LIMIT is not None:
    cmd += ['--limit', str(VAL_LIMIT)]
result = stream_run(cmd, check=False)
upload_run(validation_run, f'task6-runs/{RUN_NAME}/{__STAGE__}')
result.check_returncode()
'''.replace('__RUN_DIR__', run_variable).replace('__STAGE__', stage_variable))]
    if merged:
        result[0] = md(''.join(result[0]['source']).replace('saved final adapter', 'complete BF16 best_model'))
        result[1] = code(''.join(result[1]['source']).replace(
            "cmd += ['--backend', 'transformers' if USE_4BIT else 'vllm']",
            "cmd += ['--backend', 'vllm']"))
    return result

def learning_curves(run_variable, stage_variable, label):
    return [md('''## Learning curves before final evaluation
Plot native training/validation metrics against optimizer step, then upload the PNG to Drive.
Validation points appear only at completed evaluation intervals; no interpolation or smoothing.
'''), code('''import sys
from pathlib import Path

from mmcultureqa.training.curves import plot_learning_curves

local_run = Path(__RUN_DIR__)
if not local_run.is_absolute():
    local_run = PROJECT_ROOT / local_run
plot_learning_curves(local_run, '__LABEL__')
upload_run(local_run, f'task6-runs/{RUN_NAME}/{__STAGE__}')
'''.replace('__RUN_DIR__', run_variable).replace('__STAGE__', stage_variable).replace('__LABEL__', label))]


sft_notebook += learning_curves('SFT', 'SFT_STAGE', 'sft')
sft_notebook += [code("print((Path(SFT) / 'selection.json').read_text(encoding='utf-8'))\nprint((Path(SFT) / 'validation/scores.json').read_text(encoding='utf-8'))\n")]
grpo_notebook += learning_curves('GRPO', 'GRPO_STAGE', 'grpo')
grpo_notebook += final_validation('GRPO', 'GRPO_STAGE')

inference_notebook = experiment03_profile(gpu_setup, inference=True) + [md('''## Test inference from the merged BF16 model
Download the selected run's best_model from Drive. No original Qwen/adapter download is needed.
Model and processor loading are local-files-only; missing files raise an error instead of falling back to the Hub.
Uses the saved prompt and preprocessing from the exported training run.
Supports merged Qwen3-VL-4B and Qwen3-VL-8B checkpoints.
Set INPUT_RUN_NAME to the training run to download; model family and prompt restore from that artifact.
Validation scoring belongs to the SFT/GRPO notebooks. This notebook accepts test splits only.
Current devtest is unlabelled: generate predictions to submit; no local score can be computed.
'''), code('''INPUT_RUN_NAME = RUN_NAME  # experiment03 / experiment05_qwen3vl8b
MODEL_STAGE = SFT_STAGE
SPLIT = 'devtest'
TEST_LIMIT = None  # full split; optionally 4 for an inference smoke test
if SPLIT not in {'devtest', 'test'}:
    raise ValueError('Use devtest/test here; validation is evaluated in the training notebooks')
if not re.fullmatch(r'[A-Za-z0-9_-]+', INPUT_RUN_NAME) or MODEL_STAGE not in {'sft', 'sft-smoke'}:
    raise ValueError('Choose a valid SFT run and stage')
from mmcultureqa.models.artifacts import require_merged_model
local_model = PROJECT_ROOT / 'outputs/inputs/models' / INPUT_RUN_NAME / MODEL_STAGE / 'best_model'
copy(cloud(f'task6-runs/{INPUT_RUN_NAME}/{MODEL_STAGE}/best_model'), local_model)
require_merged_model(local_model)
saved = json.loads((local_model / 'run_config.json').read_text(encoding='utf-8'))
predictions = PROJECT_ROOT / 'outputs/runs' / INPUT_RUN_NAME / 'predictions'
prediction_file = predictions / f'sft_{SPLIT}.jsonl'
overrides = {'model.name': str(local_model), 'model.revision': None, 'model.adapter': None,
             'model.adapter_revision': None, 'model.processor': str(local_model),
             'model.processor_revision': None, 'model.local_files_only': True, 'model.merged_model': True,
             'model.dtype': 'bfloat16', 'model.load_in_4bit': False,
             'inference.backend': 'vllm', 'inference.split': SPLIT,
             'inference.output': str(prediction_file), 'data.limit': TEST_LIMIT}
run_command('mmcultureqa.inference.predict', 'configs/inference.yaml', overrides)
if not prediction_file.is_file():
    raise FileNotFoundError(f'Inference produced no predictions: {prediction_file}')
upload_predictions(predictions, f'task6-runs/{INPUT_RUN_NAME}/predictions')
'''), code('''from mmcultureqa.common import read_jsonl, write_jsonl
test_rows = read_jsonl(PROJECT_ROOT / 'data/prepared' / f'{SPLIT}.jsonl')
if TEST_LIMIT is not None:
    test_rows = test_rows[:TEST_LIMIT]
if test_rows and all(row.get('answer') for row in test_rows):
    gold = predictions / f'{SPLIT}_gold_en.jsonl'
    write_jsonl(gold, test_rows)
    stream_run([sys.executable, '-m', 'mmcultureqa.score', '--dtype', 'float16',
                '--gold', str(gold), '--lang', 'en', '--pred', str(prediction_file),
                '--out', str(predictions / f'sft_{SPLIT}_scores.json')], check=True)
    upload_predictions(predictions, f'task6-runs/{INPUT_RUN_NAME}/predictions')
else:
    print(f'{SPLIT} has no public answers. Predictions saved for submission; no local score available.')
''')]


# Active notebooks: answer-only SFT, inference, and the paused legacy GRPO recipe.
def sft_model_notebook(run_name, config_name, title, modern=False):
    result = []
    for cell in sft_notebook:
        source = ''.join(cell['source']).replace('experiment03', run_name).replace('Experiment 03', title)
        source = source.replace("'configs/sft.yaml'", repr('configs/' + config_name))
        source = source.replace("MODEL_SIZE = '4B'", "MODEL_SIZE = '8B'")
        if cell['cell_type'] == 'markdown':
            source = source.replace('train batch 8 x accumulation 1',
                                    'train batch 4 x accumulation 2' if modern else 'train batch 2 x accumulation 4')
            source = source.replace('batch 8 and accumulation 1', 'effective batch 8')
            source = source.replace('Periodic validation: batch 8',
                                    'Periodic validation: batch 4' if modern else 'Periodic validation: batch 2')
            if source.startswith('## SFT only'):
                source = '## ' + title + '\n' + source.split('\n', 1)[1]
        result.append(code(source) if cell['cell_type'] == 'code' else md(source))
    return result


sft8b_notebook = sft_model_notebook('experiment05_qwen3vl8b', 'sft.yaml', 'Qwen3-VL-8B · SFT')

for name, cells in [('01_hf_to_drive_rclone', transfer), ('02_sft_colab', sft_notebook),
                    ('03_grpo_colab', grpo_notebook), ('04_inference_colab', inference_notebook),
                    ('05_sft_qwen3vl8b_colab', sft8b_notebook)]:
    titles = {'01_hf_to_drive_rclone': '# Task 6 · Hugging Face → Drive',
              '02_sft_colab': '# Task 6 · SFT · Qwen3-VL-4B', '03_grpo_colab': '# Task 6 · GRPO only',
              '04_inference_colab': '# Task 6 · Inference and scoring only',
              '05_sft_qwen3vl8b_colab': '# Task 6 · SFT · Qwen3-VL-8B'}
    cells = [md(''.join(cells[0]['source']).replace('# Task 6 · English QA', titles[name], 1))] + cells[1:]
    target = ROOT / 'notebooks' / f'{name}.ipynb'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3',
        'language': 'python', 'name': 'python3'}, 'colab': {'name': target.name}},
        'nbformat': 4, 'nbformat_minor': 5}, ensure_ascii=False, indent=1), encoding='utf-8')
