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
checkout = Path(CHECKOUT)
if not checkout.exists():
    subprocess.run(['git', 'clone', REPO_URL, str(checkout)], check=True)
else:
    actual = subprocess.check_output(['git', '-C', str(checkout), 'remote', 'get-url', 'origin'], text=True).strip()
    if actual != REPO_URL:
        raise ValueError('CHECKOUT belongs to a different repository; choose another CHECKOUT')
    dirty = subprocess.check_output(['git', '-C', str(checkout), 'status', '--porcelain'], text=True)
    if dirty.strip():
        raise RuntimeError('Checkout has edits; preserve your changes or choose a fresh CHECKOUT before switching revision')
subprocess.run(['git', '-C', str(checkout), 'fetch', 'origin', GIT_REF], check=True)
subprocess.run(['git', '-C', str(checkout), 'checkout', '--detach', 'FETCH_HEAD'], check=True)
os.chdir(checkout / PROJECT_SUBDIR)
subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--no-build-isolation', '-e', '.'], check=True)
print('Project:', Path.cwd())
print('Commit:', subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
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
'''), code('''ROOT_FOLDER_ID = ''  # REQUIRED: ID or URL of the MMCQA folder
RCLONE_CONFIG_PATH = '/content/rclone.conf'
REMOTE = ''  # auto-detect when the config contains one Google Drive remote
'''), code('''!sudo apt-get -qq update && sudo apt-get -qq install -y rclone
import configparser, re, tempfile
from urllib.parse import urlparse
config_source = Path(RCLONE_CONFIG_PATH).expanduser()
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
    subprocess.run(rclone_base + ['copy', str(source), str(destination),
                   '--transfers', '4', '--checkers', '8', '--progress', *extra], check=True)
subprocess.run(rclone_base + ['lsd', cloud(''), '--max-depth', '1'], check=True)
print('All storage is relative to MMCQA folder ID:', root_id)
''')]

data_setup = [md('''## Prepare data on Colab SSD
The download location is automatically derived from ROOT_FOLDER_ID.
'''), code('''USE_DRIVE_DATA = True
if USE_DRIVE_DATA:
    copy(cloud('task6-data/raw'), 'data/raw')
else:
    subprocess.run([sys.executable, '-m', 'mmcultureqa.data.prepare', 'download'], check=True)
subprocess.run([sys.executable, '-m', 'mmcultureqa.data.prepare', 'prepare'], check=True)
''')]

gpu_setup = setup + [md('''## Install GPU environment
Restart the runtime if requested, then rerun clone/setup. GPU training has not been run on the CPU-only development machine.
'''), code('''%pip -q install -r requirements.txt
!python -m pip check
!nvidia-smi
''')] + storage + data_setup + [md('''## Experiment settings
Only ROOT_FOLDER_ID determines cloud storage. RUN_NAME groups your experiments under it.
Periodic checkpoints upload after each save; final adapter uploads after training finishes.
Local SSD is used for training and inference. A disconnected runtime can only recover checkpoints already uploaded.
Default: native BF16 LoRA, CUDA FP16 mmBERT; GRPO uses vLLM colocate with sleep enabled.
SFT's optional in-training generation monitor uses the native ms-swift Transformers path.
Final validation/test inference uses vLLM. This is an unbenchmarked 24 GB starting configuration.
'''), code('''import json
USE_4BIT = False
PREDICT_WITH_GENERATE = False  # default SFT: val loss + token accuracy; True: generation + mmBERT
RUN_FULL = False
RUN_NAME = 'experiment01'
if not re.fullmatch(r'[A-Za-z0-9_-]+', RUN_NAME):
    raise ValueError('RUN_NAME must contain only letters, numbers, underscores or hyphens')
RUNS = Path('outputs/runs') / RUN_NAME
RUNS.mkdir(parents=True, exist_ok=True)
SFT_STAGE = 'sft' if RUN_FULL else 'sft-smoke'
GRPO_STAGE = 'grpo' if RUN_FULL else 'grpo-smoke'
SFT = str(RUNS / SFT_STAGE)
GRPO = str(RUNS / GRPO_STAGE)
common = {'model.dtype': 'bfloat16', 'model.load_in_4bit': USE_4BIT,
          'training.predict_with_generate': PREDICT_WITH_GENERATE}

def run(module, config, overrides, local_output, cloud_subdir):
    cmd = [sys.executable, '-m', module, '--config', config]
    for key, value in overrides.items():
        cmd += ['--set', f'{key}={json.dumps(value)}']
    env = dict(os.environ)
    env.update(MMCQA_RCLONE_CONFIG=str(config_path), MMCQA_DRIVE_ROOT_ID=root_id,
               MMCQA_SYNC_DESTINATION=cloud(cloud_subdir))
    result = subprocess.run(cmd, env=env)
    # Also preserve final adapter and logs, including after a failed training subprocess.
    if Path(local_output).exists():
        copy(local_output, cloud(cloud_subdir))
    result.check_returncode()
''')]

transfer = setup + storage + [md('''## Download English QA and upload automatically below MMCQA
Includes train/dev/devtest JSONL, Parquet and image ZIPs; excludes audio and other languages.
Only compressed archives and labels are uploaded. All later notebooks use the same relative data path.
'''), code('''%pip -q install huggingface-hub==0.36.0 Pillow==11.3.0
!python -m mmcultureqa.data.prepare download
!python -m mmcultureqa.data.prepare prepare
'''), code('''filters = ['--exclude', '.cache/**', '--exclude', 'train/**', '--exclude', 'dev/**', '--exclude', 'devtest/**']
copy('data/raw', cloud('task6-data/raw'), *filters)
subprocess.run(rclone_base + ['check', 'data/raw', cloud('task6-data/raw'),
                '--one-way', '--size-only', *filters], check=True)
print('Dataset ready under MMCQA/task6-data/raw')
''')]

sft_notebook = gpu_setup + [md('''## SFT only · ms-swift + PEFT
Default PREDICT_WITH_GENERATE=False: monitor teacher-forced loss and token accuracy.
True: native ms-swift generation evaluation with mmBERT P/R/F1 plugin; select best by F1.
Generation evaluation replaces the teacher-forced validation pass; it does not report val loss simultaneously.
Edit SFT_OVERRIDES for prompt and training parameters. For same-stage resume, set
RESUME_CHECKPOINT to checkpoint-N already saved under this run on Drive.
'''), code('''SFT_OVERRIDES = {}  # e.g. {'training.learning_rate': 0.0001, 'prompt_file': 'prompts/qa.txt'}
RESUME_CHECKPOINT = ''  # e.g. checkpoint-100; leave empty for a fresh run
sft_args = {**common, 'training.output_dir': SFT}
if not RUN_FULL:
    sft_args.update({'data.limit': 8, 'training.max_steps': 1, 'training.eval_strategy': 'no',
                     'training.load_best_model_at_end': False, 'training.save_steps': 1})
if RESUME_CHECKPOINT:
    copy(cloud(f'task6-runs/{RUN_NAME}/{SFT_STAGE}'), SFT)
    sft_args['resume_from_checkpoint'] = str(Path(SFT) / RESUME_CHECKPOINT)
sft_args.update(SFT_OVERRIDES)
run('mmcultureqa.training.sft', 'configs/sft.yaml', sft_args, SFT, f'task6-runs/{RUN_NAME}/{SFT_STAGE}')
'''), code('''if not (Path(SFT) / 'final/adapter_config.json').is_file():
    raise FileNotFoundError('No final SFT adapter was saved')
print('SFT adapter on Drive:', cloud(f'task6-runs/{RUN_NAME}/{SFT_STAGE}/final'))
print('Notebook 03: select the same ROOT_FOLDER_ID, SFT_RUN_NAME and INPUT_SFT_STAGE.')
''')]

grpo_notebook = gpu_setup + [md('''## GRPO only · ms-swift + PEFT
Training always generates rollouts. PREDICT_WITH_GENERATE=False disables extra validation rollouts.
True enables native ms-swift validation rollout evaluation at eval_steps: inspect
eval_rewards/MMCQABERTScore/mean for raw mmBERT F1 (sampled responses, not teacher-forced accuracy).
GRPO does not have the supervised token-loss/accuracy semantics of SFT.
Paths are automatically derived from MMCQA root. This notebook never trains SFT.
A full GRPO experiment should use a full SFT checkpoint, not a smoke adapter.
'''), code('''SFT_RUN_NAME = RUN_NAME
INPUT_SFT_STAGE = 'sft' if RUN_FULL else 'sft-smoke'
SFT_CHECKPOINT = 'final'  # or checkpoint-N
GRPO_OVERRIDES = {}  # e.g. {'reward.device': 'cuda', 'training.learning_rate': 0.000005}
RESUME_CHECKPOINT = ''  # GRPO checkpoint-N in this output run, for same-stage resume
local_adapter = Path('outputs/inputs/sft') / SFT_RUN_NAME / INPUT_SFT_STAGE / SFT_CHECKPOINT
copy(cloud(f'task6-runs/{SFT_RUN_NAME}/{INPUT_SFT_STAGE}/{SFT_CHECKPOINT}'), local_adapter)
if not (local_adapter / 'adapter_config.json').is_file():
    raise FileNotFoundError('SFT adapter not found. Check the selected run/stage/checkpoint.')
if not any((local_adapter / name).is_file() for name in ['adapter_model.safetensors', 'adapter_model.bin']):
    raise FileNotFoundError('SFT weights missing; complete SFT save/upload first')
grpo_args = {**common, 'model.adapter': str(local_adapter), 'model.adapter_revision': None,
             'training.output_dir': GRPO}
if not RUN_FULL:
    grpo_args.update({'data.limit': 8, 'training.max_steps': 1, 'training.save_steps': 1})
if RESUME_CHECKPOINT:
    copy(cloud(f'task6-runs/{RUN_NAME}/{GRPO_STAGE}'), GRPO)
    grpo_args['resume_from_checkpoint'] = str(Path(GRPO) / RESUME_CHECKPOINT)
grpo_args.update(GRPO_OVERRIDES)
run('mmcultureqa.training.grpo', 'configs/grpo.yaml', grpo_args, GRPO, f'task6-runs/{RUN_NAME}/{GRPO_STAGE}')
'''), code('''if not (Path(GRPO) / 'final/adapter_config.json').is_file():
    raise FileNotFoundError('No final GRPO adapter was saved')
print('GRPO adapter on Drive:', cloud(f'task6-runs/{RUN_NAME}/{GRPO_STAGE}/final'))
''')]

def final_validation(run_variable, stage_variable):
    return [md('''## Final held-out validation score
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
result = subprocess.run(cmd)
copy(validation_run, cloud(f'task6-runs/{RUN_NAME}/{__STAGE__}'))
result.check_returncode()
'''.replace('__RUN_DIR__', run_variable).replace('__STAGE__', stage_variable))]

sft_notebook += final_validation('SFT', 'SFT_STAGE')
grpo_notebook += final_validation('GRPO', 'GRPO_STAGE')

inference_notebook = gpu_setup + [md('''## Test inference only
Select run/stage names below. No manual Drive filesystem paths are needed.
Validation scoring belongs to the SFT/GRPO notebooks. This notebook accepts test splits only.
Current devtest is unlabelled: generate predictions to submit; no local score can be computed.
'''), code('''INPUT_RUN_NAME = RUN_NAME
SPLIT = 'devtest'
TEST_LIMIT = None  # full split; optionally 4 for an inference smoke test
if SPLIT not in {'devtest', 'test'}:
    raise ValueError('Use devtest/test here; validation is evaluated in the training notebooks')
ADAPTER_STAGES = {'sft': SFT_STAGE, 'grpo': GRPO_STAGE}
predictions = RUNS / 'predictions'
for label, stage in ADAPTER_STAGES.items():
    adapter = Path('outputs/inputs/inference') / INPUT_RUN_NAME / stage / 'final'
    copy(cloud(f'task6-runs/{INPUT_RUN_NAME}/{stage}/final'), adapter)
    overrides = {**common, 'model.adapter': str(adapter), 'model.adapter_revision': None,
                 'inference.backend': 'transformers' if USE_4BIT else 'vllm',
                 'inference.split': SPLIT, 'inference.output': str(predictions / f'{label}_{SPLIT}.jsonl')}
    overrides['data.limit'] = TEST_LIMIT
    run('mmcultureqa.inference.predict', 'configs/inference.yaml', overrides,
        predictions, f'task6-runs/{RUN_NAME}/predictions')
'''), code('''from mmcultureqa.common import read_jsonl, write_jsonl
test_rows = read_jsonl(Path('data/prepared') / f'{SPLIT}.jsonl')
if TEST_LIMIT is not None:
    test_rows = test_rows[:TEST_LIMIT]
if test_rows and all(row.get('answer') for row in test_rows):
    gold = predictions / f'{SPLIT}_gold_en.jsonl'
    write_jsonl(gold, test_rows)
    for label in ADAPTER_STAGES:
        subprocess.run([sys.executable, '-m', 'mmcultureqa.score', '--dtype', 'float16',
                        '--gold', str(gold), '--lang', 'en',
                        '--pred', str(predictions / f'{label}_{SPLIT}.jsonl'),
                        '--out', str(predictions / f'{label}_{SPLIT}_scores.json')], check=True)
    copy(predictions, cloud(f'task6-runs/{RUN_NAME}/predictions'))
else:
    print(f'{SPLIT} has no public answers. Predictions saved for submission; no local score available.')
''')]

for name, cells in [('01_hf_to_drive_rclone', transfer), ('02_sft_colab', sft_notebook),
                    ('03_grpo_colab', grpo_notebook), ('04_inference_colab', inference_notebook)]:
    titles = {'01_hf_to_drive_rclone': '# Task 6 · Hugging Face → Drive',
              '02_sft_colab': '# Task 6 · SFT only', '03_grpo_colab': '# Task 6 · GRPO only',
              '04_inference_colab': '# Task 6 · Inference and scoring only'}
    cells = [md(''.join(cells[0]['source']).replace('# Task 6 · English QA', titles[name], 1))] + cells[1:]
    target = ROOT / 'notebooks' / f'{name}.ipynb'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3',
        'language': 'python', 'name': 'python3'}, 'colab': {'name': target.name}},
        'nbformat': 4, 'nbformat_minor': 5}, ensure_ascii=False, indent=1), encoding='utf-8')
