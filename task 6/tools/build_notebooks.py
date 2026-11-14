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

transfer = setup + [md('''## 1. Download all three English QA splits on Colab
Only `qa/mena/*_en.{jsonl,parquet}` and three `images_mena_*.zip` are selected.
No audio, other languages, or embedded-media demo files.
'''), code('''%pip -q install huggingface-hub==0.36.0 Pillow==11.3.0
!python -m mmcultureqa.data.prepare download
!python -m mmcultureqa.data.prepare prepare
'''), md('''## 2. Editable rclone settings
Set everything in the next cell. `RCLONE_CONFIG_PATH` points to an existing, authenticated
rclone.conf (upload it via Colab's Files panel, or mount Drive in the optional cell).
`REMOTE` is the section name in that file, without a trailing colon.

Set `ROOT_FOLDER_ID` to a Drive folder ID or its full folder URL. When it is set,
`DESTINATION_SUBDIR` is relative to that folder. Empty means My Drive root, overriding
any root_folder_id already in the supplied config. Example: select the MMCQA folder as
root and use `task6-data/raw` as destination. Without a root, use `MMCQA/task6-data/raw`.
The supplied config is copied to a private temporary file; the original is never modified.
An authenticated config is still required: a folder ID alone does not grant access.
'''), code('''# Optional: run if rclone.conf is stored on Drive rather than uploaded to /content.
# from google.colab import drive
# drive.mount('/content/drive')
'''), code('''# Change these settings directly in the notebook.
RCLONE_CONFIG_PATH = '/content/rclone.conf'  # or /content/drive/MyDrive/.../rclone.conf
REMOTE = 'gdrive'
ROOT_FOLDER_ID = ''  # folder ID or https://drive.google.com/drive/folders/...
DESTINATION_SUBDIR = ''  # empty: auto-select task6-data/raw if root set, else MMCQA/task6-data/raw
TRANSFERS = 4
CHECKERS = 8
'''), code('''!sudo apt-get -qq update && sudo apt-get -qq install -y rclone
import configparser, re, tempfile
from urllib.parse import urlparse

original_config = Path(RCLONE_CONFIG_PATH).expanduser()
if not original_config.is_file():
    raise FileNotFoundError(f'Upload rclone.conf or mount Drive, then set RCLONE_CONFIG_PATH: {original_config}')
sections = configparser.RawConfigParser()
sections.read(original_config, encoding='utf-8-sig')
if REMOTE not in sections or sections[REMOTE].get('type') != 'drive':
    raise ValueError('REMOTE must name a Google Drive section in the supplied rclone.conf')
root_id = ROOT_FOLDER_ID.strip()
if root_id.startswith(('https://', 'http://')):
    match = re.search(r'/folders/([^/?]+)', urlparse(root_id).path)
    if not match:
        raise ValueError('Use a Google Drive folder URL containing /folders/<id>')
    root_id = match.group(1)
if root_id and not re.fullmatch(r'[A-Za-z0-9_-]+', root_id):
    raise ValueError('Invalid Drive folder ID')
subdir = DESTINATION_SUBDIR.strip() or ('task6-data/raw' if root_id else 'MMCQA/task6-data/raw')
if subdir.startswith('/') or ':' in subdir or '..' in Path(subdir).parts:
    raise ValueError('DESTINATION_SUBDIR must be a relative folder path')
if TRANSFERS < 1 or CHECKERS < 1:
    raise ValueError('TRANSFERS and CHECKERS must be positive')

# rclone may refresh tokens; keep writes in a temporary copy, not the original.
temporary_config = tempfile.TemporaryDirectory(prefix='mmcqa-rclone-')
config_path = Path(temporary_config.name) / 'rclone.conf'
config_path.write_bytes(original_config.read_bytes())
config_path.chmod(0o600)
rclone_base = ['rclone', '--config', str(config_path), '--drive-root-folder-id', root_id]
DESTINATION = f'{REMOTE}:{subdir}'
print('Config source:', original_config)
print('Drive root:', root_id or 'My Drive')
print('Destination relative to root:', DESTINATION)
# Read-only check before the upload cell. Credentials are never printed.
subprocess.run(rclone_base + ['lsd', f'{REMOTE}:', '--max-depth', '1'], check=True)
'''), md('''## 3. Upload archives and labels from Colab → Drive
`copy` is incremental and does not delete destination files. Keep archives compressed on Drive;
the experiment notebook copies them to Colab's SSD and extracts locally.
'''), code('''subprocess.run(rclone_base + ['copy', 'data/raw', DESTINATION,
                '--progress', '--transfers', str(TRANSFERS), '--checkers', str(CHECKERS), '--exclude', '.cache/**',
                '--exclude', 'train/**', '--exclude', 'dev/**', '--exclude', 'devtest/**'], check=True)
subprocess.run(rclone_base + ['check', 'data/raw', DESTINATION,
                '--one-way', '--size-only', '--exclude', '.cache/**', '--exclude', 'train/**',
                '--exclude', 'dev/**', '--exclude', 'devtest/**'], check=True)
'''), code('''# Remove temporary credentials after the transfer. Reconfigure next session if needed.
temporary_config.cleanup()
''')]

experiment = setup + [md('''## 1. Install the pinned GPU training environment
After this cell, restart the runtime if Colab requests it, then rerun the clone/setup cell.
Do not install the CPU-only requirements in vendor/organizer into this GPU environment.
'''), code('''%pip -q install -r requirements.txt
!python -m pip check
!nvidia-smi
'''), code('''from google.colab import drive
drive.mount('/content/drive')
'''), md('''## 2. Copy data from Drive or download directly from Hugging Face
All extraction and image reads happen on local SSD, not mounted Drive.
'''), code('''import shutil
USE_DRIVE_DATA = True
drive_raw = Path('/content/drive/MyDrive/MMCQA/task6-data/raw')
if USE_DRIVE_DATA:
    if not drive_raw.exists():
        raise FileNotFoundError('Run the transfer notebook first, or set USE_DRIVE_DATA=False')
    shutil.copytree(drive_raw, Path('data/raw'), dirs_exist_ok=True)
else:
    subprocess.run([sys.executable, '-m', 'mmcultureqa.data.prepare', 'download'], check=True)
subprocess.run([sys.executable, '-m', 'mmcultureqa.data.prepare', 'prepare'], check=True)
'''), md('''## 3. Editable experiment settings
Edit prompts/qa.txt and YAML files, or use dotted --set overrides below.
4-bit is explicit for a memory-limited Colab GPU. Set False for native precision.
GPU smoke runs are needed before full training; they have not been executed on the CPU-only development machine.
'''), code('''USE_4BIT = True
RUN_FULL = False
RUN_NAME = 'experiment01'  # use a new name for a fresh experiment
RUNS = Path('/content/drive/MyDrive/MMCQA/task6-runs') / RUN_NAME
RUNS.mkdir(parents=True, exist_ok=True)
SFT = str(RUNS / ('sft' if RUN_FULL else 'sft-smoke'))
GRPO = str(RUNS / ('grpo' if RUN_FULL else 'grpo-smoke'))

def run(module, config, overrides):
    cmd = [sys.executable, '-m', module, '--config', config]
    for key, value in overrides.items():
        cmd += ['--set', f'{key}={value}']
    subprocess.run(cmd, check=True)

common = {'model.load_in_4bit': str(USE_4BIT).lower()}
'''), md('''## 4. SFT (answer-only loss)
Smoke: 1 optimizer step. Full: configured 2 epochs, internal validation used for checkpoint selection.
Saved `final` contains LoRA adapter and processor; the base weights are not duplicated.
'''), code('''sft_args = {**common, 'training.output_dir': SFT}
if not RUN_FULL:
    sft_args.update({'data.limit': 8, 'training.max_steps': 1, 'training.eval_strategy': 'no',
                     'training.load_best_model_at_end': 'false', 'training.save_steps': 1})
run('mmcultureqa.training.sft', 'configs/sft.yaml', sft_args)
'''), md('''## Optional: continue GRPO directly from the published SFT adapter
Skip the SFT cell if using this. Its published split differs from our generated split.
For internal validation claims, import its original held-out manifest; do not assume our split was unseen.
The official dev split is separate according to the reference repo's audit.
'''), code('''USE_PUBLISHED_ADAPTER = False
SFT_ADAPTER = 'anhbilong/qwen3-vl-4b-mmcultureqa-split512' if USE_PUBLISHED_ADAPTER else f'{SFT}/final'
ADAPTER_REVISION = 'e62fdf28389cdc11f45f295f63e3e4c354e8eabe' if USE_PUBLISHED_ADAPTER else 'null'
'''), md('''## 5. GRPO continuation with trainable SFT LoRA
Generates fresh groups during training; no precomputed inference file is required.
mmBERT reward runs on CPU by default to preserve GPU VRAM; use reward.device=cuda if memory permits.
beta=0 explicitly avoids treating the pre-SFT base as the KL reference.
'''), code('''grpo_args = {**common, 'model.adapter': SFT_ADAPTER, 'model.adapter_revision': ADAPTER_REVISION,
             'training.output_dir': GRPO}
if not RUN_FULL:
    grpo_args.update({'data.limit': 8, 'training.max_steps': 1, 'training.save_steps': 1})
run('mmcultureqa.training.grpo', 'configs/grpo.yaml', grpo_args)
'''), md('''## 6. Compare SFT and GRPO with identical inference settings
Smoke predictions cover only 4 rows; do not score them against full dev.
For actual comparison, set RUN_FULL=True and complete both full runs first.
Use split=devtest for answer-free submission generation.
'''), code('''SPLIT = 'dev'
for label, adapter in [('sft', SFT_ADAPTER), ('grpo', f'{GRPO}/final')]:
    overrides = {**common, 'model.adapter': adapter,
                 'model.adapter_revision': ADAPTER_REVISION if label == 'sft' else 'null',
                 'inference.split': SPLIT, 'inference.output': str(RUNS / f'{label}_{SPLIT}.jsonl')}
    if not RUN_FULL:
        overrides['data.limit'] = 4
    run('mmcultureqa.inference.predict', 'configs/inference.yaml', overrides)
'''), md('''## 7. Official scoring (full labelled dev only)
The vendored backbone and wrapper are unchanged from the reference repository.
For leaderboard dependency parity use a separate environment with vendor/organizer/requirements.txt;
the GPU environment below uses the same pinned encoder and metric settings, but a different torch version.
'''), code('''if RUN_FULL and SPLIT == 'dev':
    for label in ['sft', 'grpo']:
        subprocess.run([sys.executable, 'vendor/organizer/scorer/score.py',
                        '--gold', 'data/raw/qa/mena/dev_en.jsonl',
                        '--pred', str(RUNS / f'{label}_dev.jsonl'),
                        '--out', str(RUNS / f'{label}_scores.json')], check=True)
else:
    print('Scoring skipped: smoke subset or unlabelled split.')
''')]

for name, cells in [('01_hf_to_drive_rclone', transfer), ('02_sft_grpo_inference_colab', experiment)]:
    target = ROOT / 'notebooks' / f'{name}.ipynb'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3',
        'language': 'python', 'name': 'python3'}, 'colab': {'name': target.name}},
        'nbformat': 4, 'nbformat_minor': 5}, ensure_ascii=False, indent=1), encoding='utf-8')
