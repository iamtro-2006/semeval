# MMCultureQA: Qwen3-VL 4B / 8B SFT

Active models: Qwen3-VL-4B-Instruct and Qwen3-VL-8B-Instruct only. Qwen3.5 notebook,
configuration and separate runtime requirements have been removed. Legacy artifact readers
and archived reasoning CLI tests remain for backwards compatibility.

## Notebooks

| Notebook | Purpose | Default run |
|---|---|---|
| 01_hf_to_drive_rclone | Prepare shared dataset | task6-data/raw |
| 02_sft_colab | Qwen3-VL-4B SFT | experiment03 |
| 03_grpo_colab | Paused legacy GRPO recipe, unchanged | experiment01_qlora |
| 04_inference_colab | Inference from a merged model on Drive | INPUT_RUN_NAME |
| 05_sft_qwen3vl8b_colab | Qwen3-VL-8B SFT | experiment05_qwen3vl8b |

Both SFT notebooks use configs/sft.yaml. MODEL_SIZE selects the model identifier and pinned
revision. Use a distinct RUN_NAME for each experiment; changing MODEL_SIZE does not rename a run.
All project paths resolve under /content/semeval/task 6 on Colab.

## Shared recipe

BF16 frozen base + language LoRA (not NF4): r=16, alpha=32, dropout=0.05, attention q/k/v/o
and MLP gate/up/down. Microbatch 8, accumulation 1, effective batch 8 on one GPU.
One epoch, peak LR 5e-5, cosine schedule, warmup 3%, weight decay 0.01, grad norm 1.0.
Save/eval every 170 optimizer steps. SFT_OVERRIDES can change these explicitly.
Full validation at step 0 and every evaluation; highest generated BERTScore-F1 selects the
best checkpoint. Loss/token accuracy are also logged. Step 0 is a baseline, not a saved checkpoint.

System: You are a helpful visual question answering assistant.

User: Answer the question using the image and relevant cultural knowledge. Give a direct,
accurate answer in English. Followed by Question: {question}.

## Sequential GPU use for periodic validation

1. Run native teacher-forced loss/token accuracy validation.
2. Save a temporary snapshot of the CURRENT adapter; no held-out answer is fed to the VLM.
3. Move training parameters, buffers, gradients and optimizer state to host RAM.
4. Start a separate vLLM process, load BF16 base + current language adapter, generate greedily.
5. Exit vLLM, then run the unchanged organizer scorer in a separate process.
6. Exit scorer, restore the original training tensors to their devices and continue training.

F1 is returned to the framework for checkpoint selection. Tensor identity is preserved and
devices are restored even on failure; child errors fail training rather than fabricate scores.
This is single-GPU BF16-only orchestration, NOT a persistent sleeping vLLM engine. It reloads
vLLM at every evaluation and repeats validation once for loss and once for generation.
No second Qwen remains on GPU while training, but CPU RAM must hold the training state plus
child-process memory. Sufficient host RAM is required. BF16 8B TRAINING can still exceed L4 VRAM;
evaluation offload does not reduce training activations. GPU execution has not been validated locally.

monitor.vllm controls memory utilization (0.9), max context (2048), concurrency (2),
CPU weight offload (4 GiB), and eager execution. CPU offload slows generation; use 0 if it fits.
Keep TRAIN_PROJECTOR=False: vLLM adapter evaluation supports language LoRA in this recipe.
Native predict_with_generate stays false: the custom evaluation integration invokes vLLM instead.

## Export, inference and storage

After training restores the best adapter, save final/ locally, exit training, reload BF16 base
and safe-merge to best_model/. Drive receives the complete best_model/ and reports, not a
standalone adapter, intermediate checkpoints or optimizer state. Intermediate adapters used
for evaluation are temporary. Resume uses a full LOCAL checkpoint-N, with no automatic cloud copy.
Use training.num_train_epochs=1 to stop at total epoch 1; changing it mid-run changes the LR horizon.

Notebook 04 downloads task6-runs/<INPUT_RUN_NAME>/<MODEL_STAGE>/best_model from the configured
Drive root. Model family, prompt and image preprocessing restore from the exported artifact.
Predictions and uploads use INPUT_RUN_NAME, avoiding collisions between 4B and 8B results.
Use devtest/test there; validation belongs to SFT. Unlabelled splits cannot be scored locally.
Inference vLLM settings live in configs/inference.yaml; CPU offload is configurable without file patching.
Never bypass prediction signature checks to resume another model into an existing output file.

Runtime: requirements-vllm.txt (Swift 3.11.3, Transformers 4.57.6, torch 2.8.0, vLLM 0.11.0).
The organizer scorer is unchanged. Dataset revision e850ede73e2f1ef3dde12e0210de67d107247ef0;
image-disjoint seed-42 split, 9000 train / 1000 validation. No automatic CoT evaluation.

## Verification

python "task 6/tools/build_notebooks.py"
python -m unittest discover -s "task 6/tests" -v

CPU fixtures/mocks verify argument mapping, notebook execution, subprocess ordering, F1 return,
error restoration, artifact handling and upload filters. They do not prove GPU fit, throughput
or score improvements. vLLM sleep-mode reference: https://docs.vllm.ai/en/v0.11.0/features/sleep_mode.html
