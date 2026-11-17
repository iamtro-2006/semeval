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
1.2 epochs, peak LR 2e-4, cosine schedule, warmup 3%, weight decay 0.01, grad norm 1.0; label_smoothing_factor=0.005.
Save every 225 optimizer steps; evaluate loss every 125 steps. SFT_OVERRIDES can change these explicitly.
During training, full validation computes loss/token accuracy at step 0 and every 125 steps.
There is NO vLLM, generation F1, scorer, sleep/wake or CPU model swapping during SFT.
All checkpoints are retained (save_total_limit=null); the trainer does not restore a provisional best.

System: You are a visual question answering assistant with expertise in cultures worldwide.

User prompt (prompts/qa.txt):

```text
Question: {question}

Use the image together with your cultural and historical knowledge to answer the question. Focus on relevant objects, distinctive features, their arrangement and relationships, and readable text (excluding watermarks). Consider how these details help identify the subject and understand its context.

When the subject or its meaning is ambiguous, compare plausible interpretations and determine which is best supported by the image and the question. Give specific names and explanations where possible, and acknowledge uncertainty when the available details are insufficient.

Provide a complete, accurate, culturally grounded answer in English. Address every part of the question, including any requested identification, description, purpose, or cultural significance. Support your answer with relevant details and keep the explanation focused on what the question asks.
```

The native framework label smoother is retained with next-token shifting explicitly enabled for
Qwen3-VL SFT; padding/prompt labels masked with -100 retain the framework masking behavior.
Val loss with label smoothing is not directly comparable with prior unsmoothed val loss.

## Post-training checkpoint selection

The training cell finishes and its process exits, releasing the training model and optimizer.
The next cell calls `python -m mmcultureqa.select_checkpoint --run-dir <SFT>`:

1. Enumerate every local checkpoint-N in numeric order (normally 6 checkpoints / 1350 steps).
2. Reload the original BF16 base and safe-merge that candidate adapter into a temporary model.
3. Exit the merge process. Generate ALL 1000 validation answers using the complete merged model.
4. Exit generation. Load mmBERT separately and score with the unchanged organizer scorer.
5. Save scores/predictions for that checkpoint, then delete only its temporary merged weights.
6. Select the highest F1 (earliest step breaks ties), export the winner to best_model/ and copy
   its validation report to validation/. No redundant final generation is run.

Generation uses the SAME configs/inference.yaml defaults, predict entry point and vLLM loader
as notebook 04: BF16, batch 8, max_num_seqs 8, greedy, max_new_tokens 256, context 2048,
CUDA graphs enabled, CPU weight offload 0. Runtime LoRA and sleep are disabled for merged models.
This removes the slow hot-LoRA code path; exact questions/sec still depends on images and answer length.
No base+LoRA engine is used to select a checkpoint. Legacy helper modules are inactive.

Only one temporary merged candidate exists at once; after ranking it is deleted before the next
candidate. The final winner is merged once again, avoiding keeping 9 full models on disk.
All resumable adapter checkpoints/optimizer states remain LOCAL and take additional disk space.
Per-checkpoint reports and ranking.json are retained. Completed scores can be reused on rerun when
config, inference settings, adapter weights and dataset fingerprints match. A scoring failure stops
selection; it never silently skips a candidate. Training does not need to be repeated to rerun selection.

BF16 8B training and merged inference must fit the available GPU. There is no GPU validation in
this development environment; CPU tests verify orchestration, selection and artifacts only.

## Export, inference and storage

The selected merged BF16 model and winning validation report are uploaded only AFTER selection.
Drive receives best_model/, selection.json, checkpoint_selection/ranking.json and reports; no
standalone adapters, intermediate checkpoints or optimizer states are uploaded. Local final/ is the
last training adapter/processor; selection.json points to the actual winning checkpoint, and the
exporter is explicitly given that checkpoint. Notebook 04 uses only best_model/.
Resume training uses a full LOCAL checkpoint-N. Use a fresh RUN_NAME for a new experiment.

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

CPU fixtures/mocks verify argument mapping, notebook execution, merge/score ordering, F1 selection,
score reuse, artifact handling and upload filters. They do not prove GPU fit, throughput
or score improvements. vLLM sleep-mode reference: https://docs.vllm.ai/en/v0.11.0/features/sleep_mode.html
