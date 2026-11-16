# Local CPU validation

## Experiment03 SFT + merged BF16 artifacts

- SFT defaults: 2 epochs, peak LR 1e-4, microbatch 8 / accumulation 1, eval/save every 225 steps, teacher-forced loss/token accuracy on the fixed seed-42 monitor subset. No periodic generation or mmBERT; best selected by lowest val loss.
- Projector LoRA is optional (off by default), explicitly selecting Qwen3-VL merger/DeepStack linear_fc1/linear_fc2 in addition to all existing language attention/MLP targets. ViT stays frozen; training checks requested projector parameters are trainable.
- The training subprocess saves the selected adapter locally. A separate exporter reloads the original BF16 base, applies the selected adapter, safely merges, saves processor/tokenizer + frozen prompt/run metadata, validates all weight shards, then atomically publishes best_model. SFT uploads only that complete model and reports.
- Inference downloads experiment03's merged model from Drive; local-only BF16 loading has no adapter or original-Qwen Hub fallback. Final validation uses this exported BF16 artifact once on the full manifest. Legacy adapter evaluation remains supported.
- CPU tests use mocks for model libraries to check BF16 reload flags, selected adapter provenance, safe merge call, idempotency/no-overwrite, complete shard validation, local-only loading, upload filters and notebook paths. They do not execute actual PEFT merges or GPU forward/backward.
- The GRPO notebook and its YAML are unchanged. Existing GRPO hyperparameters/backend are preserved; the new SFT/inference workflow does not require a GRPO output.

## Earlier recipe checks

## Unified SFT generation defaults

- SFT now uses NF4 QLoRA + BF16 compute, train microbatch 8 / accumulation 1 (effective batch 8 unchanged), eval batch 8, generation evaluation enabled and best checkpoint selected by BERTScore-F1.
- SFT and GRPO share the fixed random seed-42 monitoring subset, default 256 rows; the complete training split and final 1,000-row validation remain independent of that limit.
- SFT, GRPO and inference notebook defaults use 4-bit / Transformers with optional vLLM installation disabled. CPU regressions check inherited SFT arguments, generation/teacher-forced metric switching, effective batch preservation and notebook/backend defaults. GPU batch-8 fitting and speed are unmeasured.

## GRPO QLoRA recipe update

- GRPO defaults to bitsandbytes NF4 + double quantization with BF16 compute, PEFT LoRA, `use_vllm=false`, no model/optimizer offload, and no vLLM sleep.
- ms-swift 3.11.3's non-vLLM branch was checked: `PtEngine.from_model_template(self.model, ...)` reuses the training model. No second Qwen engine is constructed on this path; mmBERT remains a separate reward encoder.
- Single-GPU recipe: train microbatch 4, accumulation 4, steps_per_generation 4, 16 rollout completions / 4 questions, eval batch 8, num_generations 4. Eval grouping and generation grouping are validated before GPU model allocation.
- CPU regressions check actual inherited YAML -> native arguments, NF4/BF16 flags, grouping after overrides, separate SFT input/QLoRA output runs, optional vLLM install/import checks, and Transformers selection for final validation.
- Disabled-vLLM native arguments explicitly set `vllm_mode=None`, including after stale mode overrides. The actual `_init_rollout` method extracted from pinned ms-swift 3.11.3 source was executed with the Transformers flags to verify its mode guard accepts this combination; full GPU initialization remains untested.
- `msgspec==0.19.0` is declared directly: pinned ms-swift GRPO utilities import it even without vLLM. Notebook preflight and the GRPO launcher import the concrete GRPO trainer before loading model/scorer weights so lazy dependency failures surface early. Actual trainer imports still need verification in the Colab GPU environment.
- GPU quantized loading, generation/backward, batch-4 VRAM fitting, numerical quality and speed vs the previous BF16/vLLM recipe have not been benchmarked. A CPU test pass does not establish a sub-eight-hour runtime.

## Earlier dataset and backend checks

Completed against the actual downloaded snapshot:

- All three English QA JSONL files match the equivalent Parquet rows exactly: train 10,000; dev 1,000; devtest 1,000.
- All 12,000 referenced images found in their respective archives; CRC read and Pillow image verification completed.
- Source IDs and image paths are disjoint across train/dev/devtest.
- Prepared train/validation contain 9,000/1,000 rows with disjoint IDs and image groups; dev/devtest remain untouched.
- All Python source files compile; all notebook Python cells parse, with IPython magic/shell lines excluded from AST parsing.
- All three inherited YAML configs load; prompt substitution works; LoRA regex matches the intended language-layer projection path and excludes a vision-layer path.
- ms-swift v3.11.3 source API inspected. Exported native SFT and GRPO arguments checked against source dataclass field names and inherited Transformers arguments for generation enabled/disabled.
- Native schema export checked: SFT messages/images only, GRPO messages/images/solution/id, no reference answer inserted into GRPO prompt, validation count independent of train smoke limit.
- Native metric_mapping and ORM registration tested with stub framework/encoder, including precision/recall/F1 return keys, exact reward and empty-answer zero handling. Actual mmBERT inference remains unexecuted.
- Scorer source copied with its existing organizer provenance and license; no scorer modifications.
- BF16/CUDA mmBERT/vLLM defaults checked in YAML and generated notebooks. Native rollout option names and standalone VllmEngine keyword arguments checked against ms-swift v3.11.3 source. vLLM 0.11.0's torch 2.8.0/torchvision 0.23.0 dependency pins inspected.
- mmBERT FP16 casting and separate FP16/FP32 scorer caches verified with a stub model. Final scoring uses a precision-selectable wrapper; original organizer files remain unchanged. GPU precision differences and speed have not been measured.

Not performed locally: GPU model loading, SFT forward/backward, GRPO rollout/update, mmBERT reward execution, vLLM initialization/LoRA synchronization, 24 GB memory fitting, Colab execution, Google Drive authentication or transfer. The notebooks contain executable one-step GPU smoke paths for both SFT and GRPO, followed by adapter reload/inference. Full-run accuracy and speed remain unmeasured.
