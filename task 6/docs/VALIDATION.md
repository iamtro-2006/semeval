# Local CPU validation

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
