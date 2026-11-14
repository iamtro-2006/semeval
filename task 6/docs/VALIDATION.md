# Local CPU validation

Completed against the actual downloaded snapshot:

- All three English QA JSONL files match the equivalent Parquet rows exactly: train 10,000; dev 1,000; devtest 1,000.
- All 12,000 referenced images found in their respective archives; CRC read and Pillow image verification completed.
- Source IDs and image paths are disjoint across train/dev/devtest.
- Prepared train/validation contain 9,000/1,000 rows with disjoint IDs and image groups; dev/devtest remain untouched.
- All Python source files compile; all notebook Python cells parse, with IPython magic/shell lines excluded from AST parsing.
- All three inherited YAML configs load; prompt substitution works; LoRA regex matches the intended language-layer projection path and excludes a vision-layer path.
- Exact reward callable accepts conversational GRPO completions and the answer column correctly.
- Scorer source copied with its existing organizer provenance and license; no scorer modifications.

Not performed locally: GPU model loading, SFT forward/backward, GRPO rollout/update, mmBERT reward execution, Colab execution, Google Drive authentication or transfer. The notebooks contain executable one-step GPU smoke paths for both SFT and GRPO, followed by adapter reload/inference. Full-run accuracy and speed remain unmeasured.
