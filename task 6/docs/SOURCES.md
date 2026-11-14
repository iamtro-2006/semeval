# Sources and implementation choices

- Reference repository: https://github.com/gitHuyNgo/semeval-2027 at `fc891cdc005d33c87eb63640e70b236fac713f0a` (read-only snapshot in sibling repo-review-semeval-2027). Reused the model/adapter revisions and unchanged vendored organizer scorer, not the old metadata-dependent dataset loader.
- Dataset card: https://huggingface.co/datasets/QCRI/MMCQA-SemEval27/blob/e850ede73e2f1ef3dde12e0210de67d107247ef0/README.md
- Qwen3-VL model: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
- Transformers model/processor: https://huggingface.co/docs/transformers/v4.57.3/en/model_doc/qwen3_vl
- PEFT continuation: https://huggingface.co/docs/peft/en/package_reference/peft_model
- GRPO 0.24 docs: https://huggingface.co/docs/trl/v0.24.0/en/grpo_trainer
- Exact trainer implementation inspected: https://github.com/huggingface/trl/blob/v0.24.0/trl/trainer/grpo_trainer.py
- SFT implementation inspected: https://github.com/huggingface/trl/blob/v0.24.0/trl/trainer/sft_trainer.py
- Trainer arguments: https://huggingface.co/docs/transformers/v4.57.3/en/main_classes/trainer
- Official scorer source: https://github.com/mmcultureqa-semeval27/mmcultureqa/tree/main/scorer (copied snapshot provenance in vendor/organizer/PROVENANCE.md)
- rclone Drive setup: https://rclone.org/drive/ ; copy: https://rclone.org/commands/rclone_copy/

SFT deliberately uses a small custom multimodal collator plus Transformers Trainer, so answer-only masking is explicit and verified against the actual native chat template. GRPO delegates policy loss, rollout grouping, sampling and gradient accumulation to TRL rather than reimplementing RL. All parameters are configuration-driven. Version pinning and source inspection are not a substitute for the provided GPU smoke experiment.
