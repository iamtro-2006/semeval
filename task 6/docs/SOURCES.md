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

Training backend migrated to ms-swift **v3.11.3**, source commit `96d08aa09907e4e828ab4e13e157cf406994b817`:

- https://github.com/modelscope/ms-swift/blob/v3.11.3/swift/llm/train/sft.py
- https://github.com/modelscope/ms-swift/blob/v3.11.3/swift/llm/train/rlhf.py
- https://github.com/modelscope/ms-swift/blob/v3.11.3/swift/trainers/trainers.py
- https://github.com/modelscope/ms-swift/blob/v3.11.3/swift/plugin/metric.py
- https://github.com/modelscope/ms-swift/blob/v3.11.3/swift/plugin/orm.py
- https://github.com/modelscope/ms-swift/blob/v3.11.3/swift/trainers/rlhf_trainer/grpo_trainer.py
- https://github.com/modelscope/ms-swift/blob/v3.11.3/requirements/framework.txt

The pinned version supports Qwen3-VL and constrains transformers<4.58, trl<0.25, peft<0.19, datasets<4. Dependency pins follow those constraints (datasets changed to 3.6.0). Metric/reward plugins use v3.11.3's metric_mapping and orms, not the different 4.x plugin API. Native ms-swift owns templates, masking, optimizer, accumulation, rollout, loss and evaluation. SFT without generation is the default. No GPU execution has been completed locally.


vLLM default update:
- https://github.com/vllm-project/vllm/blob/v0.11.0/requirements/cuda.txt (torch 2.8.0, torchvision 0.23.0)
- https://docs.vllm.ai/en/v0.11.0/models/supported_models.html (Qwen3-VL LoRA support)
- https://github.com/modelscope/ms-swift/blob/v3.11.3/swift/trainers/rlhf_trainer/rollout_mixin.py (colocate, LoRA sync, sleep)
- https://github.com/modelscope/ms-swift/blob/v3.11.3/swift/llm/infer/infer_engine/vllm_engine.py (standalone inference API)
