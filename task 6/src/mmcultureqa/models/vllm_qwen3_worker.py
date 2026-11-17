"""Backport vLLM PR #27027 inside the engine process, without editing site-packages."""
from vllm.v1.worker.gpu_worker import Worker


def install_language_only_mapping():
    from vllm.model_executor.models.qwen3_vl import Qwen3VLForConditionalGeneration
    from vllm.model_executor.models.module_mapping import MultiModelKeys

    def get_mm_mapping(self):
        # v0.11.0 incorrectly uses model.visual.* although actual modules are visual.*.
        # Exact dense-model correction from upstream PR #27027.
        return MultiModelKeys.from_string_field(
            language_model='language_model', connector='visual.merger', tower_model='visual.')

    Qwen3VLForConditionalGeneration.get_mm_mapping = get_mm_mapping


def audit_language_only_lora(model):
    manager = getattr(model, 'lora_manager', None)
    if manager is None or not manager.modules:
        raise RuntimeError('No LoRA modules registered in the vLLM model')
    names = list(manager.modules)
    if any(name.startswith(('visual.', 'model.visual.')) for name in names):
        raise RuntimeError('vLLM incorrectly wrapped the frozen vision encoder with LoRA')
    if not any(name.startswith('language_model.') for name in names):
        raise RuntimeError('vLLM did not wrap the language model with LoRA')


class Qwen3LanguageLoRAWorker(Worker):
    def load_model(self, *args, **kwargs):
        install_language_only_mapping()
        result = super().load_model(*args, **kwargs)
        if self.vllm_config.lora_config is not None:
            audit_language_only_lora(self.model_runner.model)
            print('[validation] Verified: LoRA wraps language modules only; vision stays unchanged', flush=True)
        return result
