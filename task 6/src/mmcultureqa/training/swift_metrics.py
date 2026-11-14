"""mmBERT metric/reward plugins for ms-swift 3.11.3."""
import json
from functools import lru_cache


@lru_cache(maxsize=4)
def get_scorer(settings_json):
    import torch
    from bert_score import BERTScorer
    from huggingface_hub import snapshot_download
    rc = json.loads(settings_json)
    dtype = rc.get('dtype', 'float32')
    if dtype not in {'float16', 'float32'}:
        raise ValueError('mmBERT dtype must be float16 or float32')
    if dtype == 'float16' and not rc['device'].startswith('cuda'):
        raise ValueError('FP16 mmBERT requires CUDA; set dtype=float32 for CPU')
    directory = snapshot_download(rc['model'], revision=rc['revision'])
    scorer = BERTScorer(model_type=directory, num_layers=rc['num_layers'], batch_size=rc['batch_size'],
                       device=rc['device'], idf=False, rescale_with_baseline=False, use_fast_tokenizer=True)
    scorer._tokenizer.model_max_length = 512
    scorer._model.to(dtype=getattr(torch, dtype))
    scorer._model.eval()
    return scorer


def score_texts(predictions, references, settings):
    import torch
    if len(predictions) != len(references):
        raise ValueError('Predictions and references must have the same length')
    predictions = [str(x).strip()[:5000] for x in predictions]
    references = [str(x).strip()[:5000] for x in references]
    result = {key: [0.0] * len(predictions) for key in ('precision', 'recall', 'f1')}
    valid = [i for i, text in enumerate(predictions) if text]
    if valid:
        scorer = preload_scorer(settings)
        with torch.inference_mode():
            values = scorer.score([predictions[i] for i in valid], [references[i] for i in valid])
        for key, tensor in zip(result, values):
            if not torch.isfinite(tensor).all():
                raise FloatingPointError('Non-finite mmBERT score; check inputs or select dtype=float32')
            for index, value in zip(valid, tensor.tolist()):
                result[key][index] = float(value)
    return result


def preload_scorer(settings):
    values = {k: settings[k] for k in ('model', 'revision', 'num_layers', 'batch_size', 'device')}
    values['dtype'] = settings.get('dtype', 'float32')
    return get_scorer(json.dumps(values, sort_keys=True))


def register_plugins(cfg):
    from swift.plugin import metric_mapping, orms, ORM
    from swift.utils import Serializer

    def generation_bertscore(prediction):
        predictions = [Serializer.from_tensor(x) for x in prediction.predictions]
        references = [Serializer.from_tensor(x) for x in prediction.label_ids]
        scores = score_texts(predictions, references, cfg['monitor'])
        return {f'bertscore_{name}': sum(values) / len(values) for name, values in scores.items()}

    class MMCQABERTScore(ORM):
        def __call__(self, completions, solution, **kwargs):
            return score_texts(completions, solution, cfg['reward'])['f1']

    class MMCQAExact(ORM):
        def __call__(self, completions, solution, **kwargs):
            return [float(c.strip() == s.strip()) for c, s in zip(completions, solution)]

    metric_mapping['mmcqa_bertscore'] = (generation_bertscore, None)
    orms['mmcqa_bertscore'] = MMCQABERTScore
    orms['mmcqa_exact'] = MMCQAExact
