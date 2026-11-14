"""Configurable GRPO reward, distinct from the unchanged official evaluator."""


def completion_text(completion):
    if isinstance(completion, str):
        return completion
    pieces = []
    for message in completion:
        content = message['content']
        pieces.append(content if isinstance(content, str) else ''.join(
            block.get('text', '') for block in content if block.get('type') == 'text'))
    return ''.join(pieces)


def build_reward(cfg):
    rc = cfg['reward']
    if rc['type'] == 'exact':
        def exact(completions, answer, **kwargs):
            return [float(completion_text(c).strip() == a.strip()) for c, a in zip(completions, answer)]
        return exact
    if rc['type'] != 'mmbert':
        raise ValueError('reward.type must be mmbert or exact')
    from huggingface_hub import snapshot_download
    from bert_score import BERTScorer
    import torch
    directory = snapshot_download(rc['model'], revision=rc['revision'])
    scorer = BERTScorer(model_type=directory, num_layers=rc['num_layers'],
                       batch_size=rc['batch_size'], device=rc['device'], idf=False,
                       rescale_with_baseline=False, use_fast_tokenizer=True)
    scorer._tokenizer.model_max_length = 512
    scorer._model.eval()

    def mmbert(completions, answer, **kwargs):
        texts = [completion_text(c) for c in completions]
        result = [0.0] * len(texts)
        # Empty answers must receive zero, including whitespace-only generations.
        valid = [i for i, text in enumerate(texts) if text.strip()]
        if valid:
            with torch.inference_mode():
                _, _, f1 = scorer.score([texts[i] for i in valid], [answer[i] for i in valid])
            for i, score in zip(valid, f1.tolist()):
                result[i] = float(score)
        return result
    return mmbert
