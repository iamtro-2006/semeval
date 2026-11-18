"""Drive uploads happen in the launcher after successful training.

Intermediate checkpoints stay local. Keep this hook for existing callers.
"""


def upload_filters(kind):
    """Ordered rclone allowlists: new files stay local unless explicitly selected."""
    if kind in {'run', 'merged_run'}:
        rules = (['- /final/training_args.bin', '+ /final/**'] if kind == 'run'
                 else ['+ /best_model/**', '+ /selection.json'])
        rules += [
            '+ /run.json',
            '+ /swift_arguments.json',
            '+ /logging.jsonl',
            '+ /trainer_state.json',
            '+ /learning_curves.png',
            '+ /checkpoint_selection/ranking.json',
            '+ /prompt_comparison.json',
            '+ /prompt_comparison.csv',
            '+ /prompt_comparison.html',
            '+ /validation_cot/prompt.txt',
        ]
        for folder in ('validation', 'validation_cot', 'validation_subset_*'):
            for filename in ('scores.json', 'evaluation.json', 'predictions.jsonl', 'predictions.traces.jsonl'):
                rules.append(f'+ /{folder}/{filename}')
    elif kind == 'predictions':
        rules = []
        for stage in ('sft', 'grpo'):
            for split in ('dev', 'devtest', 'test'):
                name = f'{stage}_{split}'
                rules.extend((f'+ /{name}.jsonl', f'+ /{name}/run.json', f'+ /{name}_scores.json',
                              f'+ /{name}.traces.jsonl'))
    elif kind == 'data':
        # JSONL and Parquet contain the same examples; preparation only reads JSONL.
        rules = ['+ /REVISION']
        for split in ('train', 'dev', 'devtest'):
            rules.extend((f'+ /qa/mena/{split}_en.jsonl', f'+ /archives/images_mena_{split}.zip'))
    elif kind == 'reasoning':
        rules = [f'+ /{name}' for name in ('train_reasoning.jsonl', 'dataset.json', 'generation.json',
                                          'attempts.jsonl', 'rejected.jsonl', 'audit_sample.jsonl')]
    elif kind == 'inference_comparison':
        rules = ['+ /comparison.html', '+ /comparison.csv', '+ /comparison.json', '+ /settings.json']
        for model in ('base', 'sft'):
            for filename in ('config.json', 'predictions.jsonl', 'predictions.traces.jsonl',
                             'predictions.signature', 'scores.json', 'predictions/run.json'):
                rules.append(f'+ /{model}/{filename}')
    else:
        raise ValueError(f'Unknown upload kind: {kind}')
    rules.append('- **')
    return [argument for rule in rules for argument in ('--filter', rule)]


def checkpoint_callbacks():
    return []
