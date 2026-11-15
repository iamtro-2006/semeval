"""Drive uploads happen in the launcher after successful training.

Intermediate checkpoints stay local. Keep this hook for existing callers.
"""


def upload_filters(kind):
    """Ordered rclone allowlists: new files stay local unless explicitly selected."""
    if kind == 'run':
        rules = [
            '- /final/training_args.bin',
            '+ /final/**',  # PEFT adapter and processor/tokenizer assets
            '+ /run.json',
            '+ /swift_arguments.json',
            '+ /logging.jsonl',
            '+ /trainer_state.json',
            '+ /learning_curves.png',
        ]
        for folder in ('validation', 'validation_subset_*'):
            for filename in ('scores.json', 'evaluation.json', 'predictions.jsonl'):
                rules.append(f'+ /{folder}/{filename}')
    elif kind == 'predictions':
        rules = []
        for stage in ('sft', 'grpo'):
            for split in ('devtest', 'test'):
                name = f'{stage}_{split}'
                rules.extend((f'+ /{name}.jsonl', f'+ /{name}/run.json', f'+ /{name}_scores.json'))
    elif kind == 'data':
        # JSONL and Parquet contain the same examples; preparation only reads JSONL.
        rules = ['+ /REVISION']
        for split in ('train', 'dev', 'devtest'):
            rules.extend((f'+ /qa/mena/{split}_en.jsonl', f'+ /archives/images_mena_{split}.zip'))
    else:
        raise ValueError(f'Unknown upload kind: {kind}')
    rules.append('- **')
    return [argument for rule in rules for argument in ('--filter', rule)]


def checkpoint_callbacks():
    return []
