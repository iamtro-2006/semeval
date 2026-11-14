"""Plot native ms-swift training logs without loading a model."""
import json
import math
from pathlib import Path


def load_points(run_dir):
    """Prefer full-run logs; checkpoints may only contain a partial history."""
    run_dir = Path(run_dir)
    logfile = run_dir / 'logging.jsonl'
    if logfile.is_file():
        records = [json.loads(line) for line in logfile.read_text(encoding='utf-8').splitlines() if line.strip()]
    else:
        statefile = run_dir / 'trainer_state.json'
        if not statefile.is_file():
            raise FileNotFoundError(f'No training log found in {run_dir}')
        records = json.loads(statefile.read_text(encoding='utf-8'))['log_history']
    points = {}
    for record in records:
        step = record.get('step')
        if step is None:
            step = str(record.get('global_step/max_steps', '')).split('/')[0]
        try:
            step = int(step)
        except (ValueError, TypeError):
            continue
        for key, value in record.items():
            if isinstance(value, (int, float)) and math.isfinite(value):
                # On resume, the last record for a step takes precedence.
                points.setdefault(key, {})[step] = float(value)
    return {key: sorted(values.items()) for key, values in points.items()}


def plot_learning_curves(run_dir, stage):
    import matplotlib.pyplot as plt
    if stage not in ('sft', 'grpo'):
        raise ValueError('stage must be sft or grpo')
    points = load_points(run_dir)
    if stage == 'sft':
        panels = [('Loss', [('loss', 'Train loss'), ('eval_loss', 'Validation loss')]),
                  ('Generation validation F1', [('eval_bertscore_f1', 'Validation mmBERT F1')])]
    else:
        reward_keys = [key for key in points if key.startswith('rewards/') and key.endswith('/mean')]
        panels = [('Reward', [('reward', 'Train reward'), ('eval_reward', 'Validation reward')]
                   + [(key, 'Train ' + key) for key in reward_keys]
                   + [('eval_' + key, 'Validation ' + key) for key in reward_keys]),
                  ('Optimization loss (not QA accuracy)', [('loss', 'Train loss')])]
    panels = [(title, [(key, label) for key, label in series if points.get(key)])
              for title, series in panels]
    panels = [(title, series) for title, series in panels if series]
    if not panels:
        raise ValueError('Training log contains no plottable loss/reward metrics')
    fig, axes = plt.subplots(1, len(panels), figsize=(8 * len(panels), 4.5), squeeze=False)
    for ax, (title, series) in zip(axes[0], panels):
        for key, label in series:
            x, y = zip(*points[key])
            ax.plot(x, y, label=label, marker='o' if key.startswith('eval_') else None,
                    markersize=3)
        ax.set(title=f'{stage.upper()} - {title}', xlabel='Optimizer step')
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    if not any(key.startswith('eval_') and ('loss' in key or 'reward' in key or 'bertscore' in key)
               for key in points):
        print('No periodic validation metrics logged (e.g. smoke run or evaluation interval not reached).')
    fig.tight_layout()
    destination = Path(run_dir) / 'learning_curves.png'
    fig.savefig(destination, dpi=160, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print(f'Learning curves saved to {destination}')
    return destination
