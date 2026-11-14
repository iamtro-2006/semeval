"""Optional checkpoint upload configured by the Colab launcher environment."""
import os
import subprocess


def checkpoint_callbacks():
    if not os.environ.get('MMCQA_SYNC_DESTINATION'):
        return []
    from transformers import TrainerCallback

    class SyncCheckpoint(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            if state.is_world_process_zero:
                subprocess.run([
                    'rclone', '--config', os.environ['MMCQA_RCLONE_CONFIG'],
                    '--drive-root-folder-id', os.environ['MMCQA_DRIVE_ROOT_ID'],
                    'copy', args.output_dir, os.environ['MMCQA_SYNC_DESTINATION'],
                    '--transfers', '4', '--checkers', '8',
                ], check=True)

    return [SyncCheckpoint()]
