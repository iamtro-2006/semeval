"""CPU regression checks for Colab paths/imports; no model downloads or GPU needed.

Run: python -m unittest discover -s "task 6/tests" -v
"""
import ast
from fnmatch import fnmatchcase
import json
import os
import random
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]


def cells(name):
    notebook = json.loads((PROJECT / 'notebooks' / f'{name}.ipynb').read_text(encoding='utf-8'))
    return [''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code']


def find_cell(name, marker):
    return next(source for source in cells(name) if marker in source)


class ColabNotebookTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.content = Path(self.temporary.name) / 'content'
        self.project = self.content / 'semeval' / 'task 6'
        self.project.mkdir(parents=True)
        shutil.copytree(PROJECT / 'src', self.project / 'src', ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copytree(PROJECT / 'configs', self.project / 'configs')
        shutil.copytree(PROJECT / 'prompts', self.project / 'prompts')
        self.old_cwd = Path.cwd()
        self.old_path = list(sys.path)
        self.old_modules = {k: v for k, v in sys.modules.items() if k == 'mmcultureqa' or k.startswith('mmcultureqa.')}
        self.addCleanup(self.restore)

    def restore(self):
        os.chdir(self.old_cwd)
        sys.path[:] = self.old_path
        for name in list(sys.modules):
            if name == 'mmcultureqa' or name.startswith('mmcultureqa.'):
                del sys.modules[name]
        sys.modules.update(self.old_modules)

    def test_every_cell_compiles_and_local_imports_exist(self):
        for notebook in (PROJECT / 'notebooks').glob('*.ipynb'):
            for source in cells(notebook.stem):
                tree = ast.parse(source, filename=notebook.name)
                for node in ast.walk(tree):
                    names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                             else [node.module] if isinstance(node, ast.ImportFrom) else [])
                    for name in names:
                        if name and name.startswith('mmcultureqa'):
                            local = PROJECT / 'src' / Path(*name.split('.'))
                            self.assertTrue(local.with_suffix('.py').is_file() or (local / '__init__.py').is_file(), name)

    def test_setup_imports_in_current_kernel_and_child_after_chdir(self):
        source = find_cell('02_sft_colab', 'REPO_URL =').replace(
            "CHECKOUT = '/content/semeval'", f'CHECKOUT = {str(self.project.parent)!r}')
        real_popen = subprocess.Popen
        children = []

        def popen(command, **kwargs):
            if command[0] == 'git' or command[1:3] == ['-m', 'pip']:
                command = [sys.executable, '-c', 'pass']
            else:
                children.append((command, kwargs))
            return real_popen(command, **kwargs)

        def git_output(command, **kwargs):
            if command[-1] == 'origin':
                return 'https://github.com/iamtro-2006/semeval.git\n'
            return '' if command[-1] == '--porcelain' else 'test-commit\n'

        ns = {}
        os.chdir(self.content)
        # Simulate an already cached module from a previous checkout.
        from types import ModuleType
        with patch.dict(os.environ):
            sys.modules['mmcultureqa'] = ModuleType('mmcultureqa')
            with patch('subprocess.Popen', side_effect=popen), patch('subprocess.check_output', side_effect=git_output):
                exec(source, ns)
            self.assertEqual(ns['PACKAGE_ROOT'], self.project)
            self.assertEqual(sys.path[0], str(self.project / 'src'))
            os.chdir(self.content)
            probe = '''from pathlib import Path
from mmcultureqa.common import ROOT, path
import mmcultureqa.data.prepare
import mmcultureqa.training.sft
import mmcultureqa.training.grpo
import mmcultureqa.training.curves
import mmcultureqa.inference.predict
import mmcultureqa.evaluate
import mmcultureqa.score
assert path('outputs/runs/test') == ROOT / 'outputs/runs/test'
print(ROOT)
'''
            result = ns['stream_run']([sys.executable, '-c', probe])
            self.assertEqual(result.returncode, 0)
        self.assertEqual(children[-1][1]['cwd'], self.project)

    def experiment(self, notebook='02_sft_colab'):
        commands, copies = [], []
        self.copy_options = []
        sys.path.insert(0, str(self.project / 'src'))

        def adapter(folder):
            folder.mkdir(parents=True, exist_ok=True)
            (folder / 'adapter_config.json').write_text('{}', encoding='utf-8')
            (folder / 'adapter_model.safetensors').write_bytes(b'fixture')

        def copy(source, destination, *args):
            copies.append((str(source), str(destination)))
            self.copy_options.append(args)
            if str(source).startswith('remote:'):
                adapter(Path(destination))

        def stream_run(command, check=True):
            commands.append(command)
            self.assertTrue(Path(command[command.index('--config') + 1]).is_absolute())
            overrides = {command[i + 1].split('=', 1)[0]: json.loads(command[i + 1].split('=', 1)[1])
                         for i, value in enumerate(command) if value == '--set'}
            if command[2] == 'mmcultureqa.inference.predict':
                output = Path(overrides['inference.output'])
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text('{"id": "test", "answer": "ok"}\n', encoding='utf-8')
            else:
                adapter(Path(overrides['training.output_dir']) / 'final')
            return subprocess.CompletedProcess(command, 0)

        import re
        ns = dict(PROJECT_ROOT=self.project, Path=Path, re=re, sys=sys, stream_run=stream_run,
                  copy=copy, cloud=lambda p: f'remote:{p}')
        exec(find_cell(notebook, 'USE_4BIT ='), ns)
        os.chdir(self.content)
        return ns, commands, copies

    def test_training_uses_absolute_configs_outputs_and_adapter(self):
        ns, commands, copies = self.experiment()
        exec(find_cell('02_sft_colab', 'SFT_OVERRIDES ='), ns)
        exec(find_cell('03_grpo_colab', 'SFT_RUN_NAME ='), ns)
        self.assertEqual([c[2] for c in commands], ['mmcultureqa.training.sft', 'mmcultureqa.training.grpo'])
        self.assertTrue((Path(ns['GRPO']) / 'final/adapter_config.json').is_file())
        self.assertTrue(all(Path(source).is_absolute() for source, dest in copies if dest.startswith('remote:')))
        from mmcultureqa.storage import upload_filters
        self.assertEqual(self.copy_options[0], tuple(upload_filters('run')))

    def test_inference_uploads_both_predictions_without_training_final(self):
        ns, commands, copies = self.experiment()
        exec(find_cell('04_inference_colab', 'INPUT_RUN_NAME ='), ns)
        predictions = ns['predictions']
        self.assertTrue((predictions / 'sft_devtest.jsonl').is_file())
        self.assertTrue((predictions / 'grpo_devtest.jsonl').is_file())
        self.assertFalse((predictions / 'final').exists())
        uploads = [(source, dest) for source, dest in copies if dest.startswith('remote:')]
        self.assertEqual(len(uploads), 2)
        self.assertTrue(all(source == str(predictions) for source, dest in uploads))
        self.assertEqual([c[2] for c in commands], ['mmcultureqa.inference.predict'] * 2)
        self.assertIn('inference.backend="transformers"', commands[0])
        self.assertIn('model.load_in_4bit=true', commands[0])
        self.assertIn('inference.backend="transformers"', commands[1])
        self.assertIn('model.load_in_4bit=true', commands[1])
        self.assertIn('remote:task6-runs/experiment01_qlora/grpo/final', copies[2])
        from mmcultureqa.storage import upload_filters
        self.assertEqual(self.copy_options[1], tuple(upload_filters('predictions')))
        self.assertEqual(self.copy_options[3], tuple(upload_filters('predictions')))

    def test_upload_allowlists_keep_only_selected_artifacts(self):
        sys.path.insert(0, str(self.project / 'src'))
        from mmcultureqa.storage import upload_filters

        def selected(kind, filename):
            args = upload_filters(kind)
            for flag, rule in zip(args[::2], args[1::2]):
                self.assertEqual(flag, '--filter')
                decision, pattern = rule.split(' ', 1)
                if fnmatchcase(filename, pattern.lstrip('/')):
                    return decision == '+'
            self.fail('Every upload allowlist must end with a deny-all rule')

        keep = {
            'run': ['final/adapter_config.json', 'final/adapter_model.safetensors',
                    'final/adapter_model.bin', 'final/tokenizer.json', 'final/preprocessor_config.json',
                    'final/video_preprocessor_config.json', 'final/chat_template.jinja',
                    'run.json', 'swift_arguments.json', 'logging.jsonl', 'trainer_state.json',
                    'learning_curves.png', 'validation/scores.json', 'validation/evaluation.json',
                    'validation/predictions.jsonl', 'validation_subset_4/scores.json'],
            'predictions': ['sft_devtest.jsonl', 'grpo_devtest.jsonl', 'sft_test.jsonl',
                            'grpo_test/run.json', 'sft_test_scores.json'],
            'data': ['REVISION', 'qa/mena/train_en.jsonl', 'qa/mena/dev_en.jsonl',
                     'qa/mena/devtest_en.jsonl', 'archives/images_mena_train.zip',
                     'archives/images_mena_dev.zip', 'archives/images_mena_devtest.zip'],
        }
        drop = {
            'run': ['checkpoint-170/adapter_model.safetensors', 'checkpoint-170/optimizer.pt',
                    'optimizer.pt', 'swift_data/train.jsonl', 'final/training_args.bin',
                    'validation/gold_en.jsonl', 'validation/inference_config.json',
                    'validation/prompt.txt', 'validation/predictions.signature',
                    'validation/predictions/run.json', 'tensorboard/events.out.tfevents',
                    'some_new_directory/large_file.bin'],
            'predictions': ['devtest_gold_en.jsonl', 'sft_devtest.signature',
                            'some_new_file.jsonl', 'some_new_directory/cache.bin'],
            'data': ['qa/mena/train_en.parquet', 'README.md', '.cache/huggingface/download/metadata',
                     'train/images/example.jpg', 'qa/mena/train_arz.jsonl', 'archives/audio.zip'],
        }
        for kind, filenames in keep.items():
            for filename in filenames:
                with self.subTest(kind=kind, keep=filename):
                    self.assertTrue(selected(kind, filename))
        for kind, filenames in drop.items():
            for filename in filenames:
                with self.subTest(kind=kind, drop=filename):
                    self.assertFalse(selected(kind, filename))
        # Where installed, verify rclone's actual directory traversal and rule order too.
        rclone = shutil.which('rclone')
        if rclone:
            for kind in keep:
                folder = self.content / f'upload-fixture-{kind}'
                for filename in keep[kind] + drop[kind]:
                    file = folder / filename
                    file.parent.mkdir(parents=True, exist_ok=True)
                    file.write_text('fixture', encoding='utf-8')
                result = subprocess.check_output(
                    [rclone, 'lsjson', str(folder), '--recursive', '--files-only', *upload_filters(kind)],
                    text=True, encoding='utf-8')
                self.assertEqual({entry['Path'] for entry in json.loads(result)}, set(keep[kind]))
        with self.assertRaises(ValueError):
            upload_filters('unknown')

    def test_incomplete_adapter_fails_before_gpu_command(self):
        ns, commands, _ = self.experiment()
        folder = self.project / 'outputs/broken'
        folder.mkdir(parents=True)
        (folder / 'adapter_config.json').write_text('{}', encoding='utf-8')
        with self.assertRaisesRegex(FileNotFoundError, 'weights missing'):
            ns['require_adapter'](folder)
        self.assertFalse(commands)

    def test_grpo_notebook_quantizes_and_uses_separate_output_run(self):
        ns, commands, copies = self.experiment('03_grpo_colab')
        self.assertTrue(ns['USE_4BIT'])
        self.assertEqual(ns['RUN_NAME'], 'experiment01_qlora')
        exec(find_cell('03_grpo_colab', 'SFT_RUN_NAME ='), ns)
        self.assertEqual(ns['SFT_RUN_NAME'], 'experiment01')
        self.assertIn('remote:task6-runs/experiment01/sft/final', copies[0])
        command = commands[0]
        self.assertIn('model.load_in_4bit=true', command)
        validation = []
        ns['stream_run'] = lambda cmd, check=True: validation.append(cmd) or subprocess.CompletedProcess(cmd, 0)
        exec(find_cell('03_grpo_colab', 'VAL_LIMIT ='), ns)
        self.assertEqual(validation[0][-2:], ['--backend', 'transformers'])
        install = find_cell('03_grpo_colab', 'INSTALL_VLLM =')
        self.assertIn('INSTALL_VLLM = False', install)

    def test_all_gpu_notebooks_default_to_4bit_generation_without_vllm(self):
        for notebook in ('02_sft_colab', '03_grpo_colab', '04_inference_colab'):
            ns, _, _ = self.experiment(notebook)
            self.assertTrue(ns['USE_4BIT'], notebook)
            self.assertTrue(ns['PREDICT_WITH_GENERATE'], notebook)
            install = ast.parse(find_cell(notebook, 'INSTALL_VLLM ='))
            assignment = next(node for node in install.body if isinstance(node, ast.Assign)
                              and any(isinstance(t, ast.Name) and t.id == 'INSTALL_VLLM' for t in node.targets))
            self.assertIs(assignment.value.value, False)

    def grpo_config(self):
        sys.path.insert(0, str(self.project / 'src'))
        from mmcultureqa.common import config, write_jsonl
        with patch.object(sys, 'argv', ['test', '--config', str(self.project / 'configs/grpo.yaml')]):
            cfg = config()
        manifest = self.project / 'data/prepared'
        write_jsonl(manifest / 'train.jsonl', [dict(id='train-1', image='train.jpg', question='Train?', answer='train-gold')])
        write_jsonl(manifest / 'validation.jsonl', [dict(id='val-1', image='val.jpg', question='Val?', answer='val-gold')])
        adapter = self.project / 'outputs/sft/final'
        adapter.mkdir(parents=True)
        (adapter / 'adapter_config.json').write_text('{"r": 16}', encoding='utf-8')
        return cfg

    def test_native_grpo_arguments_use_qlora_and_batched_transformers(self):
        cfg = self.grpo_config()
        from mmcultureqa.training.swift_backend import build_arguments
        with patch.dict(os.environ, WORLD_SIZE='1'):
            args = build_arguments(cfg, 'grpo')
        self.assertEqual(args['quant_method'], 'bnb')
        self.assertEqual(args['quant_bits'], 4)
        self.assertEqual(args['bnb_4bit_quant_type'], 'nf4')
        self.assertTrue(args['bnb_4bit_use_double_quant'])
        self.assertEqual(args['bnb_4bit_compute_dtype'], 'bfloat16')
        self.assertEqual(args['train_type'], 'lora')
        self.assertFalse(args['use_vllm'])
        self.assertIsNone(args['vllm_mode'])
        self.assertFalse(args['offload_model'])
        self.assertFalse(args['offload_optimizer'])
        self.assertEqual(args['sleep_level'], 0)
        self.assertEqual(args['per_device_train_batch_size'], 4)
        self.assertEqual(args['per_device_eval_batch_size'], 8)
        self.assertEqual(args['per_device_train_batch_size'] * args['steps_per_generation'], 16)
        self.assertEqual(args['num_generations'], 4)
        self.assertEqual(args['reward_funcs'], ['mmcqa_bertscore'])
        self.assertEqual(args['beta'], 0)
        self.assertNotIn('ref_adapters', args)
        self.assertTrue(args['load_best_model_at_end'])
        records = [json.loads(line) for line in Path(args['dataset'][0]).read_text(encoding='utf-8').splitlines()]
        self.assertEqual(records[0]['solution'], 'train-gold')
        self.assertNotIn('train-gold', json.dumps(records[0]['messages']))

    def test_native_sft_uses_qlora_generation_and_preserves_effective_batch(self):
        self.grpo_config()  # create disjoint train/validation fixture manifests
        from mmcultureqa.common import config
        from mmcultureqa.training.swift_backend import build_arguments
        with patch.object(sys, 'argv', ['test', '--config', str(self.project / 'configs/sft.yaml')]):
            cfg = config()
        self.assertIsNone(cfg['data']['limit'])
        self.assertEqual(cfg['monitor']['limit'], 256)
        args = build_arguments(cfg, 'sft')
        self.assertEqual(args['quant_bits'], 4)
        self.assertEqual(args['bnb_4bit_quant_type'], 'nf4')
        self.assertEqual(args['bnb_4bit_compute_dtype'], 'bfloat16')
        self.assertTrue(args['predict_with_generate'])
        self.assertEqual(args['metric'], 'mmcqa_bertscore')
        self.assertEqual(args['metric_for_best_model'], 'bertscore_f1')
        self.assertTrue(args['greater_is_better'])
        self.assertEqual(args['per_device_train_batch_size'], 8)
        self.assertEqual(args['gradient_accumulation_steps'], 1)
        self.assertEqual(args['per_device_train_batch_size'] * args['gradient_accumulation_steps'], 8)
        self.assertEqual(args['per_device_eval_batch_size'], 8)
        cfg['training']['predict_with_generate'] = False
        teacher_forced = build_arguments(cfg, 'sft')
        self.assertIsNone(teacher_forced['metric'])
        self.assertEqual(teacher_forced['metric_for_best_model'], 'loss')
        self.assertFalse(teacher_forced['greater_is_better'])

    def test_grpo_checks_eval_grouping_after_overrides(self):
        cfg = self.grpo_config()
        from mmcultureqa.training.swift_backend import build_arguments, validate_grpo_batching
        with patch.dict(os.environ, WORLD_SIZE='1'):
            args = build_arguments(cfg, 'grpo')
            for batch in (1, 2, 4):
                validate_grpo_batching({**args, 'per_device_train_batch_size': batch})
            with self.assertRaisesRegex(ValueError, 'eval batch'):
                validate_grpo_batching({**args, 'per_device_eval_batch_size': 1})
            with self.assertRaisesRegex(ValueError, 'generation_batch_size'):
                validate_grpo_batching({**args, 'generation_batch_size': 15})
            with self.assertRaisesRegex(ValueError, 'positive'):
                validate_grpo_batching({**args, 'generation_batch_size': 0})
            cfg['swift'] = {'per_device_eval_batch_size': 1}
            with self.assertRaisesRegex(ValueError, 'eval batch'):
                build_arguments(cfg, 'grpo')

    def test_transformers_rollout_clears_stale_vllm_mode(self):
        cfg = self.grpo_config()
        from mmcultureqa.training.swift_backend import build_arguments
        with patch.dict(os.environ, WORLD_SIZE='1'):
            cfg['training']['vllm_mode'] = 'colocate'
            self.assertIsNone(build_arguments(cfg, 'grpo')['vllm_mode'])
            cfg['swift'] = {'vllm_mode': 'server'}
            self.assertIsNone(build_arguments(cfg, 'grpo')['vllm_mode'])
            cfg['swift'] = {'use_vllm': True, 'vllm_mode': 'colocate'}
            self.assertEqual(build_arguments(cfg, 'grpo')['vllm_mode'], 'colocate')

    def test_monitor_subset_is_seed42_reproducible_and_shared_by_stages(self):
        cfg = self.grpo_config()
        from mmcultureqa.common import write_jsonl
        from mmcultureqa.training.swift_backend import build_arguments
        rows = [dict(id=f'val-{i}', image=f'val-{i}.jpg', question=f'Question {i}?', answer=f'Answer {i}')
                for i in range(30)]
        manifest = self.project / 'data/prepared/validation.jsonl'
        write_jsonl(manifest, rows)
        cfg['monitor']['limit'] = 8
        cfg['training']['seed'] = 123  # validation sampling stays at seed 42
        expected = [str(self.project / r['image']) for r in random.Random(42).sample(rows, 8)]
        before = random.getstate()
        with patch.dict(os.environ, WORLD_SIZE='1'):
            for stage in ('sft', 'grpo', 'grpo'):
                args = build_arguments(cfg, stage)
                records = [json.loads(line) for line in Path(args['val_dataset'][0]).read_text(encoding='utf-8').splitlines()]
                self.assertEqual([r['images'][0] for r in records], expected)
                self.assertEqual(len(set(r['images'][0] for r in records)), 8)
        self.assertEqual(random.getstate(), before)
        self.assertEqual(len(manifest.read_text(encoding='utf-8').splitlines()), 30)

    def test_monitor_full_validation_keeps_original_order(self):
        cfg = self.grpo_config()
        from mmcultureqa.training.swift_backend import build_arguments
        with patch.dict(os.environ, WORLD_SIZE='1'):
            for limit in (None, 1, 256):
                cfg['monitor']['limit'] = limit
                args = build_arguments(cfg, 'grpo')
                records = [json.loads(line) for line in Path(args['val_dataset'][0]).read_text(encoding='utf-8').splitlines()]
                self.assertEqual([r['id'] for r in records], ['val-1'])

    def test_quantized_evaluator_automatically_selects_transformers(self):
        cfg = self.grpo_config()
        from mmcultureqa import evaluate
        run = self.project / 'outputs/qlora-test'
        (run / 'final').mkdir(parents=True)
        (run / 'final/adapter_config.json').write_text('{}', encoding='utf-8')
        cfg['prompt_text'] = (self.project / cfg['prompt_file']).read_text(encoding='utf-8')
        (run / 'run.json').write_text(json.dumps(cfg), encoding='utf-8')
        inferred = []

        def child(command, **kwargs):
            if command[2] == 'mmcultureqa.inference.predict':
                config_file = Path(command[command.index('--config') + 1])
                frozen = json.loads(config_file.read_text(encoding='utf-8'))
                inferred.append(frozen)
                Path(frozen['inference']['output']).write_text(
                    '{"id": "val-1", "answer": "fixture"}\n', encoding='utf-8')
            else:
                Path(command[command.index('--out') + 1]).write_text('{}', encoding='utf-8')
            return subprocess.CompletedProcess(command, 0)

        with patch.object(sys, 'argv', ['evaluate', '--run-dir', str(run)]), patch('subprocess.run', side_effect=child):
            evaluate.main()
        self.assertTrue(inferred[0]['model']['load_in_4bit'])
        self.assertEqual(inferred[0]['inference']['backend'], 'transformers')
        self.assertIsNone(inferred[0]['data']['limit'])


if __name__ == '__main__':
    unittest.main()
