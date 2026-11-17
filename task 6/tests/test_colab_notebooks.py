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
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

PROJECT = Path(__file__).resolve().parents[1]


def cells(name):
    notebook = json.loads((PROJECT / 'notebooks' / f'{name}.ipynb').read_text(encoding='utf-8'))
    return [''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code']


def find_cell(name, marker):
    return next(source for source in cells(name) if marker in source)


def merged_fixture(folder):
    folder.mkdir(parents=True, exist_ok=True)
    files = {'config.json': {'model_type': 'qwen3_vl', 'torch_dtype': 'bfloat16'},
             'tokenizer_config.json': {}, 'preprocessor_config.json': {}, 'tokenizer.json': {},
             'merge.json': {'signature': 'fixture'},
             'run_config.json': {'model': {'min_pixels': 65536, 'max_pixels': 262144},
                                 'system_prompt': 'Saved cultural QA system prompt'}}
    for name, value in files.items():
        (folder / name).write_text(json.dumps(value), encoding='utf-8')
    (folder / 'model.safetensors').write_bytes(b'fixture-bf16-weights')
    (folder / 'prompt.txt').write_text('Saved prompt: {question}', encoding='utf-8')


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

    def experiment(self, notebook='02_sft_colab', smoke=False):
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
                if str(source).endswith('/best_model'):
                    merged_fixture(Path(destination))
                else:
                    adapter(Path(destination))

        def stream_run(command, check=True):
            commands.append(command)
            if command[2] in ('mmcultureqa.export', 'mmcultureqa.select_checkpoint'):
                merged_fixture(Path(command[command.index('--run-dir') + 1]) / 'best_model')
                return subprocess.CompletedProcess(command, 0)
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
        settings = find_cell(notebook, 'USE_4BIT =')
        if smoke:
            settings = settings.replace('RUN_FULL = True', 'RUN_FULL = False')
        exec(settings, ns)
        os.chdir(self.content)
        return ns, commands, copies

    def test_training_uses_absolute_configs_outputs_and_adapter(self):
        ns, commands, copies = self.experiment()
        exec(find_cell('02_sft_colab', 'SFT_OVERRIDES ='), ns)
        self.assertEqual([c[2] for c in commands], ['mmcultureqa.training.sft'])
        exec(find_cell('02_sft_colab', 'mmcultureqa.select_checkpoint'), ns)
        self.assertEqual(commands[-1][2], 'mmcultureqa.select_checkpoint')
        self.assertTrue((Path(ns['SFT']) / 'best_model/model.safetensors').is_file())
        self.assertTrue(all(Path(source).is_absolute() for source, dest in copies if dest.startswith('remote:')))
        from mmcultureqa.storage import upload_filters
        self.assertEqual(self.copy_options[0], tuple(upload_filters('merged_run')))
        ns, commands, _ = self.experiment('03_grpo_colab')
        exec(find_cell('03_grpo_colab', 'SFT_RUN_NAME ='), ns)
        self.assertEqual(commands[0][2], 'mmcultureqa.training.grpo')
        self.assertEqual(self.copy_options[1], tuple(upload_filters('run')))

    def test_inference_downloads_only_merged_experiment03_model(self):
        ns, commands, copies = self.experiment('04_inference_colab')
        exec(find_cell('04_inference_colab', 'INPUT_RUN_NAME ='), ns)
        predictions = ns['predictions']
        self.assertTrue((predictions / 'sft_devtest.jsonl').is_file())
        self.assertFalse((predictions / 'grpo_devtest.jsonl').exists())
        self.assertFalse((predictions / 'final').exists())
        uploads = [(source, dest) for source, dest in copies if dest.startswith('remote:')]
        self.assertEqual(len(uploads), 1)
        self.assertTrue(all(source == str(predictions) for source, dest in uploads))
        self.assertEqual([c[2] for c in commands], ['mmcultureqa.inference.predict'])
        self.assertIn('inference.backend="vllm"', commands[0])
        self.assertIn('model.load_in_4bit=false', commands[0])
        self.assertIn('model.local_files_only=true', commands[0])
        self.assertIn('model.adapter=null', commands[0])
        self.assertIn('remote:task6-runs/experiment03/sft/best_model', copies[0])
        from mmcultureqa.storage import upload_filters
        self.assertEqual(self.copy_options[1], tuple(upload_filters('predictions')))

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
                            'grpo_test/run.json', 'sft_test_scores.json', 'sft_devtest.traces.jsonl'],
            'reasoning': ['train_reasoning.jsonl', 'dataset.json', 'generation.json',
                          'attempts.jsonl', 'rejected.jsonl', 'audit_sample.jsonl'],
            'inference_comparison': ['settings.json', 'comparison.html', 'comparison.csv', 'comparison.json',
                                     'base/predictions.jsonl', 'base/config.json', 'base/predictions/run.json',
                                     'sft/scores.json', 'sft/predictions.traces.jsonl', 'sft/predictions.signature'],
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
            'reasoning': ['teacher_weights/model.safetensors', 'images/example.jpg', 'cache.bin'],
            'inference_comparison': ['gold_en.jsonl', 'best_model/model.safetensors', 'sft/optimizer.pt', 'cache.bin'],
            'data': ['qa/mena/train_en.parquet', 'README.md', '.cache/huggingface/download/metadata',
                     'train/images/example.jpg', 'qa/mena/train_arz.jsonl', 'archives/audio.zip'],
        }
        keep['merged_run'] = ['best_model/model.safetensors', 'best_model/config.json',
                              'best_model/tokenizer.json', 'best_model/merge.json',
                              'best_model/prompt.txt', 'best_model/run_config.json', 'selection.json',
                              'run.json', 'logging.jsonl', 'learning_curves.png', 'validation/scores.json',
                              'validation/predictions.traces.jsonl', 'validation_cot/scores.json',
                              'validation_cot/predictions.jsonl', 'validation_cot/predictions.traces.jsonl',
                              'validation_cot/prompt.txt',
                              'prompt_comparison.html', 'prompt_comparison.csv', 'prompt_comparison.json']
        drop['merged_run'] = ['final/adapter_model.safetensors', 'final/adapter_config.json',
                              'checkpoint-225/adapter_model.safetensors', 'checkpoint-225/optimizer.pt',
                              'swift_data/train.jsonl', '.merge-failed/best_model/model.safetensors',
                              'validation/gold_en.jsonl']
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

    def test_notebooks_use_expected_precision_and_monitoring(self):
        profiles = [('02_sft_colab', False, False), ('03_grpo_colab', True, True),
                    ('04_inference_colab', False, False)]
        for notebook, fourbit, generate in profiles:
            ns, _, _ = self.experiment(notebook)
            self.assertEqual(ns['USE_4BIT'], fourbit, notebook)
            self.assertEqual(ns['PREDICT_WITH_GENERATE'], generate, notebook)
            install = ast.parse(find_cell(notebook, 'INSTALL_VLLM ='))
            assignment = next(node for node in install.body if isinstance(node, ast.Assign)
                              and any(isinstance(t, ast.Name) and t.id == 'INSTALL_VLLM' for t in node.targets))
            self.assertIs(assignment.value.value, notebook != '03_grpo_colab')

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
        self.assertEqual(cfg['monitor']['limit'], 256)
        from mmcultureqa.training.swift_backend import build_arguments
        with patch.dict(os.environ, WORLD_SIZE='1'):
            args = build_arguments(cfg, 'grpo')
        self.assertNotIn('eval_on_start', args)
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

    def test_qwen3_vl_smoothing_uses_next_token_alignment(self):
        sys.path.insert(0, str(self.project / 'src'))
        from mmcultureqa.training.swift_backend import configure_causal_label_smoothing
        native = MagicMock(return_value=0.5)
        trainer = SimpleNamespace(label_smoother=native)
        outputs, labels = object(), [-100, 7, 9, -100]
        configure_causal_label_smoothing(trainer, 0.005)
        self.assertEqual(trainer.label_smoother(outputs, labels), 0.5)
        native.assert_called_once_with(outputs, labels, shift_labels=True)
        unchanged = SimpleNamespace(label_smoother=None)
        configure_causal_label_smoothing(unchanged, 0.)
        self.assertIsNone(unchanged.label_smoother)
        with self.assertRaisesRegex(RuntimeError, 'initialize'):
            configure_causal_label_smoothing(unchanged, 0.005)

    def test_native_sft_selects_generated_f1_every_125_steps(self):
        self.grpo_config()  # create disjoint train/validation fixture manifests
        from mmcultureqa.common import config
        from mmcultureqa.training.swift_backend import build_arguments
        with patch.object(sys, 'argv', ['test', '--config', str(self.project / 'configs/sft.yaml')]):
            cfg = config()
        self.assertIsNone(cfg['data']['limit'])
        self.assertIsNone(cfg['monitor']['limit'])
        from mmcultureqa.common import write_jsonl
        write_jsonl(self.project / 'data/prepared/validation.jsonl', [
            dict(id=f'val-{i}', image=f'val-{i}.jpg', question='Val?', answer='Gold') for i in range(1000)])
        args = build_arguments(cfg, 'sft')
        self.assertEqual(len(Path(args['val_dataset'][0]).read_text(encoding='utf-8').splitlines()), 1000)
        self.assertEqual(args['acc_strategy'], 'token')
        self.assertNotIn('quant_bits', args)
        self.assertFalse(args['predict_with_generate'])
        self.assertIsNone(args['metric'])
        self.assertEqual(args['metric_for_best_model'], 'loss')
        self.assertFalse(args['greater_is_better'])
        self.assertIsNone(args['save_total_limit'])
        self.assertFalse(args['load_best_model_at_end'])
        self.assertEqual(args['num_train_epochs'], 1.2)
        self.assertEqual(args['learning_rate'], 2e-4)
        self.assertEqual(args['label_smoothing_factor'], 0.005)
        self.assertEqual(args['eval_steps'], 125)
        self.assertEqual(args['save_steps'], 225)
        self.assertEqual(args['per_device_train_batch_size'], 8)
        self.assertEqual(args['gradient_accumulation_steps'], 1)
        self.assertEqual(args['per_device_train_batch_size'] * args['gradient_accumulation_steps'], 8)
        self.assertEqual(args['per_device_eval_batch_size'], 1)
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

    def test_optional_projector_lora_matches_only_selected_layers(self):
        cfg = self.grpo_config()
        from mmcultureqa.training.swift_backend import build_arguments
        import re
        language = ['model.language_model.layers.0.self_attn.q_proj',
                    'model.language_model.layers.2.self_attn.k_proj',
                    'model.language_model.layers.3.self_attn.v_proj',
                    'model.language_model.layers.4.self_attn.o_proj',
                    'model.language_model.layers.5.mlp.gate_proj',
                    'model.language_model.layers.6.mlp.up_proj',
                    'model.language_model.layers.7.mlp.down_proj']
        projector = ['model.visual.merger.linear_fc1', 'model.visual.merger.linear_fc2',
                     'model.visual.deepstack_merger_list.0.linear_fc1',
                     'model.visual.deepstack_merger_list.2.linear_fc2']
        for enabled in (False, True):
            cfg['peft']['train_projector'] = enabled
            args = build_arguments(cfg, 'sft')
            self.assertTrue(args['freeze_vit'])
            self.assertEqual(args['freeze_aligner'], not enabled)
            for name in language:
                self.assertIsNotNone(re.fullmatch(args['target_regex'], name), name)
            for name in projector:
                self.assertEqual(re.fullmatch(args['target_regex'], name) is not None, enabled, name)
            self.assertIsNone(re.fullmatch(args['target_regex'], 'model.visual.blocks.0.attn.qkv'))
            self.assertIsNone(re.fullmatch(args['target_regex'], 'model.language_model.embed_tokens'))
        with patch.dict(os.environ, WORLD_SIZE='1'):
            self.assertTrue(build_arguments(cfg, 'grpo')['freeze_aligner'])

    def test_merged_checkpoint_requires_all_shards_and_restores_saved_prompt(self):
        cfg = self.grpo_config()
        from mmcultureqa.models.artifacts import configure_merged_model, require_merged_model
        folder = self.project / 'outputs/downloaded/best_model'
        merged_fixture(folder)
        updated = configure_merged_model(cfg, folder)
        self.assertIsNone(updated['model']['adapter'])
        self.assertIsNone(updated['model']['revision'])
        self.assertTrue(updated['model']['local_files_only'])
        self.assertFalse(updated['model']['load_in_4bit'])
        self.assertEqual(updated['prompt_file'], str(folder / 'prompt.txt'))
        self.assertEqual(updated['system_prompt'], 'Saved cultural QA system prompt')
        self.assertEqual(cfg['model']['name'], 'Qwen/Qwen3-VL-4B-Instruct')
        (folder / 'model.safetensors.index.json').write_text(
            '{"weight_map": {"layer.weight": "model-00001-of-00002.safetensors"}}', encoding='utf-8')
        with self.assertRaisesRegex(FileNotFoundError, 'shard missing'):
            require_merged_model(folder)
        (folder / 'model.safetensors.index.json').unlink()
        (folder / 'config.json').write_text('{"quantization_config": {"load_in_4bit": true}}', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'not a quantized base'):
            require_merged_model(folder)

    def test_export_reloads_bf16_base_merges_selected_adapter_and_is_idempotent(self):
        cfg = self.grpo_config()
        from mmcultureqa.export import export_merged
        from mmcultureqa.models.artifacts import require_merged_model
        run = self.project / 'outputs/export-test'
        adapter = run / 'final'
        adapter.mkdir(parents=True)
        (adapter / 'adapter_config.json').write_text('{"r":16}', encoding='utf-8')
        (adapter / 'adapter_model.safetensors').write_bytes(b'selected-best-adapter')
        cfg['prompt_text'] = 'Frozen training prompt {question}'
        (run / 'run.json').write_text(json.dumps(cfg), encoding='utf-8')
        (run / 'selection.json').write_text(
            '{"best_model_checkpoint":"checkpoint-225", "best_metric":0.8, "metric_for_best_model":"eval_loss"}',
            encoding='utf-8')

        def save_weights(folder, **kwargs):
            self.assertTrue(kwargs['safe_serialization'])
            folder = Path(folder)
            folder.mkdir(parents=True)
            (folder / 'config.json').write_text('{"torch_dtype":"bfloat16"}', encoding='utf-8')
            (folder / 'model.safetensors').write_bytes(b'merged-bf16-model')

        def save_processor(folder):
            for name in ('tokenizer_config.json', 'preprocessor_config.json', 'tokenizer.json'):
                (Path(folder) / name).write_text('{}', encoding='utf-8')

        base = SimpleNamespace(is_loaded_in_4bit=False, is_loaded_in_8bit=False)
        merged = SimpleNamespace(config=SimpleNamespace(use_cache=False), to=MagicMock(),
                                 save_pretrained=MagicMock(side_effect=save_weights))
        adapter_model = SimpleNamespace(merge_and_unload=MagicMock(return_value=merged))
        base_factory = MagicMock(return_value=base)
        adapter_factory = MagicMock(return_value=adapter_model)
        processor_factory = MagicMock(return_value=SimpleNamespace(save_pretrained=save_processor))
        libraries = {
            'torch': SimpleNamespace(bfloat16='bf16', cuda=SimpleNamespace(is_available=lambda: True)),
            'transformers': SimpleNamespace(AutoModelForImageTextToText=SimpleNamespace(from_pretrained=base_factory),
                                            AutoProcessor=SimpleNamespace(from_pretrained=processor_factory)),
            'peft': SimpleNamespace(PeftModel=SimpleNamespace(from_pretrained=adapter_factory)),
        }
        with patch.dict(sys.modules, libraries):
            destination = export_merged(run)
            self.assertEqual(export_merged(run), destination)
        self.assertEqual(base_factory.call_count, 1)
        self.assertEqual(base_factory.call_args.kwargs['dtype'], 'bf16')
        self.assertNotIn('quantization_config', base_factory.call_args.kwargs)
        self.assertEqual(adapter_factory.call_args.args[1], str(adapter))
        adapter_model.merge_and_unload.assert_called_once_with(safe_merge=True)
        self.assertEqual(require_merged_model(destination), destination)
        report = json.loads((destination / 'merge.json').read_text(encoding='utf-8'))
        self.assertEqual(report['selection']['best_model_checkpoint'], 'checkpoint-225')
        self.assertEqual((destination / 'prompt.txt').read_text(encoding='utf-8'), cfg['prompt_text'])
        (adapter / 'adapter_model.safetensors').write_bytes(b'different-adapter')
        with self.assertRaisesRegex(FileExistsError, 'different adapter'):
            export_merged(run)
        checkpoint = run / 'checkpoint-125'
        checkpoint.mkdir()
        (checkpoint / 'adapter_config.json').write_text('{"r":16}')
        (checkpoint / 'adapter_model.safetensors').write_bytes(b'checkpoint-specific-weights')
        candidate = run / 'candidates/checkpoint-125/best_model'
        with patch.dict(sys.modules, libraries):
            self.assertEqual(export_merged(run, adapter_dir=checkpoint, output_dir=candidate), candidate)
        self.assertEqual(adapter_factory.call_args.args[1], str(checkpoint))
        self.assertEqual(processor_factory.call_args.args[0], str(run / 'final'))
        self.assertEqual(require_merged_model(candidate), candidate)


    def test_merged_model_loader_uses_local_bf16_without_adapter_or_hub_fallback(self):
        cfg = self.grpo_config()
        from mmcultureqa.models.artifacts import configure_merged_model
        from mmcultureqa.models.loading import load
        folder = self.project / 'outputs/local-model'
        merged_fixture(folder)
        cfg = configure_merged_model(cfg, folder)
        model_factory = MagicMock(return_value=SimpleNamespace(config=SimpleNamespace(use_cache=True)))
        processor_factory = MagicMock(return_value=SimpleNamespace(tokenizer=SimpleNamespace(padding_side='right')))
        adapter_factory = MagicMock()
        libraries = {
            'torch': SimpleNamespace(bfloat16='bf16', cuda=SimpleNamespace(
                is_available=lambda: True, is_bf16_supported=lambda: True)),
            'transformers': SimpleNamespace(AutoModelForImageTextToText=SimpleNamespace(from_pretrained=model_factory),
                                            AutoProcessor=SimpleNamespace(from_pretrained=processor_factory),
                                            BitsAndBytesConfig=MagicMock()),
            'peft': SimpleNamespace(LoraConfig=MagicMock(), PeftModel=adapter_factory, get_peft_model=MagicMock(),
                                    prepare_model_for_kbit_training=MagicMock()),
        }
        with patch.dict(sys.modules, libraries):
            load(cfg)
        for factory in (model_factory, processor_factory):
            self.assertEqual(factory.call_args.args[0], str(folder))
            self.assertTrue(factory.call_args.kwargs['local_files_only'])
            self.assertIsNone(factory.call_args.kwargs['revision'])
        self.assertNotIn('quantization_config', model_factory.call_args.kwargs)
        adapter_factory.from_pretrained.assert_not_called()

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
        merged_fixture(run / 'best_model')
        # A run downloaded from Drive can contain only the self-contained model.
        (run / 'best_model/run_config.json').write_text(json.dumps(cfg), encoding='utf-8')
        (run / 'run.json').unlink()
        with patch.object(sys, 'argv', ['evaluate', '--run-dir', str(run)]), patch('subprocess.run', side_effect=child):
            evaluate.main()
        self.assertFalse(inferred[1]['model']['load_in_4bit'])
        self.assertEqual(inferred[1]['inference']['backend'], 'vllm')
        self.assertIsNone(inferred[1]['model']['adapter'])
        self.assertTrue(inferred[1]['model']['local_files_only'])
        report = json.loads((run / 'validation/evaluation.json').read_text(encoding='utf-8'))
        self.assertEqual(report['model_artifact'], 'merged_bf16')
        with patch.object(sys, 'argv', ['evaluate', '--run-dir', str(run), '--prompt-mode', 'cot']), patch('subprocess.run', side_effect=child):
            evaluate.main()
        cot = inferred[-1]
        self.assertEqual(Path(cot['inference']['output']).parent.name, 'validation_cot')
        self.assertEqual(cot['inference']['max_new_tokens'], 1024)
        self.assertEqual(cot['inference']['on_format_error'], 'raise')
        self.assertEqual(cot['inference']['format_retries'], 1)
        self.assertIn('Final answer:', Path(cot['inference']['prompt_file']).read_text(encoding='utf-8'))
        cot_report = json.loads((run / 'validation_cot/evaluation.json').read_text(encoding='utf-8'))
        self.assertEqual(report['gold_sha256'], cot_report['gold_sha256'])
        self.assertEqual(report['model_signature'], cot_report['model_signature'])

    def test_teacher_generation_retries_resumes_and_never_prompts_validation(self):
        self.grpo_config()
        from mmcultureqa.common import config, read_jsonl, write_jsonl
        from mmcultureqa.data.teacher_reasoning import generate_dataset
        from mmcultureqa.data.distillation import load_reasoned_train
        with patch.object(sys, 'argv', ['teacher', '--config', str(self.project / 'configs/teacher_reasoning.yaml')]):
            cfg = config()
        train = read_jsonl(self.project / 'data/prepared/train.jsonl')
        train.append(dict(id='train-2', image='second.jpg', question='Second?', answer='second-gold'))
        write_jsonl(self.project / 'data/prepared/train.jsonl', train)
        validation = read_jsonl(self.project / 'data/prepared/validation.jsonl')
        prompts, calls = [], []
        good = 'Native thinking</think>' + json.dumps({'supported': True, 'steps': [
            'Locate the visible object at the image centre.',
            'Connect its visible form with relevant cultural knowledge.',
            'Use those observations to answer the question.'], 'issue': ''})

        def generate(requests, params, **kwargs):
            calls.append(requests)
            if len(calls) == 1:
                texts = [('Native thinking</think>{"supported":false,"issue":"unclear image"}', 'stop'),
                         ('Unfinished reasoning', 'length')]
            else:
                texts = [(good, 'stop'), ('Unfinished reasoning again', 'length')]
            return [SimpleNamespace(outputs=[SimpleNamespace(text=t, finish_reason=f)]) for t, f in texts]

        def template(messages, **kwargs):
            prompts.append(messages)
            return 'formatted image prompt'

        factory = MagicMock(return_value=SimpleNamespace(generate=generate))
        hub = MagicMock(return_value=SimpleNamespace(model_info=lambda *a, **k: SimpleNamespace(sha='pinned-commit')))
        modules = {'torch': SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True,
                        is_bf16_supported=lambda: True, get_device_capability=lambda: (8, 9))),
                   'huggingface_hub': SimpleNamespace(HfApi=hub),
                   'transformers': SimpleNamespace(AutoProcessor=SimpleNamespace(from_pretrained=MagicMock(
                        return_value=SimpleNamespace(apply_chat_template=template)))),
                   'vllm': SimpleNamespace(LLM=factory, SamplingParams=lambda **k: k)}
        with patch.dict(sys.modules, modules), patch('mmcultureqa.data.teacher_reasoning.image', return_value='PIL'), redirect_stdout(StringIO()):
            generate_dataset(cfg)
            # Simulate a torn final append from an interrupted process.
            folder = self.project / cfg['teacher']['output_dir']
            with (folder / 'attempts.jsonl').open('ab') as handle:
                handle.write(b'{"partial":')
            generate_dataset(cfg)
        self.assertEqual(factory.call_count, 1)
        self.assertEqual(hub.call_count, 1)
        self.assertEqual(factory.call_args.kwargs['revision'], 'pinned-commit')
        self.assertNotIn('quantization', factory.call_args.kwargs)  # use checkpoint's prequantized weights
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(prompts), 4)
        for messages in prompts:
            text = messages[-1]['content'][-1]['text']
            self.assertNotIn('val-gold', text)
            self.assertNotIn('Val?', text)
            self.assertTrue('train-gold' in text or 'second-gold' in text)
        accepted = read_jsonl(folder / 'train_reasoning.jsonl')
        self.assertEqual([r['id'] for r in accepted], ['train-1'])
        self.assertTrue(accepted[0]['reasoning'].startswith('1. Locate'))
        report = json.loads((folder / 'dataset.json').read_text(encoding='utf-8'))
        self.assertTrue(report['complete'])
        self.assertEqual(report['rejected'], 1)
        student = {'data': {'reasoning_dir': str(folder)}}
        with self.assertRaisesRegex(ValueError, 'coverage'):
            load_reasoned_train(student, train, validation)
        student['data']['reasoning_min_coverage'] = 0.5
        with redirect_stdout(StringIO()):
            joined = load_reasoned_train(student, train, validation)
        self.assertEqual(joined, accepted)
        with self.assertRaisesRegex(ValueError, 'split mismatch'):
            load_reasoned_train(student, train, [])
        cfg['system_prompt'] += ' changed'
        with self.assertRaisesRegex(ValueError, 'Resume teacher/prompt/data mismatch'):
            generate_dataset(cfg)

    def test_distilled_sft_has_no_gold_input_and_rejects_mixed_or_smoke_data(self):
        self.grpo_config()
        from mmcultureqa.common import config, read_jsonl, write_jsonl
        from mmcultureqa.data.distillation import digest, load_reasoned_train
        from mmcultureqa.training.swift_backend import build_arguments
        with patch.object(sys, 'argv', ['sft', '--config', str(self.project / 'configs/sft_reasoning.yaml')]):
            cfg = config()
        train = read_jsonl(self.project / 'data/prepared/train.jsonl')
        validation = read_jsonl(self.project / 'data/prepared/validation.jsonl')
        directory = self.project / cfg['data']['reasoning_dir']
        accepted = [{**train[0], 'reasoning': '1. Observe the object.\n2. Apply cultural knowledge.\n3. Derive the answer.'}]
        write_jsonl(directory / 'train_reasoning.jsonl', accepted)
        report = dict(schema_version=1, complete=True, smoke=False, accepted=1,
                      source_train_hash=digest(train), source_validation_hash=digest(validation), accepted_hash=digest(accepted))
        (directory / 'dataset.json').write_text(json.dumps(report), encoding='utf-8')
        with redirect_stdout(StringIO()):
            args = build_arguments(cfg, 'sft')
        self.assertEqual(args['per_device_train_batch_size'], 2)
        self.assertEqual(args['gradient_accumulation_steps'], 4)
        self.assertEqual(args['quant_bits'], 4)
        self.assertEqual(args['loss_scale'], 'default')
        records = read_jsonl(args['dataset'][0])
        self.assertNotIn('train-gold', records[0]['messages'][-2]['content'])
        self.assertIn('<think>\n1. Observe', records[0]['messages'][-1]['content'])
        self.assertTrue(records[0]['messages'][-1]['content'].endswith('<answer>train-gold</answer>'))
        val = read_jsonl(args['val_dataset'][0])
        self.assertEqual(val[0]['messages'][-1]['content'], '<answer>val-gold</answer>')
        report['smoke'] = True
        (directory / 'dataset.json').write_text(json.dumps(report), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'smoke'):
            load_reasoned_train(cfg, train, validation)
        report['smoke'] = False
        accepted[0]['id'] = 'val-1'
        report['accepted_hash'] = digest(accepted)
        write_jsonl(directory / 'train_reasoning.jsonl', accepted)
        (directory / 'dataset.json').write_text(json.dumps(report), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'non-training'):
            load_reasoned_train(cfg, train, validation)

    def test_reasoning_inference_restores_contract_and_scores_only_final_answer(self):
        self.grpo_config()
        from mmcultureqa.common import write_jsonl, read_jsonl
        from mmcultureqa.inference import predict
        model = self.project / 'outputs/student/best_model'
        merged_fixture(model)
        saved = json.loads((model / 'run_config.json').read_text(encoding='utf-8'))
        saved['reasoning'] = {'enabled': True, 'max_new_tokens': 1024, 'vllm_max_model_len': 4096}
        (model / 'run_config.json').write_text(json.dumps(saved), encoding='utf-8')
        write_jsonl(self.project / 'data/prepared/devtest.jsonl', [dict(id='test-1', image='test.jpg', question='Test?')])
        observed = []

        def engine(cfg):
            observed.append(cfg)
            return lambda rows: ['<think>1. Visible evidence.\n2. Cultural context.</think>\n<answer>Final only</answer>']

        argv = ['predict', '--config', str(self.project / 'configs/inference.yaml'), '--set', f'model.name={json.dumps(str(model))}',
                '--set', 'inference.dtype=float16']
        with patch.dict(sys.modules, {'torch': SimpleNamespace(), 'transformers': SimpleNamespace(set_seed=lambda s: None)}), \
                patch.object(sys, 'argv', argv), patch('mmcultureqa.models.vllm_loading.load_vllm', side_effect=engine), \
                patch('mmcultureqa.models.loading.load', side_effect=AssertionError('Unexpected Transformers model')), \
                patch('mmcultureqa.inference.predict.provenance'), redirect_stdout(StringIO()):
            predict.main()
            predict.main()  # complete output resumes without loading the engine again
        self.assertEqual(len(observed), 1)
        cfg = observed[0]
        self.assertEqual(cfg['model']['dtype'], 'float16')
        self.assertEqual(cfg['system_prompt'], 'Saved cultural QA system prompt')
        self.assertEqual(cfg['inference']['max_new_tokens'], 1024)
        self.assertEqual(cfg['inference']['vllm_max_model_len'], 4096)
        self.assertEqual(read_jsonl(cfg['inference']['output']), [{'id': 'test-1', 'answer': 'Final only'}])
        self.assertIn('Visible evidence', read_jsonl(Path(cfg['inference']['output']).with_suffix('.traces.jsonl'))[0]['generation'])

    def test_reasoning_parsers_reject_truncation_and_ambiguous_answers(self):
        sys.path.insert(0, str(self.project / 'src'))
        from mmcultureqa.data.distillation import parse_teacher, final_answer
        plain = 'Step 1: Identify the airline.\nStep 2: Interpret the branding.\n\nFinal answer: Kuwait Airways; its design represents national identity.'
        self.assertEqual(final_answer(plain, require_think=False), 'Kuwait Airways; its design represents national identity.')
        self.assertEqual(final_answer('Step 1: Reasoning.\n<answer>Complete answer.</answer>', require_think=False), 'Complete answer.')
        for invalid in ('Step 1: No conclusion.', 'Step 1: Reasoning.\nFinal answer:',
                        'Final answer: One.\nFinal answer: Two.'):
            with self.assertRaises(ValueError):
                final_answer(invalid, require_think=False)
        with self.assertRaises(ValueError):
            final_answer(plain)
        for text, finish in [('Thinking without closing tag', 'stop'), ('</think>{}', 'length'),
                             ('</think>{"supported":false}', 'stop'), ('</think>{"supported":true,"steps":[]}', 'stop')]:
            with self.assertRaises(ValueError):
                parse_teacher(text, finish)
        for text in ['Plain answer', '<think>unfinished <answer>guess</answer>',
                     '<think>steps</think><answer>a</answer><answer>b</answer>',
                     '<think>steps</think><answer></answer>']:
            with self.assertRaises(ValueError):
                final_answer(text)

    def test_two_models_share_bf16_sft_config(self):
        self.grpo_config()
        from mmcultureqa.common import config
        from mmcultureqa.training.swift_backend import build_arguments
        with patch.object(sys, 'argv', ['sft', '--config', str(self.project / 'configs/sft.yaml')]):
            cfg = config()
        for model in ('Qwen/Qwen3-VL-4B-Instruct', 'Qwen/Qwen3-VL-8B-Instruct'):
            cfg['model']['name'] = model
            args = build_arguments(cfg, 'sft')
            self.assertTrue(args['eval_on_start'])
            self.assertNotIn('quant_bits', args)
            self.assertEqual(args['metric_for_best_model'], 'loss')
            self.assertEqual(args['per_device_train_batch_size'] * args['gradient_accumulation_steps'], 8)

    def test_active_notebook_set_has_only_two_qwen_sft_models(self):
        active = {p.stem for p in (PROJECT / 'notebooks').glob('*.ipynb')}
        self.assertEqual(active, {'01_hf_to_drive_rclone', '02_sft_colab', '03_grpo_colab',
                                 '04_inference_colab', '05_sft_qwen3vl8b_colab'})
        for name in ('02_sft_colab', '05_sft_qwen3vl8b_colab'):
            ns, commands, _ = self.experiment(name)
            exec(find_cell(name, 'SFT_OVERRIDES ='), ns)
            self.assertTrue(commands[0][commands[0].index('--config') + 1].endswith('sft.yaml'))
            self.assertIn('model.name=\"Qwen/Qwen3-VL-', ' '.join(commands[0]))
        self.assertNotIn('RUNTIME_PROFILE', find_cell('04_inference_colab', 'INSTALL_VLLM ='))

    def test_periodic_vllm_validation_restores_training_and_returns_f1(self):
        self.grpo_config()
        from contextlib import contextmanager
        from mmcultureqa.common import config, write_jsonl
        from mmcultureqa.training.vllm_validation import install_validation
        with patch.object(sys, 'argv', ['sft', '--config', str(self.project / 'configs/sft.yaml')]):
            cfg = config()
        events = []
        trainer = SimpleNamespace(
            evaluate=lambda **kw: {'eval_loss': 0.8},
            save_model=lambda folder: (Path(folder).mkdir(), (Path(folder) / 'adapter_config.json').write_text('{}')),
            log=lambda value: events.append(('log', value)),
        )
        @contextmanager
        def swapped(_):
            events.append('offload')
            try:
                yield
            finally:
                events.append('restore')
        def generate(config):
            events.append('generate')
            child_cfg = json.loads(Path(config).read_text())
            self.assertFalse(child_cfg['model']['load_in_4bit'])
            self.assertFalse(child_cfg['inference']['do_sample'])
            self.assertEqual(child_cfg['model']['adapter_revision'], None)
            write_jsonl(child_cfg['inference']['output'], [{'id': 'val-1', 'answer': 'Answer'}])
        def child(cmd, **kw):
            self.assertEqual(events[0], 'offload')
            self.assertIn('mmcultureqa.score', cmd)
            events.append('score')
            Path(cmd[-1]).write_text(json.dumps({'result': [{'en': {'BERTScore-F1': 0.87}}]}))
        (self.project / cfg['training']['output_dir']).mkdir(parents=True, exist_ok=True)
        processor = SimpleNamespace(save_pretrained=lambda _: None)
        with patch('mmcultureqa.training.vllm_validation.training_on_cpu', swapped), \
             patch('mmcultureqa.training.vllm_validation.subprocess.run', side_effect=child), \
             patch('mmcultureqa.training.vllm_validation.PersistentValidationWorker') as worker:
            worker.return_value.predict.side_effect = generate
            controller = install_validation(trainer, cfg, processor)
            controller.preflight()
            metrics = trainer.evaluate()
        self.assertEqual(metrics, {'eval_loss': 0.8, 'eval_bertscore_f1': 0.87})
        self.assertEqual(events[:4], ['offload', 'generate', 'generate', 'restore'])
        self.assertEqual(events[4:8], ['offload', 'generate', 'score', 'restore'])

    def test_persistent_vllm_reuses_base_and_refreshes_lora_then_sleeps(self):
        self.grpo_config()
        from mmcultureqa.common import config, write_jsonl, read_jsonl
        from mmcultureqa.training.vllm_worker import ValidationSession
        with patch.object(sys, 'argv', ['sft', '--config', str(self.project / 'configs/sft.yaml')]):
            cfg = config()
        cfg['model']['load_in_4bit'] = False
        cfg['inference'] = dict(output=str(self.project / 'pred.jsonl'), batch_size=8,
                                vllm_cpu_offload_gb=0)
        events = []
        requests = []
        engine = MagicMock()
        engine.list_loras.return_value = {1}
        engine.wake_up.side_effect = lambda: events.append('wake')
        engine.remove_lora.side_effect = lambda i: events.append('unload')
        engine.sleep.side_effect = lambda **k: events.append('sleep')
        def generate(rows, adapter_request=None):
            events.append('generate')
            requests.append(adapter_request)
            return [Path(adapter_request.path).name for _ in rows]
        generate.engine = engine
        modules = {'swift': SimpleNamespace(), 'swift.llm': SimpleNamespace(
            AdapterRequest=lambda name, path: SimpleNamespace(name=name, path=path))}
        with patch.dict(sys.modules, modules), \
             patch('mmcultureqa.models.vllm_loading.load_vllm', return_value=generate) as factory:
            session = ValidationSession(cfg)
            for name in ('step125', 'step250'):
                adapter = self.project / name
                adapter.mkdir()
                (adapter / 'adapter_config.json').write_text(json.dumps({
                    'base_model_name_or_path': cfg['model']['name'], 'r': 16}))
                cfg['model']['adapter'] = str(adapter)
                session.predict(cfg)
                self.assertEqual(read_jsonl(cfg['inference']['output'])[0]['answer'], name)
            session.close()
        factory.assert_called_once()
        self.assertTrue(factory.call_args.kwargs['persistent'])
        self.assertNotEqual(requests[0].name, requests[1].name)
        self.assertEqual(events, ['generate', 'unload', 'sleep', 'wake', 'generate', 'unload', 'sleep'])
        engine.engine_core.shutdown.assert_called_once()

    def test_persistent_vllm_sleeps_on_generation_error_and_rejects_changed_base(self):
        self.grpo_config()
        from mmcultureqa.common import config
        from mmcultureqa.training.vllm_worker import ValidationSession
        with patch.object(sys, 'argv', ['sft', '--config', str(self.project / 'configs/sft.yaml')]):
            cfg = config()
        cfg['inference'] = dict(output=str(self.project / 'pred.jsonl'), batch_size=8)
        adapter = self.project / 'snapshot'
        adapter.mkdir()
        (adapter / 'adapter_config.json').write_text(json.dumps({
            'base_model_name_or_path': cfg['model']['name'], 'r': 16}))
        cfg['model']['adapter'] = str(adapter)
        generate = MagicMock(side_effect=RuntimeError('bad generation'))
        generate.engine.list_loras.return_value = {7}
        modules = {'swift': SimpleNamespace(), 'swift.llm': SimpleNamespace(
            AdapterRequest=lambda name, path: SimpleNamespace(name=name, path=path))}
        with patch.dict(sys.modules, modules), \
             patch('mmcultureqa.models.vllm_loading.load_vllm', return_value=generate):
            session = ValidationSession(cfg)
            with self.assertRaisesRegex(RuntimeError, 'bad generation'):
                session.predict(cfg)
            generate.engine.remove_lora.assert_called_once_with(7)
            generate.engine.sleep.assert_called_once_with(level=1)
            cfg['model']['revision'] = 'different'
            with self.assertRaisesRegex(ValueError, 'changed'):
                session.predict(cfg)
            session.close()

    def test_validation_worker_acknowledges_only_after_sleep_and_closes_on_stop(self):
        self.grpo_config()
        from mmcultureqa.training.vllm_worker import serve
        file = self.project / 'worker-config.json'
        file.write_text('{}')
        connection = MagicMock()
        connection.recv.side_effect = [dict(command='predict', config=str(file)),
                                       dict(command='predict', config=str(file)), dict(command='stop')]
        events = []
        with patch('mmcultureqa.training.vllm_worker.ValidationSession') as factory:
            factory.return_value.predict.side_effect = lambda _: events.append('generate-and-sleep')
            factory.return_value.close.side_effect = lambda: events.append('close')
            connection.send.side_effect = lambda _: events.append('ack')
            serve(connection)
            factory.assert_called_once()
        self.assertEqual(events, ['generate-and-sleep', 'ack', 'generate-and-sleep', 'ack', 'close'])
        connection.close.assert_called_once()

    def test_post_training_selection_merges_every_checkpoint_and_keeps_best_f1(self):
        self.grpo_config()
        from mmcultureqa.common import config
        from mmcultureqa.select_checkpoint import select_checkpoint
        with patch.object(sys, 'argv', ['sft', '--config', str(self.project / 'configs/sft.yaml')]):
            cfg = config()
        cfg['prompt_text'] = 'Question: {question}'
        run = self.project / 'outputs/selection-run'
        run.mkdir(parents=True)
        (run / 'run.json').write_text(json.dumps(cfg))
        for step in (125, 250, 375):
            cp = run / f'checkpoint-{step}'
            cp.mkdir()
            (cp / 'adapter_config.json').write_text('{}')
            (cp / 'adapter_model.safetensors').write_bytes(str(step).encode())
        events = []
        def process(cmd, **kwargs):
            module = cmd[3]
            folder = Path(cmd[cmd.index('--run-dir') + 1])
            if module == 'mmcultureqa.export':
                adapter = Path(cmd[cmd.index('--adapter') + 1])
                events.append(('merge', adapter.name))
                destination = Path(cmd[cmd.index('--output') + 1]) if '--output' in cmd else run / 'best_model'
                self.assertFalse(list((run / 'checkpoint_selection').glob('*/best_model')))
                merged_fixture(destination)
            elif module == 'mmcultureqa.evaluate':
                events.append(('score', folder.name))
                self.assertEqual(cmd[-2:], ['--backend', 'vllm'])
                report = folder / 'validation'
                report.mkdir()
                value = {'checkpoint-125': 0.81, 'checkpoint-250': 0.89, 'checkpoint-375': 0.85}[folder.name]
                (report / 'scores.json').write_text(json.dumps({'result': [{'en': {'BERTScore-F1': value}}]}))
                (report / 'evaluation.json').write_text(json.dumps(dict(full_validation=True,
                    evaluated_rows=1, model_artifact='merged_bf16')))
            else:
                self.fail(module)
        with patch('mmcultureqa.select_checkpoint.subprocess.run', side_effect=process):
            selection = select_checkpoint(run)
            self.assertEqual(Path(selection['best_model_checkpoint']).name, 'checkpoint-250')
            self.assertEqual(selection['best_metric'], 0.89)
            self.assertTrue((run / 'best_model/model.safetensors').is_file())
            self.assertFalse(list((run / 'checkpoint_selection').glob('*/best_model')))
            self.assertEqual(len(list(run.glob('checkpoint-*'))), 3)
            self.assertEqual(events, [('merge', 'checkpoint-125'), ('score', 'checkpoint-125'),
                ('merge', 'checkpoint-250'), ('score', 'checkpoint-250'),
                ('merge', 'checkpoint-375'), ('score', 'checkpoint-375'), ('merge', 'checkpoint-250')])
            events.clear()
            select_checkpoint(run)
            self.assertEqual(events, [('merge', 'checkpoint-250')])

    def test_cpu_swap_restores_devices_after_failed_generation(self):
        sys.path.insert(0, str(self.project / 'src'))
        from mmcultureqa.training.vllm_validation import training_on_cpu
        class Tensor:
            def __init__(self, device):
                self.data = SimpleNamespace(device=device, to=lambda target: payload(target))
                self.grad = None
            @property
            def device(self):
                return self.data.device
        def payload(device):
            if isinstance(device, str):
                device = SimpleNamespace(type=device)
            return SimpleNamespace(device=device, to=lambda target: payload(target))
        cuda = SimpleNamespace(type='cuda')
        parameter, gradient, state = Tensor(cuda), Tensor(cuda), Tensor(cuda)
        parameter.grad = gradient
        trainer = SimpleNamespace(model=SimpleNamespace(parameters=lambda: [parameter], buffers=lambda: []),
                                  optimizer=SimpleNamespace(state={0: {'exp_avg': state}}))
        torch = SimpleNamespace(is_tensor=lambda x: isinstance(x, Tensor), cuda=MagicMock())
        with patch.dict(sys.modules, torch=torch):
            with self.assertRaisesRegex(RuntimeError, 'child failed'):
                with training_on_cpu(trainer):
                    self.assertEqual(parameter.device.type, 'cpu')
                    self.assertEqual(state.device.type, 'cpu')
                    raise RuntimeError('child failed')
        for tensor in (parameter, gradient, state):
            self.assertEqual(tensor.device.type, 'cuda')

    def test_step_zero_val_loss_is_preserved_in_learning_curve(self):
        sys.path.insert(0, str(self.project / 'src'))
        from mmcultureqa.training.curves import load_points
        folder = self.project / 'outputs/curve'
        folder.mkdir(parents=True)
        (folder / 'logging.jsonl').write_text('\n'.join(json.dumps(row) for row in [
            {'eval_loss': 1.2, 'global_step/max_steps': '0/1125'},
            {'loss': 0.9, 'global_step/max_steps': '5/1125'},
            {'eval_loss': 0.8, 'global_step/max_steps': '225/1125'}]), encoding='utf-8')
        self.assertEqual(load_points(folder)['eval_loss'], [(0, 1.2), (225, 0.8)])

    def test_qwen35_artifact_restores_family_and_disables_thinking_for_vllm(self):
        cfg = self.grpo_config()
        from mmcultureqa.models.artifacts import configure_merged_model
        from mmcultureqa.models.vllm_loading import load_vllm
        folder = self.project / 'outputs/qwen35/best_model'
        merged_fixture(folder)
        saved = json.loads((folder / 'run_config.json').read_text(encoding='utf-8'))
        saved['model'].update(model_type='qwen3_5', enable_thinking=False)
        (folder / 'run_config.json').write_text(json.dumps(saved), encoding='utf-8')
        cfg = configure_merged_model(cfg, folder)
        cfg['inference'] = dict(vllm_max_model_len=2048, vllm_max_num_seqs=4, vllm_gpu_memory_utilization=0.8)
        template = SimpleNamespace(enable_thinking=True, response_prefix='<think>\n',
            template_meta=SimpleNamespace(thinking_prefix='<think>\n', non_thinking_prefix='<think>\n\n</think>\n\n'))
        factory = MagicMock(return_value=SimpleNamespace(template=template))
        modules = {'torch': SimpleNamespace(bfloat16='bf16', cuda=SimpleNamespace(
            is_available=lambda: True, is_bf16_supported=lambda: True)),
            'swift': SimpleNamespace(), 'swift.infer_engine': SimpleNamespace(VllmEngine=factory,
                InferRequest=lambda **k: k, RequestConfig=lambda **k: k)}
        with patch.dict(sys.modules, modules):
            load_vllm(cfg)
        self.assertEqual(factory.call_args.kwargs['model_type'], 'qwen3_5')
        self.assertEqual(factory.call_args.kwargs['torch_dtype'], 'bf16')
        self.assertFalse(template.enable_thinking)
        self.assertEqual(template.response_prefix, template.template_meta.non_thinking_prefix)

    def test_vllm_fp16_and_sampling_parameters_are_forwarded(self):
        cfg = self.grpo_config()
        from mmcultureqa.models.vllm_loading import load_vllm
        cfg['model'].update(dtype='float16', load_in_4bit=False, adapter=None)
        cfg['inference'] = dict(max_new_tokens=512, do_sample=True, temperature=0.7, top_p=0.8,
            top_k=20, presence_penalty=1.5, vllm_max_model_len=4096, vllm_max_num_seqs=4,
            vllm_gpu_memory_utilization=0.8)
        factory = MagicMock(return_value=SimpleNamespace(infer=lambda requests, config, **k: []))
        request_config = MagicMock(return_value={})
        modules = {'torch': SimpleNamespace(float16='fp16', cuda=SimpleNamespace(
            is_available=lambda: True, is_bf16_supported=lambda: False)),
            'swift': SimpleNamespace(), 'swift.llm': SimpleNamespace(VllmEngine=factory,
                InferRequest=lambda **k: k, RequestConfig=request_config)}
        with patch.dict(sys.modules, modules):
            generate = load_vllm(cfg)
            self.assertEqual(generate([]), [])
        self.assertEqual(factory.call_args.kwargs['torch_dtype'], 'fp16')
        self.assertEqual(request_config.call_args.kwargs['presence_penalty'], 1.5)
        self.assertEqual(request_config.call_args.kwargs['top_p'], 0.8)


    def test_upstream_qwen3_mapping_backport_excludes_vision_before_lora_profile(self):
        sys.path.insert(0, str(self.project / 'src'))
        class Mapping:
            @staticmethod
            def from_string_field(**kwargs):
                return SimpleNamespace(**{k: [v] for k, v in kwargs.items()})
        class Model:
            # Exact faulty metadata shipped in vLLM 0.11.0 (upstream PR #27027).
            def get_mm_mapping(self):
                return Mapping.from_string_field(language_model='language_model',
                    connector='model.visual.merger', tower_model='model.visual.')
        names = ['visual.blocks.0.attn.proj', 'visual.merger.linear_fc1',
                 'language_model.model.layers.0.self_attn.qkv_proj']
        class BaseWorker:
            def load_model(self):
                mapping = Model().get_mm_mapping()
                prefixes = mapping.connector + mapping.tower_model
                # v0.11.0 LoRAModelManager filters using startswith on these prefixes.
                modules = {n: object() for n in names if not any(n.startswith(p) for p in prefixes)}
                self.model_runner = SimpleNamespace(model=SimpleNamespace(
                    lora_manager=SimpleNamespace(modules=modules)))
        original = Model().get_mm_mapping()
        self.assertFalse(names[0].startswith(original.tower_model[0]))
        modules = {
            'vllm.v1.worker.gpu_worker': SimpleNamespace(Worker=BaseWorker),
            'vllm.model_executor.models.qwen3_vl': SimpleNamespace(Qwen3VLForConditionalGeneration=Model),
            'vllm.model_executor.models.module_mapping': SimpleNamespace(MultiModelKeys=Mapping),
        }
        with patch.dict(sys.modules, modules):
            from mmcultureqa.models.vllm_qwen3_worker import Qwen3LanguageLoRAWorker, audit_language_only_lora
            worker = Qwen3LanguageLoRAWorker()
            worker.vllm_config = SimpleNamespace(lora_config=object())
            worker.load_model()
            self.assertEqual(list(worker.model_runner.model.lora_manager.modules), [names[2]])
            audit_language_only_lora(worker.model_runner.model)
            worker.model_runner.model.lora_manager.modules[names[0]] = object()
            with self.assertRaisesRegex(RuntimeError, 'vision encoder'):
                audit_language_only_lora(worker.model_runner.model)

    def test_persistent_loader_enables_sleep_and_requires_explicit_lora(self):
        cfg = self.grpo_config()
        from mmcultureqa.models.vllm_loading import load_vllm
        cfg['model'].update(dtype='bfloat16', load_in_4bit=False, adapter=None)
        cfg['inference'] = dict(max_new_tokens=256, do_sample=False, vllm_max_model_len=2048,
            vllm_max_num_seqs=2, vllm_gpu_memory_utilization=0.9, vllm_cpu_offload_gb=0, vllm_enforce_eager=True)
        factory = MagicMock()
        factory.return_value.infer.return_value = []
        # Swift forwards explicit named arguments and expands engine_kwargs together.
        # Match that boundary rather than accepting all arguments with a permissive mock.
        def swift_constructor(model, *, enforce_eager=False, engine_kwargs=None, **kwargs):
            def prepare(*, enforce_eager=False, **kwargs):
                self.assertTrue(enforce_eager)
            prepare(enforce_eager=enforce_eager, **(engine_kwargs or {}))
            return factory.return_value
        factory.side_effect = swift_constructor
        modules = {'torch': SimpleNamespace(bfloat16='bf16', cuda=SimpleNamespace(
            is_available=lambda: True, is_bf16_supported=lambda: True)),
            'swift': SimpleNamespace(), 'swift.llm': SimpleNamespace(VllmEngine=factory,
                InferRequest=lambda **k: k, RequestConfig=lambda **k: k)}
        request = object()
        with patch.dict(sys.modules, modules):
            generate = load_vllm(cfg, persistent=True)
            with self.assertRaisesRegex(ValueError, 'explicit'):
                generate([])
            self.assertEqual(generate([], adapter_request=request), [])
        self.assertTrue(factory.call_args.kwargs['enforce_eager'])
        self.assertNotIn('enforce_eager', factory.call_args.kwargs['engine_kwargs'])
        self.assertTrue(factory.call_args.kwargs['enable_sleep_mode'])
        self.assertTrue(factory.call_args.kwargs['enable_lora'])
        self.assertEqual(factory.call_args.kwargs['adapters'], [])
        self.assertEqual(factory.call_args.kwargs['engine_kwargs']['cpu_offload_gb'], 0)
        self.assertIs(factory.return_value.infer.call_args.kwargs['adapter_request'], request)

    def test_cot_prompt_survives_model_restore_and_format_failures_count_as_empty(self):
        self.grpo_config()
        from mmcultureqa.common import write_jsonl, read_jsonl
        from mmcultureqa.inference import predict
        model = self.project / 'outputs/direct/best_model'
        merged_fixture(model)
        write_jsonl(self.project / 'data/prepared/validation.jsonl', [
            dict(id='v1', image='v1.jpg', question='One?', answer='gold one'),
            dict(id='v2', image='v2.jpg', question='Two?', answer='gold two')])
        observed = []

        def engine(cfg):
            observed.append(cfg)
            return lambda rows: ['<think>Relevant reasoning.</think><answer>Final only</answer>', '<think>Looping forever']

        argv = ['predict', '--config', str(self.project / 'configs/inference.yaml'),
                '--set', 'model.name=' + json.dumps(str(model)), '--set', 'inference.split=validation',
                '--set', 'inference.prompt_mode=cot', '--set', 'inference.on_format_error=empty',
                '--set', 'inference.prompt_file=' + json.dumps(str(self.project / 'prompts/qa_cot.txt'))]
        captured = StringIO()
        with patch.dict(sys.modules, {'torch': SimpleNamespace(), 'transformers': SimpleNamespace(set_seed=lambda s: None)}), \
                patch.object(sys, 'argv', argv), patch('mmcultureqa.models.vllm_loading.load_vllm', side_effect=engine), \
                patch('mmcultureqa.inference.predict.provenance'), redirect_stdout(captured):
            predict.main()
            predict.main()
        self.assertEqual(len(observed), 1)
        cfg = observed[0]
        self.assertTrue(cfg['reasoning']['enabled'])
        self.assertTrue(cfg['prompt_file'].endswith('qa_cot.txt'))
        self.assertEqual(read_jsonl(cfg['inference']['output']), [{'id': 'v1', 'answer': 'Final only'}, {'id': 'v2', 'answer': ''}])
        traces = read_jsonl(Path(cfg['inference']['output']).with_suffix('.traces.jsonl'))
        self.assertIn('format_error', traces[1])
        self.assertIn('Validation:', captured.getvalue())
        self.assertIn('100%', captured.getvalue())
        retry_calls = []

        def retry_engine(runtime):
            def generate(batch):
                retry_calls.append((len(batch), runtime['inference']['do_sample'], runtime['training']['seed']))
                if len(batch) == 1:
                    return ['<think>Corrected reasoning.</think><answer>Recovered</answer>']
                return ['<think>Reasoning.</think><answer>First</answer>', '<think>Unfinished']
            return generate

        retry_output = self.project / 'outputs/retry.jsonl'
        retry_argv = argv + ['--set', 'inference.format_retries=1', '--set', 'inference.output=' + json.dumps(str(retry_output))]
        with patch.dict(sys.modules, {'torch': SimpleNamespace(), 'transformers': SimpleNamespace(set_seed=lambda s: None)}), \
                patch.object(sys, 'argv', retry_argv), patch('mmcultureqa.models.vllm_loading.load_vllm', side_effect=retry_engine), \
                patch('mmcultureqa.inference.predict.provenance'), redirect_stdout(StringIO()):
            predict.main()
        self.assertEqual(retry_calls, [(2, False, 42), (1, True, 43)])
        self.assertEqual(read_jsonl(retry_output)[1]['answer'], 'Recovered')
        self.assertEqual(len(read_jsonl(retry_output.with_suffix('.traces.jsonl'))), 3)
        failure_output = self.project / 'outputs/retry_failed.jsonl'
        failure_argv = retry_argv + ['--set', 'inference.output=' + json.dumps(str(failure_output))]
        with patch.dict(sys.modules, {'torch': SimpleNamespace(), 'transformers': SimpleNamespace(set_seed=lambda s: None)}), \
                patch.object(sys, 'argv', failure_argv), \
                patch('mmcultureqa.models.vllm_loading.load_vllm', return_value=lambda batch: ['<think>Unfinished'] * len(batch)), \
                patch('mmcultureqa.inference.predict.provenance'), redirect_stdout(StringIO()):
            with self.assertRaisesRegex(ValueError, 'after 2 attempts'):
                predict.main()
        self.assertEqual(read_jsonl(failure_output), [])
        self.assertEqual(len(read_jsonl(failure_output.with_suffix('.traces.jsonl'))), 2)

    def test_comparison_scales_metrics_and_rejects_mismatched_evaluations(self):
        sys.path.insert(0, str(self.project / 'src'))
        from mmcultureqa.inference.comparison import build_comparison, METRICS
        import hashlib
        run = self.project / 'outputs/compare'
        for index, name in enumerate(('validation', 'validation_cot')):
            folder = run / name
            folder.mkdir(parents=True)
            pred = folder / 'predictions.jsonl'
            pred.write_text('{"id":"v1","answer":"x"}\n', encoding='utf-8')
            report = dict(prompt_mode='cot' if index else 'saved', reasoning_enabled=bool(index),
                          model_signature='same-checkpoint', gold_sha256='same-gold', evaluated_rows=1,
                          scorer_dtype='float16', seed=42, load_in_4bit=False, empty_answers=0,
                          predictions_sha256=hashlib.sha256(pred.read_bytes()).hexdigest(),
                          generation={'backend': 'vllm', 'do_sample': False, 'max_new_tokens': 1024 if index else 256})
            (folder / 'evaluation.json').write_text(json.dumps(report), encoding='utf-8')
            metrics = {key: (0.75 if index else 0.70) for key in METRICS}
            metrics.update(BLEU=20 if index else 15, Coverage=1.)
            (folder / 'scores.json').write_text(json.dumps({'submission_result': metrics}), encoding='utf-8')
        output = build_comparison(run)
        self.assertIn('Zero-shot vs Zero-shot CoT', output.read_text(encoding='utf-8'))
        data = json.loads((run / 'prompt_comparison.json').read_text(encoding='utf-8'))
        self.assertEqual(data['rows'][0]['zero_shot'], 70.)
        self.assertEqual(data['rows'][0]['delta_points'], 5.)
        self.assertEqual(data['rows'][3]['zero_shot'], 15.)
        self.assertEqual(data['rows'][3]['delta_points'], 5.)
        self.assertTrue((run / 'prompt_comparison.csv').is_file())
        file = run / 'validation_cot/evaluation.json'
        report['gold_sha256'] = 'different-gold'
        file.write_text(json.dumps(report), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'gold_sha256'):
            build_comparison(run)


    def test_active_sft_notebooks_select_merged_checkpoints_after_training(self):
        for notebook in ('02_sft_colab', '05_sft_qwen3vl8b_colab'):
            ns, commands, _ = self.experiment(notebook)
            exec(find_cell(notebook, 'SFT_OVERRIDES ='), ns)
            self.assertNotIn('mmcultureqa.export', find_cell(notebook, 'SFT_OVERRIDES ='))
            exec(find_cell(notebook, 'mmcultureqa.select_checkpoint'), ns)
            self.assertEqual(commands[-1][2], 'mmcultureqa.select_checkpoint')


    def test_teacher_and_student_smoke_use_100_seed42_training_rows(self):
        self.grpo_config()
        from mmcultureqa.common import config, read_jsonl, write_jsonl
        from mmcultureqa.data.distillation import digest, sample_rows, load_reasoned_train
        from mmcultureqa.data.teacher_reasoning import generate_dataset
        with patch.object(sys, 'argv', ['teacher', '--config', str(self.project / 'configs/teacher_reasoning.yaml')]):
            cfg = config()
        train = [dict(id=f'train-{i}', question=f'Question {i}?', answer=f'Gold {i}', image=f'image-{i}.jpg') for i in range(160)]
        write_jsonl(self.project / 'data/prepared/train.jsonl', train)
        validation = read_jsonl(self.project / 'data/prepared/validation.jsonl')
        before = random.getstate()
        selected = sample_rows(train, 100, 42)
        self.assertEqual(selected, random.Random(42).sample(train, 100))
        self.assertNotEqual(selected, train[:100])
        self.assertEqual(len({r['id'] for r in selected}), 100)
        self.assertEqual(random.getstate(), before)
        cfg['teacher']['limit'] = 100
        folder = self.project / cfg['teacher']['output_dir']
        folder.mkdir(parents=True)
        spec = {'teacher': cfg['teacher'], 'system_prompt': cfg['system_prompt'],
                'prompt': (self.project / cfg['prompt_file']).read_text(encoding='utf-8'),
                'pixels': {k: cfg['model'][k] for k in ('min_pixels', 'max_pixels')},
                'source_train_hash': digest(train), 'source_validation_hash': digest(validation),
                'sampling': 'random_without_replacement'}
        (folder / 'generation.json').write_text(json.dumps({'fingerprint': digest(spec), 'resolved_revision': 'fixed'}), encoding='utf-8')
        write_jsonl(folder / 'attempts.jsonl', [{'id': r['id'], 'reasoning': '1. Observe.\n2. Connect.\n3. Conclude.', 'attempts': []} for r in selected])
        # A completed smoke journal exercises selection, publication and resume without GPU allocation.
        with redirect_stdout(StringIO()):
            generate_dataset(cfg)
        report = json.loads((folder / 'dataset.json').read_text(encoding='utf-8'))
        self.assertTrue(report['smoke'])
        self.assertEqual(report['selected_ids'], [r['id'] for r in selected])
        student = {'data': {'reasoning_dir': str(folder), 'reasoning_smoke': True}}
        with redirect_stdout(StringIO()):
            rows = load_reasoned_train(student, train, validation)
            self.assertEqual(rows, load_reasoned_train(student, train, validation))
        self.assertEqual(len(rows), 100)
        self.assertEqual({r['id'] for r in rows}, {r['id'] for r in selected})
        self.assertEqual(random.getstate(), before)
        student['data']['reasoning_smoke'] = False
        with self.assertRaisesRegex(ValueError, 'smoke'):
            load_reasoned_train(student, train, validation)



if __name__ == '__main__':
    unittest.main()
