"""CPU regression checks for Colab paths/imports; no model downloads or GPU needed.

Run: python -m unittest discover -s "task 6/tests" -v
"""
import ast
import json
import os
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

    def experiment(self):
        commands, copies = [], []

        def adapter(folder):
            folder.mkdir(parents=True, exist_ok=True)
            (folder / 'adapter_config.json').write_text('{}', encoding='utf-8')
            (folder / 'adapter_model.safetensors').write_bytes(b'fixture')

        def copy(source, destination, *args):
            copies.append((str(source), str(destination)))
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
        exec(find_cell('02_sft_colab', 'USE_4BIT ='), ns)
        os.chdir(self.content)
        return ns, commands, copies

    def test_training_uses_absolute_configs_outputs_and_adapter(self):
        ns, commands, copies = self.experiment()
        exec(find_cell('02_sft_colab', 'SFT_OVERRIDES ='), ns)
        exec(find_cell('03_grpo_colab', 'SFT_RUN_NAME ='), ns)
        self.assertEqual([c[2] for c in commands], ['mmcultureqa.training.sft', 'mmcultureqa.training.grpo'])
        self.assertTrue((Path(ns['GRPO']) / 'final/adapter_config.json').is_file())
        self.assertTrue(all(Path(source).is_absolute() for source, dest in copies if dest.startswith('remote:')))

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

    def test_incomplete_adapter_fails_before_gpu_command(self):
        ns, commands, _ = self.experiment()
        folder = self.project / 'outputs/broken'
        folder.mkdir(parents=True)
        (folder / 'adapter_config.json').write_text('{}', encoding='utf-8')
        with self.assertRaisesRegex(FileNotFoundError, 'weights missing'):
            ns['require_adapter'](folder)
        self.assertFalse(commands)


if __name__ == '__main__':
    unittest.main()
