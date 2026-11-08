"""CPU-only checks for OpenJev data, decision decoding, persistence, and notebook setup."""
import copy
import csv
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('openjev_run', ROOT / 'openjev/run.py')
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)


def make_config(folder):
    config = dict(samples=False, output=str(folder / 'results'))
    for split, target in [('train', 'group_t'), ('dev', 'individual_l,g'), ('test', 'group_a')]:
        path = folder / (split + '.tsv')
        with path.open('w', encoding='utf-8', newline='') as f:
            fields = ['StereoQueerEval_id', 'yt_title', 'yt_description', 'yt_comment', 'target']
            writer = csv.DictWriter(f, fieldnames=fields, delimiter='\t')
            writer.writeheader()
            writer.writerow(dict(zip(fields, [split + '_001', 'Shared video', 'Shared description',
                                                split + ' comment\nwith a quoted newline', target])))
        config[split] = str(path)
    return config


class OpenJevDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = run.load_samples()
        cls.questions = json.loads((ROOT / 'openjev/questions.json').read_text(encoding='utf-8'))

    def test_synthetic_coverage_and_provenance(self):
        for split in ('train', 'dev'):
            rows = [r for r in self.rows if r['split'] == split]
            self.assertGreaterEqual(sum(r['target'] != 'none' for r in rows), 4)
            self.assertGreaterEqual(sum(r['target'] == 'none' for r in rows), 2)
            self.assertTrue(all(r['origin'] == 'synthetic' and not r['human_reviewed'] for r in rows))
        texts = [' '.join(r['yt_comment'].casefold().split()) for r in self.rows]
        self.assertEqual(len(texts), len(set(texts)))

    def test_binary_and_multilabel_supervision(self):
        row = {**self.rows[0], 'target': 'individual_l,g'}
        tasks = list(run.decisions([row], self.questions))
        identities = [t for t in tasks if t['key'] != 'scope']
        self.assertEqual(len(identities), 9)
        self.assertEqual([t['key'] for t in identities if t['answer'] == 0], ['l', 'g'])
        self.assertEqual(tasks[-1]['answer'], 0)
        a_tasks = list(run.decisions([self.rows[0]], self.questions))
        self.assertEqual([t['key'] for t in a_tasks[:-1] if t['answer'] == 0], ['a'])

    def test_none_scope_is_never_supervised(self):
        row = next(r for r in self.rows if r['target'] == 'none')
        tasks = list(run.decisions([row], self.questions))
        self.assertEqual(len(tasks), 9)
        self.assertTrue(all(t['answer'] == 1 for t in tasks))
        inference = list(run.decisions([row], self.questions, inference=True))
        self.assertEqual(len(inference), 10)
        self.assertFalse(inference[-1]['supervised'])

    def test_gold_is_not_in_prompt_and_option_reversal_is_consistent(self):
        task = next(run.decisions(self.rows, self.questions))
        normal = run.prompt(task, self.questions)
        changed = copy.deepcopy(task)
        changed.update(answer=1-task['answer'], origin='other', source_id='PRIVATE_GOLD_ID')
        self.assertEqual(normal, run.prompt(changed, self.questions))
        self.assertNotIn('PRIVATE_GOLD_ID', normal)
        self.assertIn('[A] yes:', normal)
        self.assertIn('[A] no:', run.prompt(task, self.questions, reverse=True))
        self.assertEqual(run.order_for(42, 0, task), run.order_for(42, 0, task))

    def test_prepare_keeps_original_dev_and_does_not_write_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            config = make_config(folder)
            config['samples'] = True
            before = {p.name: p.read_bytes() for p in folder.iterdir()}
            _, groups, _, _, summary = run.prepare(config)
            self.assertEqual(len(groups['train']), 7)
            self.assertEqual(len(groups['dev']), 1)
            self.assertEqual(len(groups['dev_synthetic']), 6)
            self.assertEqual(len(groups['test']), 1)
            self.assertEqual(summary['shared_videos']['dev/train'], 1)
            self.assertEqual(before, {p.name: p.read_bytes() for p in folder.iterdir()})
            with self.assertRaises(ValueError):
                run.prepare({**config, 'dev': config['train']})
            with self.assertRaises(ValueError):
                run.prepare({**config, 'epohs': 3})

    def test_optional_test_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = make_config(Path(tmp))
            config['test'] = ''
            _, groups, _, _, _ = run.prepare(config)
            self.assertNotIn('test', groups)

    def test_threshold_uses_only_real_dev_and_retains_abstentions(self):
        row = {**self.rows[0], 'target': 'group_t', 'origin': 'original', 'split': 'dev'}
        p = {k: 0.01 for k in run.IDENTITIES}
        p.update(t=0.35, scope=0.1)
        threshold = run.choose_threshold([row], [p])
        self.assertLessEqual(threshold, 0.35)
        self.assertEqual(run.decode(p, threshold), ('group_t', 'predicted'))
        result = run.report([row], [p], 0.5)
        self.assertEqual(result['n'], 1)
        self.assertEqual(result['invalid_count'], 1)
        self.assertEqual(result['abstain_count'], 1)
        with self.assertRaises(ValueError):
            run.choose_threshold([{**row, 'origin': 'synthetic'}], [p])
        with self.assertRaises(ValueError):
            run.choose_threshold([{**row, 'split': 'test'}], [p])

    def test_training_identity_does_not_depend_on_optional_test(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            config = make_config(Path(tmp))
            job = run.Run(config)
            job.revision = 'pinned-weights'
            with patch('importlib.metadata.version', return_value='test-version'):
                identity = job._identity()
                config['test'] = ''
                other = run.Run(config)
                other.revision = 'pinned-weights'
                self.assertEqual(identity, other._identity())
                config['rank'] = 16
                changed = run.Run(config)
                changed.revision = 'pinned-weights'
                self.assertNotEqual(identity, changed._identity())

    def test_atomic_checkpoint_contains_optimizer_progress_and_best_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = object.__new__(run.Run)
            job.output = Path(tmp)
            job._weights = lambda: {'lora': [1, 2]}
            job.torch = SimpleNamespace(save=lambda value, path: Path(path).write_text(json.dumps(value)),
                                         get_rng_state=lambda: 'rng',
                                         cuda=SimpleNamespace(get_rng_state_all=lambda: ['cuda-rng']))
            optimizer = SimpleNamespace(state_dict=lambda: {'step': 3})
            job._checkpoint(optimizer, {'id': 1}, {'offset': 16}, {'weights': [0]}, [{'epoch': 1}])
            state = json.loads((Path(tmp) / 'checkpoint/state.pt').read_text())
            self.assertEqual(state['progress']['offset'], 16)
            self.assertEqual(state['optimizer']['step'], 3)
            self.assertEqual(state['weights'], {'lora': [1, 2]})
            self.assertEqual(state['best']['weights'], [0])
            self.assertFalse((Path(tmp) / 'checkpoint/state.tmp').exists())

    def test_notebook_compiles_and_clone_uses_main_under_content(self):
        notebook = json.loads((ROOT / 'openjev/train.ipynb').read_text(encoding='utf-8'))
        code = []
        for i, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'code':
                text = ''.join(cell['source'])
                compile(text, f'openjev-cell-{i}', 'exec')
                self.assertIsNone(cell['execution_count'])
                self.assertEqual(cell['outputs'], [])
                code.append(text)
        self.assertNotIn('drive.mount', '\n'.join(code))
        self.assertIn('BRANCH = "main"', code[0])
        self.assertIn('Path("/content")', code[0])
        calls = []
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            source = code[0].replace('Path("/content")', f'Path({str(folder)!r})')
            source = source.replace('Path("/kaggle/working")', f'Path({str(folder / "absent")!r})')
            def execute(args, **kwargs):
                calls.append(args)
                if args[1] == 'clone':
                    destination = Path(args[-1]) / 'task 8/openjev'
                    destination.mkdir(parents=True)
                    (destination / 'run.py').touch()
                return SimpleNamespace(returncode=0)
            with patch.object(subprocess, 'run', side_effect=execute), \
                    patch.object(subprocess, 'check_output', side_effect=['main\n', '']), \
                    redirect_stdout(io.StringIO()):
                exec(source, {})
                exec(source, {})
        clone = next(c for c in calls if c[1] == 'clone')
        self.assertEqual(clone[clone.index('--branch')+1], 'main')
        self.assertTrue(any('pull' in c and '--ff-only' in c for c in calls))

    def test_zero_shot_loads_only_the_base_model(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            config = make_config(Path(tmp))
            job = run.ZeroShot(config['test'], output=config['output'])
            base = SimpleNamespace(requires_grad_=unittest.mock.Mock(), eval=unittest.mock.Mock())
            with patch.object(job, '_base', return_value=base) as loader:
                job._load()
                job._load()
            loader.assert_called_once()
            base.requires_grad_.assert_called_once_with(False)
            base.eval.assert_called_once()
            self.assertNotIn('train', job.groups)
            self.assertNotIn('dev_synthetic', job.groups)
            with self.assertRaises(RuntimeError):
                job.train()

    def test_zero_shot_fixed_threshold_outputs_and_unlabeled_input(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            folder = Path(tmp)
            config = make_config(folder)
            job = run.ZeroShot(config['test'], output=config['output'])
            job.revision = 'published-checkpoint'
            scores = {k: 0.01 for k in run.IDENTITIES}
            scores.update(a=0.9, scope=0.1)
            with patch.object(job, '_load'), patch.object(job, '_predict', return_value=[scores]), \
                    patch.object(run, 'choose_threshold', side_effect=AssertionError('Must not tune')), \
                    patch('importlib.metadata.version', return_value='test-version'):
                metrics = job.infer()
            settings = json.loads((Path(config['output']) / 'settings.json').read_text())
            self.assertEqual(settings['mode'], 'zero_shot')
            self.assertEqual(settings['threshold'], 0.5)
            self.assertFalse(settings['training'])
            self.assertFalse(settings['adapters'])
            self.assertEqual(metrics['data']['full_target_exact_match'], 1)
            self.assertEqual({p.name for p in Path(config['output']).iterdir()},
                             {'settings.json', 'predictions.csv', 'metrics.json'})
            with (Path(config['output']) / 'predictions.csv').open(newline='') as f:
                self.assertEqual(next(csv.DictReader(f))['target'], 'group_a')
            # No gold is needed to construct tasks, build prompts or save scores.
            with Path(config['test']).open(newline='') as f:
                rows = list(csv.DictReader(f, delimiter='\t'))
            fields = [k for k in rows[0] if k != 'target']
            unlabeled = folder / 'unlabeled.tsv'
            with unlabeled.open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fields, delimiter='\t', extrasaction='ignore')
                writer.writeheader()
                writer.writerows(rows)
            other = run.ZeroShot(str(unlabeled), output=str(folder / 'unlabeled-results'))
            self.assertEqual(len(other.tasks['data']), 10)
            report = run.report(other.groups['data'], [scores], 0.5)
            self.assertEqual(report['status'], 'not_evaluated')
            self.assertEqual(report['coverage'], 1)

    def test_zero_shot_dev_threshold_is_explicit_and_disjoint(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            config = make_config(Path(tmp))
            with self.assertRaises(ValueError):
                run.ZeroShot(config['dev'], dev=config['dev'])
            job = run.ZeroShot(config['test'], output=config['output'], dev=config['dev'])
            job.revision = 'published-checkpoint'
            a = {k: 0.01 for k in run.IDENTITIES}
            a.update(a=0.35, scope=0.1)
            lg = {k: 0.01 for k in run.IDENTITIES}
            lg.update(l=0.35, g=0.35, scope=0.9)
            with patch.object(job, '_load'), \
                    patch.object(job, '_predict', side_effect=lambda split: [lg if split == 'dev' else a]) as predictor, \
                    patch('importlib.metadata.version', return_value='test-version'):
                metrics = job.infer(tune_on_dev=True)
            self.assertEqual([c.args[0] for c in predictor.call_args_list], ['dev', 'data'])
            self.assertEqual(metrics['protocol']['mode'], 'zero_shot_dev_threshold')
            self.assertLessEqual(metrics['protocol']['threshold'], 0.35)
            self.assertEqual(metrics['data']['full_target_exact_match'], 1)

    def test_inference_notebook_has_no_training_or_mount(self):
        notebook = json.loads((ROOT / 'openjev/inference.ipynb').read_text(encoding='utf-8'))
        codes = []
        for i, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                compile(source, f'inference-cell-{i}', 'exec')
                self.assertIsNone(cell['execution_count'])
                self.assertEqual(cell['outputs'], [])
                codes.append(source)
        text = '\n'.join(codes)
        self.assertIn('BRANCH = "main"', text)
        self.assertIn('run.ZeroShot(', text)
        self.assertIn('THRESHOLD = 0.5', text)
        self.assertIn('TUNE_ON_DEV = False', text)
        self.assertNotIn('job.train(', text)
        self.assertNotIn('drive.mount', text)
        self.assertNotIn('TRAIN =', text)


if __name__ == '__main__':
    unittest.main()
