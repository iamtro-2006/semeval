"""Regression checks for compact output, message isolation, and notebook syntax."""
import copy
import csv
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from inference import (INPUT_VARIANTS, evaluate, extract_target, load_prompt,
                       make_messages, validate_annotation, validate_demonstrations)
from inference import main


class CompactOutputTests(unittest.TestCase):
    def test_valid_labels_and_reasons(self):
        for target in ('none', 'individual_t', 'group_l,g', 'group_t,nb'):
            with self.subTest(target=target):
                raw = json.dumps({'target': target, 'reasoning': 'A supported explanation.'})
                parsed, annotation, error = validate_annotation(raw)
                self.assertEqual((parsed, error), (target, ''))
                self.assertEqual(annotation['reasoning'], 'A supported explanation.')
                self.assertEqual(extract_target(raw), (target, ''))

    def test_schema_rejects_bad_output(self):
        cases = [
            {'target': 'none'}, {'target': 'none', 'reasoning': ''},
            {'target': 'none', 'reasoning': None},
            {'target': 'none', 'reasoning': 'Reason.', 'scope': 'none'},
            {'target': 'group_g,l', 'reasoning': 'Reason.'},
            {'target': 'group_l,l', 'reasoning': 'Reason.'},
            {'target': 'group_unknown', 'reasoning': 'Reason.'},
            {'target': None, 'reasoning': 'Reason.'},
            ['none', 'Reason.'],
        ]
        for value in cases:
            with self.subTest(value=value):
                self.assertTrue(validate_annotation(json.dumps(value))[2])
        self.assertTrue(validate_annotation('{"target":"none","target":"group_t","reasoning":"R"}')[2])

    def test_target_scoring_is_independent_of_reason_format(self):
        raw = '{"target":"individual_t","reasoning":""}'
        self.assertTrue(validate_annotation(raw)[2])
        target, error = extract_target(raw)
        self.assertEqual((target, error), ('individual_t', ''))
        self.assertEqual(evaluate([{'target': target}], [target])['full_target_exact_match'], 1)

    def test_preserves_malformed_json_recovery(self):
        raw = '{"target":"group_t","reasoning":"unfinished'
        self.assertEqual(extract_target(raw), ('group_t', ''))
        self.assertTrue(validate_annotation(raw)[2])
        self.assertTrue(extract_target('{"target":"group_t", "target":"group_g",')[1])
        self.assertTrue(extract_target('{"reasoning":"unfinished')[1])

    def test_all_prompt_modes_filter_input_and_use_compact_demos(self):
        row = dict(yt_comment='Comment', yt_title='Title', yt_description='Description',
                   target='group_t', hate_speech='yes_explicit', StereoQueerEval_id='private')
        for variant, fields in INPUT_VARIANTS.items():
            for mode in ('zero_shot', 'few_shot'):
                with self.subTest(variant=variant, mode=mode):
                    name = 'fs_prompt.md' if mode == 'few_shot' else 'zs_prompt.md'
                    blocks, examples = load_prompt(ROOT / 'src/prompts' / variant / name, mode)
                    for example in examples:
                        self.assertEqual(set(example['input']), set(fields))
                        self.assertEqual(set(example['output']), {'target', 'reasoning'})
                        self.assertFalse(validate_annotation(json.dumps(example['output']))[2])
                    messages = make_messages(blocks, examples, row, 'English', variant)
                    payload = json.loads(messages[-1]['content'].split('\n\n', 1)[1])
                    self.assertEqual(payload, {k: row[k] for k in fields})
                    self.assertEqual(len(messages), 2 + 2 * len(examples))

    def test_demo_provenance_and_overlap_checks(self):
        train = dict(StereoQueerEval_id='train1', yt_comment='Comment', yt_title='Train title',
                     yt_description='Train description', target='group_t')
        evaluation = dict(StereoQueerEval_id='dev1', yt_comment='Other', yt_title='Dev title',
                          yt_description='Dev description', target='none')
        example = {'source_id': 'train1', 'source_split': 'train',
                   'input': {'yt_comment': 'Comment'},
                   'output': {'target': 'group_t', 'reasoning': 'Reason.'}}
        validate_demonstrations([example], [train], [evaluation], 'comment')
        for key, value in [('target', 'group_g'), ('reasoning', '')]:
            bad = copy.deepcopy(example)
            bad['output'][key] = value
            with self.assertRaises(ValueError):
                validate_demonstrations([bad], [train], [evaluation], 'comment')
        with self.assertRaises(ValueError):
            validate_demonstrations([example], [train], [train], 'comment')

    def test_notebook_code_compiles_and_has_no_saved_output(self):
        notebook = json.loads((ROOT / 'notebooks/inference.ipynb').read_text(encoding='utf-8'))
        for i, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'code':
                self.assertIsNone(cell['execution_count'])
                self.assertEqual(cell['outputs'], [])
                compile(''.join(cell['source']), f'notebook-cell-{i}', 'exec')

    def test_pipeline_writes_compact_predictions_and_metrics(self):
        """Exercise the CLI/output path with a fake model; no GPU or downloads."""
        from contextlib import nullcontext, redirect_stdout
        import io

        responses = iter(['{"target":"group_t","reasoning":"Targets trans people."}',
                          '{"target":"none","reasoning":""}'])

        class Inputs(dict):
            def to(self, device):
                return self

        class Generated:
            def __getitem__(self, key):
                return [1, 2]

        class Progress:
            def __init__(self, rows, **kwargs):
                self.rows = rows
            def __enter__(self):
                return iter(self.rows)
            def __exit__(self, *args):
                pass

        tokenizer = SimpleNamespace(
            pad_token_id=0,
            apply_chat_template=lambda *a, **kw: Inputs(input_ids=SimpleNamespace(shape=(1, 10))),
            decode=lambda *a, **kw: next(responses))
        model = SimpleNamespace(
            config=SimpleNamespace(max_position_embeddings=32768), eval=lambda: None,
            get_input_embeddings=lambda: SimpleNamespace(weight=SimpleNamespace(device='cpu')),
            generate=lambda **kw: Generated())
        fake_torch = SimpleNamespace(manual_seed=lambda seed: None,
                                     cuda=SimpleNamespace(is_available=lambda: False),
                                     inference_mode=nullcontext)
        fake_transformers = SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: tokenizer),
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *a, **kw: model))
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            data = temp / 'data.tsv'
            fields = ['StereoQueerEval_id', 'yt_title', 'yt_description', 'yt_comment', 'target']
            with data.open('w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields, delimiter='\t')
                writer.writeheader()
                for i, label in enumerate(('group_t', 'none')):
                    writer.writerow(dict(zip(fields, [str(i), 'Title', 'Description', 'Comment', label])))
            config = json.loads((ROOT / 'src/configs/inference.json').read_text(encoding='utf-8'))
            config['data']['files']['dev'] = str(data)
            config['prompt']['directory'] = str(ROOT / 'src/prompts')
            config['output_dir'] = str(temp / 'outputs')
            config_path = temp / 'config.json'
            config_path.write_text(json.dumps(config), encoding='utf-8')
            with patch.dict(sys.modules, {'torch': fake_torch, 'transformers': fake_transformers,
                                          'tqdm': SimpleNamespace(tqdm=Progress)}), \
                    patch.object(sys, 'argv', ['inference.py', '--config', str(config_path)]), \
                    redirect_stdout(io.StringIO()):
                main()
            run = next((temp / 'outputs').rglob('run.json')).parent
            with (run / 'predictions.tsv').open(encoding='utf-8', newline='') as stream:
                rows = list(csv.DictReader(stream, delimiter='\t'))
            self.assertEqual(rows[0]['reasoning'], 'Targets trans people.')
            self.assertNotIn('evidence_json', rows[0])
            self.assertEqual(rows[1]['annotation_valid'], 'False')
            self.assertEqual(rows[1]['pred_target'], 'none')
            report = json.loads((run / 'metrics.json').read_text(encoding='utf-8'))
            self.assertEqual(report['full_target_exact_match'], 1)
            manifest = json.loads((run / 'run.json').read_text(encoding='utf-8'))
            self.assertEqual(manifest['status'], 'complete')
            self.assertEqual(manifest['output_schema'], ['target', 'reasoning'])


if __name__ == '__main__':
    unittest.main()
