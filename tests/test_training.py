import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from intents_core import validate_document
from train import prepare_training, train_model

ROOT = Path(__file__).resolve().parents[1]


class TrainingTests(unittest.TestCase):
    def test_deterministic_preparation_uses_shared_normalization(self):
        dataset = validate_document({'intents': [
            {'tag': 'bye', 'patterns': ['Até logo'], 'responses': ['Tchau']},
            {'tag': 'hello', 'patterns': ['Olá!', 'OLA', 'Olá mundo'], 'responses': ['Oi']},
        ]})
        data = prepare_training(dataset)
        self.assertEqual(data.words, ('ate', 'logo', 'mundo', 'ola'))
        self.assertEqual(data.classes, ('bye', 'hello'))
        self.assertEqual(len(data.rows), 3)
        self.assertEqual(data, prepare_training(dataset))
        self.assertNotIn('tensorflow', sys.modules)

    def test_bag_of_words_collisions_between_classes_are_rejected(self):
        dataset = validate_document({'intents': [
            {'tag': 'a', 'patterns': ['um dois'], 'responses': ['A']},
            {'tag': 'b', 'patterns': ['dois um'], 'responses': ['B']},
        ]})
        with self.assertRaises(ValueError):
            prepare_training(dataset)

    def test_prepare_only_cli_never_creates_model_or_changes_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'model'
            original = (ROOT / 'intents.json').read_bytes()
            result = subprocess.run([sys.executable, '-X', 'utf8', str(ROOT / 'train.py'),
                                     '--prepare-only', '--output', str(output)], capture_output=True, text=True, encoding='utf-8', timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertGreater(json.loads(result.stdout)['samples'], 0)
            self.assertFalse(output.exists())
            self.assertEqual((ROOT / 'intents.json').read_bytes(), original)

    def test_output_preservation_and_bounds_happen_before_heavy_imports(self):
        data = prepare_training(validate_document({'intents': [{'tag': 'a', 'patterns': ['Oi'], 'responses': ['A']}]}))
        with tempfile.TemporaryDirectory() as directory:
            sentinel = Path(directory) / 'keep.txt'
            sentinel.write_text('preserve')
            with self.assertRaises(FileExistsError):
                train_model(data, directory)
            for epochs in (0, 501, True, 1.5):
                with self.assertRaises(ValueError):
                    train_model(data, Path(directory) / 'new', epochs)
            self.assertEqual(sentinel.read_text(), 'preserve')
        self.assertNotIn('tensorflow', sys.modules)


if __name__ == '__main__':
    unittest.main()
