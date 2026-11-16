import json
from pathlib import Path
import tempfile
import unittest
from intents_core import IntentStore, edit_lock, load_dataset, match_intent, validate_document

DOCUMENT = {'intents': [{'tag': 'greeting', 'patterns': ['Olá', 'Oi, tudo bem?'], 'responses': ['Olá!']}, {'tag': 'thanks', 'patterns': ['Muito obrigado', 'Valeu'], 'responses': ['De nada!']}]}


class IntentTests(unittest.TestCase):
    def test_normalized_matching_and_unknown_inputs(self):
        data = validate_document(DOCUMENT)
        self.assertEqual(match_intent('OLA!!!', data)['intent'], 'greeting')
        self.assertEqual(match_intent('obrigado muito', data)['intent'], 'thanks')
        self.assertIsNone(match_intent('quantos planetas existem', data))
        for message in ('', None, 'a' * 501):
            with self.assertRaises(ValueError):
                match_intent(message, data)

    def test_ambiguous_intents_do_not_guess(self):
        data = validate_document({'intents': [{'tag': 'one', 'patterns': ['quero ver um filme'], 'responses': ['1']}, {'tag': 'two', 'patterns': ['quero ver um livro'], 'responses': ['2']}]})
        self.assertIsNone(match_intent('quero ver', data))

    def test_invalid_schemas_and_shared_patterns_are_rejected(self):
        for document in ({}, [], {'intents': []}, {'intents': [{'tag': '../unsafe', 'patterns': ['oi'], 'responses': ['hi']}]}, {'intents': [{'tag': 'a', 'patterns': ['oi'], 'responses': []}]}, {'intents': [{'tag': 'a', 'patterns': ['oi'], 'responses': ['a']}, {'tag': 'b', 'patterns': ['OI!'], 'responses': ['b']}]}):
            with self.assertRaises(ValueError):
                validate_document(document)

    def test_atomic_teaching_is_deduplicated_and_visible_to_other_stores(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'intents.json'
            path.write_text(json.dumps(DOCUMENT), encoding='utf-8')
            first, second = IntentStore(path), IntentStore(path)
            self.assertTrue(first.teach('Até logo', 'goodbye', 'Tchau!'))
            self.assertFalse(first.teach('ATÉ LOGO!', 'goodbye', 'Tchau!'))
            self.assertEqual(match_intent('ate logo', second.snapshot())['intent'], 'goodbye')
            self.assertFalse(Path(str(path) + '.lock').exists())
            self.assertEqual(list(Path(directory).glob('*.tmp')), [])

    def test_failed_edit_preserves_original_bytes_and_other_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'intents.json'
            original = json.dumps(DOCUMENT).encode()
            path.write_bytes(original)
            store = IntentStore(path)
            with self.assertRaises(ValueError):
                store.teach('Olá!', 'another', 'Conflict')
            self.assertEqual(path.read_bytes(), original)
            with edit_lock(path):
                with self.assertRaises(RuntimeError):
                    store.teach('new', 'new_tag', 'Response')
                self.assertTrue(Path(str(path) + '.lock').exists())
            self.assertEqual(path.read_bytes(), original)

    def test_repository_data_validates_without_loading_pickle_or_tensorflow(self):
        data = load_dataset(Path(__file__).resolve().parents[1] / 'intents.json')
        self.assertEqual(match_intent('Olá', data)['intent'], 'greeting')
        self.assertGreater(len(data.intents), 0)


if __name__ == '__main__':
    unittest.main()
