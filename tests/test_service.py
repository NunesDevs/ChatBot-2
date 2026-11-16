import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from chat_service import ChatService
from intents_core import IntentStore
from model_runtime import inspect_artifact

DOCUMENT = {'intents': [{'tag': 'greeting', 'patterns': ['Olá'], 'responses': ['Olá!']} ]}


class ServiceTests(unittest.TestCase):
    def test_default_service_works_without_neural_model(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'intents.json'
            path.write_text(json.dumps(DOCUMENT), encoding='utf-8')
            store = IntentStore(path)
            service = ChatService(store)
            self.assertEqual(service.respond('OLA')['response'], 'Olá!')
            self.assertEqual(service.respond('unknown')['mode'], 'fallback')
            self.assertEqual(service.state()['mode'], 'patterns')

    def test_updated_intents_invalidate_a_previously_trained_model(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'intents.json'
            path.write_text(json.dumps(DOCUMENT), encoding='utf-8')
            store = IntentStore(path)
            class Predictor:
                source_digest = store.snapshot().digest
                calls = 0
                def predict(self, _message):
                    self.calls += 1
                    return {'intent': 'greeting', 'confidence': 0.9}
            model = Predictor()
            service = ChatService(store, model)
            self.assertEqual(service.respond('different greeting')['mode'], 'neural')
            store.teach('Tchau', 'goodbye', 'Até logo')
            self.assertEqual(service.respond('Tchau')['response'], 'Até logo')
            self.assertEqual(service.respond('different greeting')['mode'], 'fallback')
            self.assertEqual(model.calls, 1)
            self.assertTrue(service.state()['model_stale'])

    def test_model_metadata_and_hash_are_checked_without_importing_tensorflow(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / 'model.keras'
            model.write_bytes(b'metadata-test-not-a-neural-model')
            metadata = {'format_version': 1, 'tokenizer_version': 'nfkd-words-v1', 'source_digest': 'a' * 64, 'model_sha256': hashlib.sha256(model.read_bytes()).hexdigest(), 'words': ['ola'], 'classes': ['greeting']}
            (root / 'metadata.json').write_text(json.dumps(metadata), encoding='utf-8')
            self.assertEqual(inspect_artifact(root)[0], metadata)
            model.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                inspect_artifact(root)


if __name__ == '__main__':
    unittest.main()
