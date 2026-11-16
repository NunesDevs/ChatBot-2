import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from app import create_app

DOCUMENT = {'intents': [{'tag': 'greeting', 'patterns': ['Olá'], 'responses': ['Olá!']} ]}
TOKEN = 'unit-test-learning-key-for-local-fixtures-only'


class AppTests(unittest.TestCase):
    def test_import_does_not_initialize_models_data_or_editing_configuration(self):
        environment = {**os.environ, 'CHATBOT_MODEL_DIRECTORY': 'must-not-be-loaded',
                       'CHATBOT_ENABLE_LEARNING': '1', 'CHATBOT_LEARNING_TOKEN': 'invalid'}
        result = subprocess.run([sys.executable, '-X', 'utf8', '-c',
                                 "import app, sys; assert 'tensorflow' not in sys.modules; print('inert')"],
                                cwd=Path(__file__).resolve().parents[1], env=environment,
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), 'inert')

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / 'intents.json'
        self.original = json.dumps(DOCUMENT).encode()
        self.path.write_bytes(self.original)

    def client(self, enabled=False):
        return create_app({'TESTING': True, 'INTENTS_PATH': self.path, 'LEARNING_ENABLED': enabled, 'LEARNING_TOKEN': TOKEN, 'MODEL_DIRECTORY': ''}).test_client()

    def test_chat_starts_without_training_downloads_or_neural_imports(self):
        client = self.client()
        response = client.post('/chat', json={'message': 'OLA'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json['intent'], 'greeting')
        self.assertEqual(response.json['response'], 'Olá!')
        self.assertNotIn('tensorflow', sys.modules)
        self.assertNotIn('nltk', sys.modules)
        self.assertEqual(self.path.read_bytes(), self.original)
        self.assertEqual(response.headers['Cache-Control'], 'no-store')

    def test_json_types_content_length_and_message_limits(self):
        client = self.client()
        for payload in ({}, {'message': None}, {'message': 3}, {'message': []}, {'message': ' '}, {'message': 'a' * 501}, []):
            response = client.post('/chat', json=payload)
            self.assertEqual(response.status_code, 400)
        self.assertEqual(client.post('/chat', data='null', content_type='application/json').status_code, 400)
        self.assertEqual(client.post('/chat', data='{', content_type='application/json').status_code, 400)
        self.assertEqual(client.post('/chat', data='message=oi').status_code, 415)
        self.assertEqual(client.post('/chat', data='x' * 16385, content_type='application/json').status_code, 413)

    def test_learning_is_read_only_by_default_and_requires_operator_key(self):
        change = {'message': 'Tchau', 'tag': 'goodbye', 'response': 'Até logo!'}
        self.assertEqual(self.client().post('/learn', json=change).status_code, 403)
        client = self.client(True)
        for header in ('', 'Bearer invalid', 'Bearer ' + 'x' * 40, 'Bearer á' + 'a' * 40):
            self.assertEqual(client.post('/learn', json=change, headers={'Authorization': header}).status_code, 401)
        self.assertEqual(self.path.read_bytes(), self.original)
        self.assertNotIn(TOKEN, client.get('/').get_data(as_text=True))
        self.assertNotIn(TOKEN, client.get('/status').get_data(as_text=True))

    def test_authorized_learning_updates_only_fixture_without_retraining(self):
        client = self.client(True)
        headers = {'Authorization': 'Bearer ' + TOKEN}
        change = {'message': 'Tchau', 'tag': 'goodbye', 'response': 'Até logo!'}
        first = client.post('/learn', json=change, headers=headers)
        second = client.post('/learn', json=change, headers=headers)
        self.assertEqual(first.status_code, 200)
        self.assertTrue(first.json['changed'])
        self.assertFalse(second.json['changed'])
        self.assertEqual(client.post('/chat', json={'message': 'TCHAU!'}).json['response'], 'Até logo!')
        self.assertEqual(len(json.loads(self.path.read_text())['intents']), 2)
        self.assertNotIn('tensorflow', sys.modules)
        invalid = {'message': 'Olá', 'tag': 'conflict', 'response': 'No'}
        self.assertEqual(client.post('/learn', json=invalid, headers=headers).status_code, 400)

    def test_invalid_external_data_returns_generic_retryable_error(self):
        client = self.client()
        self.path.write_text('{', encoding='utf-8')
        for response in (client.post('/chat', json={'message': 'Olá'}), client.get('/status')):
            self.assertEqual(response.status_code, 503)
            self.assertNotIn(str(self.path), response.get_data(as_text=True))

    def test_weak_or_ambiguous_editing_configuration_fails_closed(self):
        for config in ({'LEARNING_ENABLED': True, 'LEARNING_TOKEN': ''}, {'LEARNING_ENABLED': True, 'LEARNING_TOKEN': 'short'}, {'LEARNING_ENABLED': 'false', 'LEARNING_TOKEN': TOKEN}):
            with self.assertRaises(ValueError):
                create_app({'INTENTS_PATH': self.path, **config})


if __name__ == '__main__':
    unittest.main()
