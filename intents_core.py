"""Validated local intents, lightweight matching and authorized atomic edits."""
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import threading
import unicodedata
import uuid

MAX_MESSAGE = 500
MAX_RESPONSE = 2000
MAX_BYTES = 2 * 1024 * 1024


def normalized(value):
    value = unicodedata.normalize('NFKD', value.casefold())
    return ''.join(char for char in value if unicodedata.category(char) != 'Mn')


def tokens(value):
    return tuple(re.findall(r'[^\W_]+', normalized(value), flags=re.UNICODE))


def text(value, maximum, label):
    if not isinstance(value, str) or not 1 <= len(value.strip()) <= maximum:
        raise ValueError(f'{label} deve conter de 1 a {maximum} caracteres')
    return value.strip()


@dataclass(frozen=True)
class Intent:
    tag: str
    patterns: tuple
    responses: tuple


@dataclass(frozen=True)
class Dataset:
    intents: tuple
    digest: str

    def document(self):
        return {'intents': [{'tag': item.tag, 'patterns': list(item.patterns), 'responses': list(item.responses)} for item in self.intents]}


def validate_document(document):
    if not isinstance(document, dict) or not isinstance(document.get('intents'), list) or not 1 <= len(document['intents']) <= 1000:
        raise ValueError('A base deve conter uma lista de intenções')
    intents, tags, owners = [], set(), {}
    for item in document['intents']:
        if not isinstance(item, dict):
            raise ValueError('Intenção inválida')
        tag = item.get('tag')
        if not isinstance(tag, str) or not re.fullmatch(r'[a-z][a-z0-9_]{0,63}', tag) or tag in tags:
            raise ValueError('Tags devem ser únicas, em letras minúsculas, números e sublinhado')
        tags.add(tag)
        values = {}
        for name, limit in [('patterns', MAX_MESSAGE), ('responses', MAX_RESPONSE)]:
            sequence = item.get(name)
            if not isinstance(sequence, list) or not 1 <= len(sequence) <= 100:
                raise ValueError('Cada intenção precisa de padrões e respostas')
            clean = []
            for value in sequence:
                value = text(value, limit, 'Padrão' if name == 'patterns' else 'Resposta')
                if name == 'patterns':
                    key = ' '.join(tokens(value))
                    if not key:
                        raise ValueError('O padrão precisa conter palavras ou números')
                    if key in owners and owners[key] != tag:
                        raise ValueError('Um padrão não pode pertencer a duas intenções')
                    owners[key] = tag
                if value not in clean:
                    clean.append(value)
            values[name] = tuple(clean)
        intents.append(Intent(tag, values['patterns'], values['responses']))
    canonical = json.dumps({'intents': [{'tag': i.tag, 'patterns': list(i.patterns), 'responses': list(i.responses)} for i in intents]}, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')
    if len(canonical) > MAX_BYTES:
        raise ValueError('A base excede o limite permitido')
    return Dataset(tuple(intents), hashlib.sha256(canonical).hexdigest())


def load_dataset(path):
    source = Path(path)
    try:
        if source.stat().st_size > MAX_BYTES:
            raise ValueError('A base excede o limite permitido')
        return validate_document(json.loads(source.read_text(encoding='utf-8')))
    except (OSError, UnicodeError, ValueError) as error:
        raise RuntimeError('A base de intenções está indisponível') from error


def match_intent(message, dataset):
    message = text(message, MAX_MESSAGE, 'Mensagem')
    sentence = tokens(message)
    if not sentence:
        return None
    query, ranked = set(sentence), {}
    for intent in dataset.intents:
        for pattern in intent.patterns:
            candidate = tokens(pattern)
            if candidate == sentence:
                return {'intent': intent.tag, 'confidence': 1.0, 'mode': 'exact'}
            vocabulary = set(candidate)
            score = 2 * len(query & vocabulary) / (len(query) + len(vocabulary))
            ranked[intent.tag] = max(ranked.get(intent.tag, 0), score)
    results = sorted(ranked.items(), key=lambda item: item[1], reverse=True)
    if not results or results[0][1] < 0.65:
        return None
    if len(results) > 1 and results[0][1] - results[1][1] < 0.1:
        return None
    return {'intent': results[0][0], 'confidence': results[0][1], 'mode': 'pattern'}


@contextmanager
def edit_lock(path):
    lock_path = Path(str(path) + '.lock')
    identity = uuid.uuid4().hex
    try:
        descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise RuntimeError('Outra edição está em andamento') from error
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
            handle.write(identity)
        yield
    finally:
        if lock_path.exists() and lock_path.read_text(encoding='utf-8') == identity:
            lock_path.unlink()


class IntentStore:
    def __init__(self, path):
        self.path = Path(path)
        self.lock = threading.RLock()
        self._dataset = None
        self._signature = None
        self.snapshot()

    def snapshot(self):
        with self.lock:
            stat = self.path.stat()
            signature = stat.st_mtime_ns, stat.st_size
            if signature != self._signature:
                self._dataset = load_dataset(self.path)
                self._signature = signature
            return self._dataset

    def teach(self, pattern, tag, response):
        pattern = text(pattern, MAX_MESSAGE, 'Padrão')
        response = text(response, MAX_RESPONSE, 'Resposta')
        if not isinstance(tag, str) or not re.fullmatch(r'[a-z][a-z0-9_]{0,63}', tag):
            raise ValueError('Tag inválida')
        with self.lock, edit_lock(self.path):
            current = load_dataset(self.path)
            document = current.document()
            destination = next((item for item in document['intents'] if item['tag'] == tag), None)
            if destination is None:
                destination = {'tag': tag, 'patterns': [], 'responses': []}
                document['intents'].append(destination)
            same_pattern = any(tokens(value) == tokens(pattern) for value in destination['patterns'])
            if same_pattern and response in destination['responses']:
                return False
            if not same_pattern:
                destination['patterns'].append(pattern)
            if response not in destination['responses']:
                destination['responses'].append(response)
            updated = validate_document(document)
            payload = json.dumps(updated.document(), ensure_ascii=False, indent=2) + '\n'
            if len(payload.encode('utf-8')) > MAX_BYTES:
                raise ValueError('A base excede o limite permitido')
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=self.path.parent, prefix='.' + self.path.name + '-', suffix='.tmp', delete=False) as handle:
                    temporary = Path(handle.name)
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, self.path)
                temporary = None
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
            self._dataset = updated
            stat = self.path.stat()
            self._signature = stat.st_mtime_ns, stat.st_size
            return True
