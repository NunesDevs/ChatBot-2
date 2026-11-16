"""Optional neural inference; heavyweight imports occur only when explicitly enabled."""
import hashlib
import json
import os
from pathlib import Path
import re
import threading
from intents_core import tokens

FORMAT_VERSION = 1
TOKENIZER_VERSION = 'nfkd-words-v1'


def inspect_artifact(directory):
    directory = Path(directory).resolve()
    metadata_path, model_path = directory / 'metadata.json', directory / 'model.keras'
    if metadata_path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError('Metadados do modelo excedem o limite')
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    if not isinstance(metadata, dict) or metadata.get('format_version') != FORMAT_VERSION or metadata.get('tokenizer_version') != TOKENIZER_VERSION:
        raise ValueError('Formato de modelo incompatível; treine com o CLI atual')
    for key in ('source_digest', 'model_sha256'):
        if not isinstance(metadata.get(key), str) or not re.fullmatch(r'[0-9a-f]{64}', metadata[key]):
            raise ValueError('Hash do modelo inválido')
    for key in ('words', 'classes'):
        values = metadata.get(key)
        if not isinstance(values, list) or not 1 <= len(values) <= 10000 or any(not isinstance(value, str) or not value for value in values) or len(values) != len(set(values)):
            raise ValueError('Vocabulário ou classes inválidos')
    if any(tokens(word) != (word,) for word in metadata['words']) or any(not re.fullmatch(r'[a-z][a-z0-9_]{0,63}', tag) for tag in metadata['classes']):
        raise ValueError('Vocabulário ou classes incompatíveis com o tokenizer')
    if model_path.is_symlink() or not model_path.is_file() or model_path.stat().st_size > 128 * 1024 * 1024:
        raise ValueError('Arquivo de modelo inválido')
    digest = hashlib.sha256()
    with model_path.open('rb') as handle:
        for block in iter(lambda: handle.read(65536), b''):
            digest.update(block)
    if digest.hexdigest() != metadata['model_sha256']:
        raise ValueError('Integridade do modelo não confirmada')
    return metadata, model_path


class KerasPredictor:
    def __init__(self, directory):
        metadata, model_path = inspect_artifact(directory)
        os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
        os.environ.setdefault('OMP_NUM_THREADS', '1')
        os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')
        os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '1')
        import numpy as np
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        self.model = tf.keras.models.load_model(model_path, compile=False, safe_mode=True)
        if self.model.input_shape != (None, len(metadata['words'])) or self.model.output_shape != (None, len(metadata['classes'])):
            raise ValueError('Dimensões do modelo incompatíveis com os metadados')
        if any(layer.__class__.__name__ not in {'InputLayer', 'Dense', 'Dropout'} for layer in self.model.layers):
            raise ValueError('Arquitetura neural não autorizada')
        self.source_digest = metadata['source_digest']
        self.words, self.classes = metadata['words'], metadata['classes']
        self.np = np
        self.lock = threading.Lock()

    def predict(self, message):
        vocabulary = set(tokens(message))
        row = [int(word in vocabulary) for word in self.words]
        if not any(row):
            return None
        with self.lock:
            result = self.np.asarray(self.model(self.np.asarray([row], dtype='float32'), training=False).numpy())[0]
        if len(result) != len(self.classes) or not self.np.isfinite(result).all() or (result < 0).any() or (result > 1).any():
            raise RuntimeError('Saída neural inválida')
        index = int(result.argmax())
        confidence = float(result[index])
        return {'intent': self.classes[index], 'confidence': confidence} if confidence >= 0.65 else None
