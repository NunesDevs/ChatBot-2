"""Explicit, offline intent-model training. Importing this module never trains."""
import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from intents_core import load_dataset, tokens
from model_runtime import FORMAT_VERSION, TOKENIZER_VERSION

ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class TrainingData:
    words: tuple
    classes: tuple
    rows: tuple
    source_digest: str

    def summary(self):
        return {'samples': len(self.rows), 'vocabulary': len(self.words),
                'classes': len(self.classes), 'source_digest': self.source_digest}


def prepare_training(dataset):
    """Create a deterministic vocabulary; reject indistinguishable target labels."""
    classes = tuple(sorted(item.tag for item in dataset.intents))
    samples = {}
    for intent in dataset.intents:
        for pattern in intent.patterns:
            bag = frozenset(tokens(pattern))
            if bag in samples and samples[bag] != intent.tag:
                raise ValueError('Padrões com as mesmas palavras não podem ter classes neurais diferentes')
            samples[bag] = intent.tag
    words = tuple(sorted({word for sample in samples for word in sample}))
    if not 1 <= len(words) <= 10000 or not 1 <= len(samples) <= 5000:
        raise ValueError('O treinamento aceita até 10.000 palavras e 5.000 padrões distintos')
    if len(samples) * (len(words) + len(classes)) > 8_000_000:
        raise ValueError('A matriz de treinamento excede o limite; reduza a base')
    indexes = {word: index for index, word in enumerate(words)}
    class_indexes = {tag: index for index, tag in enumerate(classes)}
    rows = tuple((tuple(sorted(indexes[word] for word in bag)), class_indexes[tag])
                 for bag, tag in sorted(samples.items(), key=lambda item: (item[1], sorted(item[0]))))
    return TrainingData(words, classes, rows, dataset.digest)


def train_model(data, output, epochs=40, seed=42):
    if isinstance(epochs, bool) or not isinstance(epochs, int) or not 1 <= epochs <= 500:
        raise ValueError('Use de 1 a 500 épocas')
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**31 - 1:
        raise ValueError('Semente inválida')
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError('A pasta de saída já existe; escolha uma nova pasta')
    output.parent.mkdir(parents=True, exist_ok=True)
    # CPU-only, bounded workers; no tf.data thread pool for this small dataset.
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('TF_NUM_INTEROP_THREADS', '1')
    os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '1')
    import numpy as np
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.keras.utils.set_random_seed(seed)
    x = np.zeros((len(data.rows), len(data.words)), dtype='float32')
    y = np.zeros((len(data.rows), len(data.classes)), dtype='float32')
    for index, (positions, target) in enumerate(data.rows):
        x[index, list(positions)] = 1
        y[index, target] = 1
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(len(data.words),)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(len(data.classes), activation='softmax'),
    ])
    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.005), metrics=['accuracy'])
    random = np.random.default_rng(seed)
    metrics = {}
    for _epoch in range(epochs):
        model.reset_metrics()
        order = random.permutation(len(x))
        for start in range(0, len(order), 5):
            batch = order[start:start + 5]
            metrics = model.train_on_batch(x[batch], y[batch], return_dict=True)
    metrics = {key: float(value) for key, value in metrics.items()}
    if not metrics or not all(math.isfinite(value) for value in metrics.values()):
        raise RuntimeError('O treinamento produziu métricas inválidas')
    # TemporaryDirectory owns only this unique staging directory.
    with tempfile.TemporaryDirectory(prefix='.intent-model-', dir=output.parent) as staging:
        artifact = Path(staging) / 'artifact'
        artifact.mkdir()
        model_path = artifact / 'model.keras'
        model.save(model_path)
        digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
        metadata = {'format_version': FORMAT_VERSION, 'tokenizer_version': TOKENIZER_VERSION,
                    'words': list(data.words), 'classes': list(data.classes),
                    'source_digest': data.source_digest, 'model_sha256': digest,
                    'epochs': epochs, 'seed': seed, 'samples': len(data.rows),
                    'training_metrics': metrics, 'metrics_scope': 'training-only; not generalization evidence'}
        (artifact / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        if output.exists() or output.is_symlink():
            raise FileExistsError('A pasta de saída foi criada durante o treinamento')
        artifact.rename(output)
    return {**data.summary(), 'epochs': epochs, 'training_metrics': metrics,
            'metrics_scope': metadata['metrics_scope'], 'model_sha256': digest}


def main(argv=None):
    parser = argparse.ArgumentParser(description='Valida ou treina explicitamente o modelo de intenções local.')
    parser.add_argument('--intents', type=Path, default=ROOT / 'intents.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'model-trained')
    parser.add_argument('--epochs', type=int, choices=range(1, 501), default=40, metavar='1..500')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--prepare-only', action='store_true', help='Valida sem TensorFlow, rede ou escrita de arquivos')
    options = parser.parse_args(argv)
    try:
        data = prepare_training(load_dataset(options.intents))
        result = data.summary() if options.prepare_only else train_model(data, options.output, options.epochs, options.seed)
    except (OSError, RuntimeError, ValueError, ImportError) as error:
        parser.exit(1, f'Não foi possível preparar/treinar: {error}\n')
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
