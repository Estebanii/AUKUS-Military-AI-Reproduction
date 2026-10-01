"""GPT-2 labels of the anchor words and the GPT-2 vocabulary of the neighbour-word analysis.

A label is the float32 mean of the GPT-2 input embeddings (``wte``) of the word's sub-tokens
(``tokenizer.encode(word, add_special_tokens=False)``); the key is the lower-cased word; an empty tokenisation gives
zeros. Labels depend only on GPT-2, never on the encoder being studied. The GPT-2 files are pinned
(:data:`replication.params.GPT2`) and their sha256 are verified before use.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .params import GPT2

GPT2_FILES = tuple(GPT2['files_sha256'])


class LabelSourceError(RuntimeError):
    pass


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_gpt2(model_dir) -> dict:
    """The pinned GPT-2 snapshot directory, every file checked against its pinned sha256."""
    model_dir = Path(model_dir)
    hashes = {name: sha256_file(model_dir / name) for name in GPT2_FILES if (model_dir / name).is_file()}
    wrong = {k: (hashes.get(k), v) for k, v in GPT2['files_sha256'].items() if hashes.get(k) != v}
    if wrong:
        raise LabelSourceError(f'GPT-2 files in {model_dir} differ from the pinned hashes: {wrong}')
    return {'dir': str(model_dir), 'revision': GPT2['revision'], 'files_sha256': hashes}


def load_gpt2(source: dict):
    """(tokenizer, wte float32 [vocab, 768]) from a resolved snapshot."""
    from safetensors import safe_open
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(source['dir'])
    with safe_open(str(Path(source['dir']) / 'model.safetensors'), 'np') as handle:
        key = 'wte.weight' if 'wte.weight' in handle.keys() else 'transformer.wte.weight'
        wte = np.asarray(handle.get_tensor(key), dtype=np.float32)
    return tokenizer, wte


def anchor_labels(words, tokenizer, wte) -> tuple:
    """``(V_dict, token_ids)``: V_dict[word.lower()] = float32 mean of the word's GPT-2 ``wte`` rows (torch float32
    mean, as in the original implementation); an empty tokenisation gives zeros."""
    import torch
    table = torch.from_numpy(np.ascontiguousarray(wte))
    labels, ids_out = {}, {}
    for word in words:
        ids = tokenizer.encode(word, add_special_tokens=False)
        if not ids:
            vector = np.zeros(wte.shape[1])
        else:
            with torch.no_grad():
                vector = torch.nn.functional.embedding(torch.tensor(ids), table).mean(dim=0).numpy()
            vector = vector.astype(np.float32)
        labels[word.lower()] = vector
        ids_out[word] = list(map(int, ids))
    return labels, ids_out


def label_matrix(word_labels, labels: dict):
    """Rows (and their positions) whose lower-cased word has a label."""
    rows, valid = [], []
    for i, word in enumerate(word_labels):
        key = str(word).lower()
        if key in labels:
            rows.append(labels[key])
            valid.append(i)
    return np.array(rows), np.asarray(valid, dtype=np.int64)
