"""Plain access to the data bundle (see data/README.md for its contents).

The bundle directory is ``data/bundle`` inside the package, or the directory named by the environment variable
``REPLICATION_DATA``. Every file is listed with its sha256 in ``data/MANIFEST.sha256``; README section 4 shows how to
verify them (shasum -c).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

PACKAGE = Path(__file__).resolve().parents[2]
ENCODERS = ('deberta-v3-base', 'ModernBERT-large', 'roberta-large', 'ettin-encoder-1b', 'deberta-v2-xlarge')
BASELINE = 'deberta-v3-base'
ALTERNATIVES = ENCODERS[1:]
DISPLAY = {'deberta-v3-base': 'DeBERTa-v3-base', 'ModernBERT-large': 'ModernBERT-large', 'roberta-large': 'RoBERTa-large',
           'ettin-encoder-1b': 'Ettin-encoder-1b', 'deberta-v2-xlarge': 'DeBERTa-v2-xlarge'}
META_COLUMNS = ['occurrence_id', 'term', 'matched_term', 'article_id', 'doc_id', 'country', 'year', 'month',
                'post_aukus', 'ai_similarity', 'pos_type', 'text_block', 'source_type', 'concept_id', 'concept_label']


def root() -> Path:
    env = os.environ.get('REPLICATION_DATA')
    return Path(env).resolve() if env else PACKAGE / 'data' / 'bundle'


def path(rel: str) -> Path:
    p = root() / rel
    if not p.is_file():
        raise FileNotFoundError(f'{p}: missing from the data bundle (see data/README.md)')
    return p


def results_dir() -> Path:
    env = os.environ.get('REPLICATION_RESULTS')
    out = Path(env).resolve() if env else PACKAGE / 'results'
    out.mkdir(parents=True, exist_ok=True)
    return out


# --------------------------------------------------------------------------- corpus
def occurrence_meta(columns=None) -> pd.DataFrame:
    """Metadata of the 37,866 occurrences (row order of every analysis of the baseline encoder)."""
    return pd.read_parquet(path('corpus/occurrences.parquet'), columns=list(columns or META_COLUMNS))


def analysis_meta() -> pd.DataFrame:
    return occurrence_meta(['occurrence_id', 'country', 'year', 'month', 'post_aukus', 'doc_id'])


def original_vectors() -> np.ndarray:
    """The semantic vectors of the original submission (``Y_vector_global``, float64, 37,866 x 768): the basis of the
    reference axes and of the original Figures 1-2."""
    import pyarrow.parquet as pq
    array = pq.read_table(path('corpus/occurrences.parquet'), columns=['Y_vector_global']).column(0).combine_chunks()
    values = array.values.to_numpy(zero_copy_only=False)
    widths = np.diff(np.asarray(array.offsets))
    if len(array) == 0 or not (widths == widths[0]).all():
        raise ValueError('Y_vector_global is empty or ragged')
    return np.asarray(values, dtype=np.float64).reshape(len(array), int(widths[0]))


def anchor_words() -> list:
    """The 100 anchor words (list order)."""
    return list(json.loads(path('corpus/anchor_words.json').read_text(encoding='utf-8')).get('words', []))


def anchor_occurrences() -> pd.DataFrame:
    """The 49,999 anchor occurrences of the A matrix (row order of every anchor encoding), with their anchor word."""
    return pd.read_parquet(path('corpus/anchor_occurrences.parquet'))


def common_support_map() -> pd.DataFrame:
    """Rows kept by all five encoders (truncation at 512 tokens never removes the masked term): original row index,
    row key and position, separately for target and anchor occurrences."""
    return pd.read_parquet(path('corpus/common_support_map.parquet'))


def terms() -> pd.DataFrame:
    return pd.read_csv(path('corpus/terms.csv'))


# --------------------------------------------------------------------------- encodings
def encoding(encoder: str, kind: str) -> tuple:
    """(U float32, rows DataFrame with row_key) of one encoder's target or anchor occurrences."""
    U = np.load(path(f'encodings/{encoder}/{kind}_U.npy'))
    rows = pd.read_parquet(path(f'encodings/{encoder}/{kind}_rows.parquet'))
    if U.dtype != np.float32 or len(U) != len(rows):
        raise ValueError(f'encodings/{encoder}/{kind}: {U.dtype} {U.shape} does not match its {len(rows)} rows')
    return U, rows


def external_encoding(encoder: str) -> tuple:
    """(U float32, rows) of one encoder's Singapore / Canada target occurrences."""
    U = np.load(path(f'external_controls/encodings/{encoder}/targets_U.npy'))
    rows = pd.read_parquet(path(f'external_controls/encodings/{encoder}/targets_rows.parquet'))
    if U.dtype != np.float32 or len(U) != len(rows):
        raise ValueError(f'external_controls/encodings/{encoder}: {U.dtype} {U.shape} vs {len(rows)} rows')
    return U, rows


def external_targets_meta() -> pd.DataFrame:
    return pd.read_parquet(path('external_controls/targets_meta.parquet'))


def external_documents() -> pd.DataFrame:
    return pd.read_parquet(path('external_controls/documents.parquet'),
                           columns=['article_id', 'country', 'publish_date', 'genre', 'title', 'content'])


def archived_json(rel: str) -> dict:
    return json.loads(path(rel).read_text(encoding='utf-8'))


def gpt2_dir() -> Path:
    env = os.environ.get('REPLICATION_GPT2')
    return Path(env).resolve() if env else root() / 'gpt2'
