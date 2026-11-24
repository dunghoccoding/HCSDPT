"""SQLite contracts and checked little-endian feature serialization."""

import sqlite3
from pathlib import Path

import numpy as np

from .features import DIMS

SCHEMA = """
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS images (
 image_id INTEGER PRIMARY KEY, path TEXT NOT NULL UNIQUE,
 sha256 TEXT NOT NULL, pixel_sha256 TEXT NOT NULL, source_group TEXT NOT NULL,
 width INTEGER, height INTEGER, duplicate_of INTEGER REFERENCES images(image_id),
 status TEXT NOT NULL, error TEXT
);
CREATE TABLE IF NOT EXISTS feature_sets (
 feature_set_id TEXT PRIMARY KEY, version TEXT NOT NULL, metadata_json TEXT NOT NULL,
 created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS normalization_params (
 normalization_id TEXT PRIMARY KEY, feature_set_id TEXT NOT NULL REFERENCES feature_sets,
 kind TEXT NOT NULL CHECK(kind IN ('raw','zscore')), status TEXT NOT NULL,
 gallery_json TEXT NOT NULL, params_json TEXT NOT NULL,
 created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
 UNIQUE(normalization_id, feature_set_id)
);
CREATE TABLE IF NOT EXISTS image_features (
 image_id INTEGER NOT NULL REFERENCES images, feature_set_id TEXT NOT NULL REFERENCES feature_sets,
 normalization_id TEXT NOT NULL,
 status TEXT NOT NULL, error TEXT, metadata_json TEXT NOT NULL,
 dimensions_json TEXT NOT NULL DEFAULT '[7,10,10,42,50]',
 f1 BLOB, f2 BLOB, f3 BLOB, f4 BLOB, f5 BLOB, f4_match BLOB,
 landmarks BLOB, alignment BLOB, overlay_path TEXT,
 PRIMARY KEY(image_id,feature_set_id,normalization_id),
 FOREIGN KEY(normalization_id,feature_set_id) REFERENCES normalization_params(normalization_id,feature_set_id),
 CHECK(status != 'ok' OR (length(f1)=28 AND length(f2)=40 AND length(f3)=40
 AND length(f4)=168 AND length(f5)=200 AND length(f4_match)=168
 AND f1 IS NOT NULL AND f2 IS NOT NULL AND f3 IS NOT NULL
 AND f4 IS NOT NULL AND f5 IS NOT NULL AND f4_match IS NOT NULL))
);
CREATE TABLE IF NOT EXISTS search_runs (
 search_run_id INTEGER PRIMARY KEY, normalization_id TEXT NOT NULL REFERENCES normalization_params,
 query_path TEXT NOT NULL, query_hash TEXT NOT NULL, exclude_duplicates INTEGER NOT NULL,
 extraction_seconds REAL NOT NULL, matching_seconds REAL NOT NULL,
 created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS search_results (
 search_run_id INTEGER NOT NULL REFERENCES search_runs, rank INTEGER NOT NULL,
 image_id INTEGER NOT NULL REFERENCES images, score REAL NOT NULL,
 s1 REAL NOT NULL,s2 REAL NOT NULL,s3 REAL NOT NULL,s4 REAL NOT NULL,s5 REAL NOT NULL,
 PRIMARY KEY(search_run_id,rank), UNIQUE(search_run_id,image_id)
);
"""


def connect(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.executescript(SCHEMA)
    return db


def pack(vector):
    vector = np.asarray(vector)
    if not np.isfinite(vector).all():
        raise ValueError('Nonfinite vector cannot be persisted')
    return vector.astype('<f4').tobytes()


def unpack(blob, dimension):
    vector = np.frombuffer(blob, dtype='<f4').astype(np.float64)
    if vector.shape != (dimension,) or not np.isfinite(vector).all():
        raise ValueError('Invalid stored vector')
    return vector


def vectors(row):
    return [unpack(row[f'f{i}'], dim) for i, dim in enumerate(DIMS, 1)]
