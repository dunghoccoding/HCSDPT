"""Check persisted vectors, snapshot isolation, and all-gallery self retrieval."""

import json
import sqlite3

import numpy as np

from .database import unpack, vectors
from .features import EPS
from .inventory import ROOT


def main():
    db = sqlite3.connect((ROOT/'artifacts/portraits.sqlite3').as_uri()+'?mode=ro',uri=True)
    db.row_factory = sqlite3.Row
    assert db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
    assert not db.execute('PRAGMA foreign_key_check').fetchall()
    snapshot = db.execute("SELECT * FROM normalization_params WHERE kind='zscore' AND status='ready' ORDER BY rowid DESC LIMIT 1").fetchone()
    raw_id = 'raw:'+snapshot['feature_set_id']
    raw = db.execute('SELECT * FROM image_features WHERE normalization_id=? AND status=? ORDER BY image_id',(raw_id,'ok')).fetchall()
    normalized = db.execute('SELECT * FROM image_features WHERE normalization_id=? AND status=? ORDER BY image_id',(snapshot['normalization_id'],'ok')).fetchall()
    for row in raw:
        v = vectors(row)
        assert np.isclose(v[3][6:38].sum(),1,atol=1e-6)
        assert np.isclose(v[3][38:].sum(),1,atol=1e-6)
        assert np.allclose(v[4].reshape(5,10).sum(1),1,atol=1e-6)
        assert np.isclose(np.linalg.norm(unpack(row['f4_match'],42)),1,atol=1e-6)
        metadata = json.loads(row['metadata_json'])
        assert min(metadata['lbp_centers'].values()) >= 25
        assert len(row['landmarks']) == 468*2*8
        assert len(row['alignment']) == 2*3*8
    assert db.execute('SELECT COUNT(*) FROM image_features f JOIN images i USING(image_id) WHERE i.duplicate_of IS NOT NULL').fetchone()[0] == 0
    raw_by_id = {row['image_id']:row for row in raw}
    params = json.loads(snapshot['params_json'])
    for row in normalized:
        original = vectors(raw_by_id[row['image_id']])
        transformed = vectors(row)
        for group,param in enumerate(params):
            constant = np.array(param['constant'],dtype=bool)
            expected = np.zeros_like(original[group])
            mean,std = np.array(param['mean']),np.array(param['std'])
            expected[~constant] = (original[group][~constant]-mean[~constant])/std[~constant]
            np.testing.assert_array_equal(transformed[group],expected.astype('<f4').astype(float))
    data = [vectors(row) for row in normalized]
    for values,row in zip(data,normalized):
        values[3] = unpack(row['f4_match'],42)
    scores = np.zeros((len(data),len(data)))
    for group,weight in enumerate([.25,.20,.25,.10,.20]):
        matrix = np.stack([v[group] for v in data])
        norms = np.linalg.norm(matrix,axis=1)
        assert (norms > EPS).all()
        matrix /= norms[:,None]
        scores += weight*np.clip(matrix @ matrix.T,-1,1)
    assert np.allclose(scores.diagonal(),1,atol=1e-6)
    assert (scores.max(axis=1)-scores.diagonal() <= 1e-6).all()
    report = dict(raw_vectors_checked=len(raw),searchable_vectors_checked=len(normalized),
                  sqlite_integrity='ok',foreign_keys='ok',duplicates_have_no_features=True,
                  histogram_checks='passed',normalization_round_trip='passed',
                  self_retrieval_top_tie_group='passed',
                  max_self_score_error=float(np.max(np.abs(scores.diagonal()-1))))
    (ROOT/'artifacts/verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    db.close()


if __name__ == '__main__':
    main()
