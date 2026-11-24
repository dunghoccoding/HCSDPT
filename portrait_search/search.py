"""Query a single ready SQLite snapshot; uploads never enter the gallery."""

import argparse
import csv
import hashlib
import json
import sqlite3
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .build_gallery import detector, model_metadata, process
from .database import pack, unpack, vectors
from .features import EPS, FeatureError
from .inventory import ROOT

WEIGHTS = np.array([.25,.20,.25,.10,.20])


def unit(vector):
    norm = np.linalg.norm(vector)
    if norm <= EPS:
        raise FeatureError('zero_norm', 'Cosine undefined for a zero-norm feature group')
    return vector/norm


def rank(query_vectors, rows, limit=5):
    query = [unit(v) for v in query_vectors]
    results = []
    for row in rows:
        gallery = vectors(row)
        gallery[3] = unpack(row['f4_match'],42)
        scores = np.array([np.clip(a @ unit(b),-1,1) for a,b in zip(query,gallery)])
        results.append(dict(image_id=row['image_id'],path=row['path'],score=float(scores @ WEIGHTS),
                            **{f's{i}':float(v) for i,v in enumerate(scores,1)}))
    results.sort(key=lambda r:(-r['score'],r['image_id']))
    return results[:limit]


def search(query_path, db_path=ROOT/'artifacts/portraits.sqlite3', normalization_id=None,
           exclude_duplicates=False, output=None):
    # Existing DB required: a typo must not create an empty database.
    db = sqlite3.connect(Path(db_path).resolve().as_uri()+'?mode=rw',uri=True)
    try:
        return _search(db,query_path,normalization_id,exclude_duplicates,output)
    finally:
        db.close()


def _search(db,query_path,normalization_id,exclude_duplicates,output):
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA foreign_keys=ON')
    if normalization_id:
        snapshot = db.execute("SELECT * FROM normalization_params WHERE normalization_id=? AND kind='zscore' AND status='ready'",(normalization_id,)).fetchone()
    else:
        snapshot = db.execute("SELECT * FROM normalization_params WHERE kind='zscore' AND status='ready' ORDER BY created_at DESC,rowid DESC LIMIT 1").fetchone()
    if snapshot is None:
        raise ValueError('No ready normalized snapshot')
    fs = db.execute('SELECT * FROM feature_sets WHERE feature_set_id=?',(snapshot['feature_set_id'],)).fetchone()
    metadata = json.loads(fs['metadata_json'])
    current = model_metadata()
    for key in ['model_sha256','roi','packages','connections']:
        if json.dumps(current[key],sort_keys=True) != json.dumps(metadata[key],sort_keys=True):
            raise ValueError(f'Query extraction differs from snapshot: {key}; rebuild gallery')
    for name in ['features.py','preprocess.py']:
        if current['code'][name] != metadata['code'][name]:
            raise ValueError(f'Extractor changed: {name}; rebuild gallery')
    query_path = Path(query_path).resolve()
    start = time.perf_counter()
    with Image.open(query_path) as source:
        rgb = np.array(source.convert('RGB'))
    file_hash = hashlib.sha256(query_path.read_bytes()).hexdigest()
    pixel_hash = hashlib.sha256(f'{rgb.shape[1]}x{rgb.shape[0]}:RGB:'.encode()+rgb.tobytes()).hexdigest()
    with detector(metadata) as landmarker:
        raw,m4,points,matrix,aligned,valid,transformed,details = process(rgb,landmarker)
    raw = [unpack(pack(v),len(v)) for v in raw]
    matching = [v.copy() for v in raw]
    for i,param in enumerate(json.loads(snapshot['params_json'])):
        constant = np.array(param['constant'],dtype=bool)
        mean,std = np.array(param['mean']),np.array(param['std'])
        z = np.zeros_like(raw[i])
        z[~constant] = (raw[i][~constant]-mean[~constant])/std[~constant]
        matching[i] = unpack(pack(z),len(z))
    matching[3] = unpack(pack(m4),42)
    for v in matching:
        unit(v)
    extraction_seconds = time.perf_counter()-start
    start = time.perf_counter()
    sql = '''SELECT f.*,i.path FROM image_features f JOIN images i USING(image_id)
             WHERE f.normalization_id=? AND f.status='ok' AND i.duplicate_of IS NULL'''
    params = [snapshot['normalization_id']]
    if exclude_duplicates:
        sql += ' AND i.sha256<>? AND i.pixel_sha256<>?'
        params += [file_hash,pixel_hash]
    rows = db.execute(sql,params).fetchall()
    results = rank(matching,rows)
    matching_seconds = time.perf_counter()-start
    with db:
        run_id = db.execute('''INSERT INTO search_runs
            (normalization_id,query_path,query_hash,exclude_duplicates,extraction_seconds,matching_seconds)
            VALUES (?,?,?,?,?,?)''',(snapshot['normalization_id'],str(query_path),file_hash,
                                   int(exclude_duplicates),extraction_seconds,matching_seconds)).lastrowid
        for index,result in enumerate(results,1):
            db.execute('INSERT INTO search_results VALUES (?,?,?,?,?,?,?,?,?)',
                       (run_id,index,result['image_id'],result['score'],*(result[f's{i}'] for i in range(1,6))))
    report = dict(search_run_id=run_id,query=str(query_path),normalization_id=snapshot['normalization_id'],
                  feature_set_id=snapshot['feature_set_id'],exclude_duplicates=exclude_duplicates,
                  gallery_candidates=len(rows),warning='Gallery has fewer than 5 eligible images' if len(rows)<5 else None,
                  extraction_seconds=extraction_seconds,matching_seconds=matching_seconds,
                  raw_vectors={f'f{i}':v.tolist() for i,v in enumerate(raw,1)},
                  matching_vectors={f'f{i}':v.tolist() for i,v in enumerate(matching,1)},
                  weights=WEIGHTS.tolist(),metadata=details,results=results)
    if output:
        output = Path(output)
        output.mkdir(parents=True,exist_ok=True)
        (output/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        with (output/'top5.csv').open('w',encoding='utf-8-sig',newline='') as file:
            writer = csv.DictWriter(file,fieldnames=['image_id','path','score','s1','s2','s3','s4','s5'])
            writer.writeheader()
            writer.writerows(results)
        Image.fromarray(rgb).save(output/'original.png')
        Image.fromarray(aligned).save(output/'aligned.png')
        from .features import overlay
        import mediapipe as mp
        Image.fromarray(overlay(aligned,transformed,valid,mp.tasks.vision.FaceLandmarksConnections)).save(output/'roi.png')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('query',type=Path)
    parser.add_argument('--db',type=Path,default=ROOT/'artifacts/portraits.sqlite3')
    parser.add_argument('--snapshot')
    parser.add_argument('--exclude-duplicates',action='store_true')
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/query_example')
    args = parser.parse_args()
    report = search(args.query,args.db,args.snapshot,args.exclude_duplicates,args.output)
    print(json.dumps({k:v for k,v in report.items() if k not in ['raw_vectors','matching_vectors','metadata']},indent=2))


if __name__ == '__main__':
    main()
