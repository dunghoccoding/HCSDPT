"""Deduplicate, extract F1-F5 and fit an immutable gallery normalization snapshot."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import time
from collections import Counter

import mediapipe as mp
import numpy as np
from PIL import Image

from . import FEATURE_VERSION
from .database import connect, pack, unpack, vectors
from .features import CONFIG, DIMS, EPS, NAMES, FeatureError, extract, overlay
from .inventory import ROOT
from .preprocess import align


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def model_metadata():
    model = ROOT / 'models/face_landmarker.task'
    connections = mp.tasks.vision.FaceLandmarksConnections
    groups = {name: [(e.start, e.end) for e in getattr(connections,name)]
              for name in dir(connections) if name.startswith('FACE_LANDMARKS_')}
    return dict(version=FEATURE_VERSION, dimensions=DIMS, names=NAMES, roi=CONFIG,
                model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
                model_url='https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
                detector=dict(num_faces=2, min_face_detection_confidence=.5,
                              min_face_presence_confidence=.5), connections=groups,
                canvas=[178,218], eye_targets=[[57,82],[121,82]],
                packages={p:importlib.metadata.version(p) for p in
                          ['mediapipe','numpy','Pillow','opencv-python','scikit-image']},
                code={name:hashlib.sha256((ROOT/'portrait_search'/name).read_bytes()).hexdigest()
                      for name in ['features.py','preprocess.py','build_gallery.py','database.py']})


def detector(metadata):
    return mp.tasks.vision.FaceLandmarker.create_from_options(mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(ROOT/'models/face_landmarker.task')),
        running_mode=mp.tasks.vision.RunningMode.IMAGE, **metadata['detector']))


def process(rgb, landmarker):
    result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    count = len(result.face_landmarks)
    if count != 1:
        raise FeatureError('no_face' if not count else 'multiple_faces', f'Detected {count} faces')
    points = np.array([[p.x*rgb.shape[1],p.y*rgb.shape[0]] for p in result.face_landmarks[0][:468]])
    connections = mp.tasks.vision.FaceLandmarksConnections
    try:
        aligned, valid, transformed, matrix, geometry = align(rgb, points, connections)
    except ValueError as exc:
        raise FeatureError('invalid_geometry', str(exc)) from exc
    raw, matching4, masks, details = extract(aligned, points, transformed, valid, connections)
    details.update(geometry)
    return raw, matching4, points, matrix, aligned, valid, transformed, details


def normalize(db, feature_set, raw_id):
    rows = db.execute('SELECT * FROM image_features WHERE feature_set_id=? AND normalization_id=? AND status=? ORDER BY image_id',
                      (feature_set, raw_id, 'ok')).fetchall()
    if not rows:
        raise ValueError('No complete feature vectors available')
    data = [vectors(row) for row in rows]
    params = []
    normalized = [[v.copy() for v in row] for row in data]
    for group in range(3):
        matrix = np.stack([row[group] for row in data])
        mean, std = matrix.mean(0), matrix.std(0, ddof=0)
        constant = std < EPS
        z = np.zeros_like(matrix)
        z[:, ~constant] = (matrix[:, ~constant]-mean[~constant])/std[~constant]
        for row, transformed in zip(normalized, z):
            row[group] = transformed
        params.append(dict(mean=mean.tolist(),std=std.tolist(),constant=constant.tolist()))
    gallery = [dict(image_id=row['image_id'], sha256=db.execute('SELECT sha256 FROM images WHERE image_id=?',
                    (row['image_id'],)).fetchone()[0]) for row in rows]
    norm_id = digest(dict(feature_set=feature_set,gallery=gallery,params=params))
    # All rows and the ready marker commit together. A failed build never exposes a partial snapshot.
    with db:
        db.execute('INSERT OR IGNORE INTO normalization_params VALUES (?,?,?,?,?,?,CURRENT_TIMESTAMP)',
                   (norm_id, feature_set, 'zscore', 'building', json.dumps(gallery), json.dumps(params)))
        zero_norm_ids = []
        for row, values in zip(rows, normalized):
            zero = any(np.linalg.norm(v) <= EPS for v in values[:3])
            if zero:
                zero_norm_ids.append(row['image_id'])
            db.execute('''INSERT OR REPLACE INTO image_features
                (image_id,feature_set_id,normalization_id,status,error,metadata_json,
                 f1,f2,f3,f4,f5,f4_match,landmarks,alignment,overlay_path)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                (row['image_id'], feature_set, norm_id, 'zero_norm' if zero else 'ok',
                 'Undefined cosine after Z-score' if zero else None, row['metadata_json'],
                 *(pack(v) for v in values), row['f4_match'],row['landmarks'],row['alignment'],row['overlay_path']))
        db.execute('UPDATE normalization_params SET status=? WHERE normalization_id=?', ('ready',norm_id))
    return norm_id, len(rows)-len(zero_norm_ids), zero_norm_ids


def build(db_path):
    start = time.perf_counter()
    with (ROOT/'artifacts/inventory/manifest.csv').open(encoding='utf-8-sig') as file:
        manifest = sorted(csv.DictReader(file), key=lambda row:row['path'])
    # Fail before mutation if files changed since inventory.
    for row in manifest:
        if hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest() != row['sha256']:
            raise ValueError(f"Manifest out of date: {row['path']}; rerun inventory")
    metadata = model_metadata()
    feature_set = digest(metadata)
    raw_id = 'raw:'+feature_set
    db = connect(db_path)
    output = ROOT/'artifacts/gallery'/feature_set[:16]
    output.mkdir(parents=True,exist_ok=True)
    with db:
        db.execute('INSERT OR IGNORE INTO feature_sets VALUES (?,?,?,CURRENT_TIMESTAMP)',
                   (feature_set,FEATURE_VERSION,json.dumps(metadata)))
        db.execute('INSERT OR IGNORE INTO normalization_params VALUES (?,?,?,?,?,?,CURRENT_TIMESTAMP)',
                   (raw_id,feature_set,'raw','building','[]','[]'))
        representatives = {}
        jobs = []
        for row in manifest:
            existing = db.execute('SELECT * FROM images WHERE path=?',(row['path'],)).fetchone()
            if existing and existing['sha256'] != row['sha256']:
                raise ValueError('Existing image changed; use a new DB to preserve prior snapshots')
            duplicate_of = representatives.get(row['pixel_sha256']) if row['pixel_sha256'] else None
            status = 'duplicate' if duplicate_of else row['status']
            db.execute('''INSERT INTO images(path,sha256,pixel_sha256,source_group,width,height,duplicate_of,status,error)
                          VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT(path) DO UPDATE SET
                          duplicate_of=excluded.duplicate_of,status=excluded.status,error=excluded.error''',
                       (row['path'],row['sha256'],row['pixel_sha256'],row['group'],row['width'] or None,
                        row['height'] or None,duplicate_of,status,row['error']))
            image_id = db.execute('SELECT image_id FROM images WHERE path=?',(row['path'],)).fetchone()[0]
            if duplicate_of is None and row['pixel_sha256']:
                representatives[row['pixel_sha256']] = image_id
            if duplicate_of is None:
                jobs.append((image_id,row))
    completed = 0
    with detector(metadata) as landmarker:
        for image_id, row in jobs:
            if db.execute('SELECT 1 FROM image_features WHERE image_id=? AND feature_set_id=? AND normalization_id=?',
                          (image_id,feature_set,raw_id)).fetchone():
                continue
            record = dict(image_id=image_id,feature_set_id=feature_set,normalization_id=raw_id,
                          status='ok',error=None,metadata_json='{}')
            try:
                with Image.open(ROOT/row['path']) as image:
                    rgb = np.array(image.convert('RGB'))
                raw,m4,points,matrix,aligned,valid,transformed,details = process(rgb,landmarker)
                for i,v in enumerate(raw,1):
                    record[f'f{i}'] = pack(v)
                record.update(f4_match=pack(m4),landmarks=np.asarray(points,dtype='<f8').tobytes(),
                              alignment=np.asarray(matrix,dtype='<f8').tobytes(),metadata_json=json.dumps(details))
                image_output = output/str(image_id)
                image_output.mkdir(exist_ok=True)
                Image.fromarray(aligned).save(image_output/'aligned.png')
                Image.fromarray(overlay(aligned,transformed,valid,mp.tasks.vision.FaceLandmarksConnections)).save(image_output/'roi.png')
                np.savez_compressed(image_output/'features.npz',**{f'f{i}':v for i,v in enumerate(raw,1)},f4_match=m4)
                record['overlay_path'] = (image_output/'roi.png').relative_to(ROOT).as_posix()
            except FeatureError as exc:
                record.update(status=exc.status,error=str(exc))
            except OSError as exc:
                record.update(status='unreadable',error=str(exc))
            keys = list(record)
            with db:
                db.execute(f"INSERT INTO image_features ({','.join(keys)}) VALUES ({','.join('?' for _ in keys)})",list(record.values()))
                db.execute('UPDATE images SET status=?,error=? WHERE image_id=?',
                           (record['status'],record['error'],image_id))
            completed += 1
            if completed % 100 == 0:
                print(f'Extracted {completed}/{len(jobs)} unique images',flush=True)
    counts = dict(db.execute('SELECT status,COUNT(*) FROM image_features WHERE normalization_id=? GROUP BY status',(raw_id,)).fetchall())
    gallery_ids = [r[0] for r in db.execute('SELECT image_id FROM image_features WHERE normalization_id=? ORDER BY image_id',(raw_id,))]
    with db:
        db.execute('UPDATE normalization_params SET status=?,gallery_json=? WHERE normalization_id=?',
                   ('ready',json.dumps(gallery_ids),raw_id))
    norm_id, eligible, zero_ids = normalize(db,feature_set,raw_id)
    summary = dict(total_files=len(manifest),unique_images=len(jobs),duplicates_skipped=len(manifest)-len(jobs),
                   extraction_statuses=counts,searchable_images=eligible,zero_norm_ids=zero_ids,
                   scale_requirement_met=eligible>=1000,feature_set_id=feature_set,normalization_id=norm_id,
                   elapsed_seconds=time.perf_counter()-start,database=str(db_path),
                   note='Baseline extraction success does not establish retrieval quality')
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    export = output/'features.csv'
    with export.open('w',encoding='utf-8-sig',newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['image_id','path']+[f'F{i}_{name}' for i,names in enumerate(NAMES,1) for name in names])
        for row in db.execute('SELECT f.*,i.path FROM image_features f JOIN images i USING(image_id) WHERE normalization_id=? AND f.status=? ORDER BY image_id',(raw_id,'ok')):
            writer.writerow([row['image_id'],row['path'],*np.concatenate(vectors(row)).tolist()])
    # Human-readable deduplication mapping; no original file is removed.
    with (output/'image_status.csv').open('w',encoding='utf-8-sig',newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['image_id','path','status','duplicate_of','error'])
        writer.writerows(db.execute('SELECT image_id,path,status,duplicate_of,error FROM images ORDER BY image_id'))
    print(json.dumps(summary,indent=2))
    db.close()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db',type=str,default=str(ROOT/'artifacts/portraits.sqlite3'))
    args = parser.parse_args()
    build(args.db)


if __name__ == '__main__':
    main()
