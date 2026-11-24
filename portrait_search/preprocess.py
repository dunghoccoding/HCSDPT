"""Run the first landmark/alignment review, preserving source coordinates."""

import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

from .inventory import ROOT


def indices(connections):
    return sorted({i for edge in connections for i in (edge.start, edge.end)})


def align(rgb, points, connections):
    right = points[indices(connections.FACE_LANDMARKS_RIGHT_EYE)].mean(axis=0)
    left = points[indices(connections.FACE_LANDMARKS_LEFT_EYE)].mean(axis=0)
    delta = left - right
    distance = np.linalg.norm(delta)
    if not np.isfinite(points).all() or distance <= 1e-8:
        raise ValueError('Nonfinite landmarks or degenerate eye axis')
    roll = math.degrees(math.atan2(delta[1], delta[0]))
    if abs(roll) > 25:
        raise ValueError(f'Eye roll {roll:.2f} exceeds 25 degrees')
    u = delta / distance
    v = np.array([-u[1], u[0]])
    brow_r = points[np.array([(70,46),(63,53),(105,52),(66,65),(107,55)])].mean(axis=1).mean(axis=0)
    brow_l = points[np.array([(336,285),(296,295),(334,282),(293,283),(300,276)])].mean(axis=1).mean(axis=0)
    origin = (brow_r + brow_l) / 2
    if np.dot(points[152] - origin, v) < 0:
        v = -v
    width = abs(np.dot(points[454] - points[234], u))
    height = np.dot(points[152] - origin, v)
    if width <= 20 or height <= 20:
        raise ValueError('Face width or brow-to-chin height <= 20 pixels')
    # Rotation and uniform scale, with the subject's right eye at (57,82).
    rotation = np.array([[u[0], u[1]], [-u[1], u[0]]]) * (64 / distance)
    translation = np.array([57., 82.]) - rotation @ right
    matrix = np.column_stack([rotation, translation])
    image = cv2.warpAffine(rgb, matrix, (178, 218), flags=cv2.INTER_LINEAR)
    valid = cv2.warpAffine(np.ones(rgb.shape[:2], dtype=np.uint8), matrix,
                           (178, 218), flags=cv2.INTER_NEAREST)
    aligned = points @ rotation.T + translation
    return image, valid, aligned, matrix, dict(roll_degrees=roll, width=float(width),
                                               height=float(height))


def main():
    output = ROOT / 'artifacts' / 'preprocess'
    output.mkdir(parents=True, exist_ok=True)
    model = ROOT / 'models' / 'face_landmarker.task'
    if not model.is_file():
        raise FileNotFoundError(f'Missing official model: {model}')
    config = dict(num_faces=2, min_face_detection_confidence=.5,
                  min_face_presence_confidence=.5)
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE, **config)
    connections = mp.tasks.vision.FaceLandmarksConnections
    metadata = dict(model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
                    config=config, packages={p: importlib.metadata.version(p) for p in
                    ['mediapipe', 'numpy', 'Pillow', 'opencv-python']})
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    with (ROOT / 'artifacts/inventory/review.csv').open(encoding='utf-8-sig') as file:
        sample = list(csv.DictReader(file))
    results = []
    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as detector:
        for row in sample:
            image_id = row['image_id']
            record = dict(image_id=image_id, path=row['path'], status='ok')
            try:
                with Image.open(ROOT / row['path']) as source:
                    rgb = np.array(source.convert('RGB'))
            except OSError as exc:
                record.update(status='unreadable', error=str(exc))
                results.append(record)
                continue
            detected = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            count = len(detected.face_landmarks)
            if count != 1:
                record['status'] = 'no_face' if count == 0 else 'multiple_faces'
                results.append(record)
                continue
            points = np.array([[p.x * rgb.shape[1], p.y * rgb.shape[0]]
                               for p in detected.face_landmarks[0][:468]], dtype=np.float64)
            np.save(output / f'{image_id}_landmarks.npy', points)
            try:
                aligned, mask, landmarks, matrix, geometry = align(rgb, points, connections)
            except ValueError as exc:
                record.update(status='invalid_geometry', error=str(exc))
                results.append(record)
                continue
            Image.fromarray(aligned).save(output / f'{image_id}_aligned.png')
            Image.fromarray(mask * 255).save(output / f'{image_id}_valid.png')
            overlay = aligned.copy()
            for edge in connections.FACE_LANDMARKS_CONTOURS:
                a, b = np.rint(landmarks[[edge.start, edge.end]]).astype(int)
                cv2.line(overlay, tuple(a), tuple(b), (0, 255, 0), 1)
            for index in [234,454,172,397,127,356,152,0,17]:
                p = tuple(np.rint(landmarks[index]).astype(int))
                cv2.circle(overlay, p, 2, (255, 0, 0), -1)
                cv2.putText(overlay, str(index), p, cv2.FONT_HERSHEY_SIMPLEX, .25, (255,255,0), 1)
            Image.fromarray(overlay).save(output / f'{image_id}_overlay.png')
            np.savez(output / f'{image_id}_alignment.npz', matrix=matrix, landmarks=landmarks)
            record.update(geometry)
            results.append(record)
    summary = dict(total=len(results), statuses={status: sum(r['status'] == status for r in results)
                   for status in sorted({r['status'] for r in results})}, results=results,
                   review_status='landmark_overlay_only_roi_not_implemented')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k != 'results'}, indent=2))


if __name__ == '__main__':
    main()
