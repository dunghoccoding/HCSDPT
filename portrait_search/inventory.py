"""Reproducible dataset inventory and stratified visual review sample."""

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


ROOT = Path(__file__).resolve().parents[1]


def inventory(dataset: Path, output: Path, sample_size: int = 80, seed: int = 42):
    dataset = dataset.resolve()
    dataset.relative_to(ROOT)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    hashes = defaultdict(list)
    pixel_hashes = defaultdict(list)
    for path in sorted(dataset.rglob('*')):
        if not path.is_file():
            continue
        row = dict(image_id=len(rows) + 1, path=path.relative_to(ROOT).as_posix(),
                   group='celeba' if 'celeba' in path.name else
                   'male_faces_strict' if 'male_faces_strict' in path.name else 'other',
                   sha256='', pixel_sha256='', width='', height='', mode='',
                   format='', status='ok', error='')
        try:
            row['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            hashes[row['sha256']].append(row['path'])
            with Image.open(path) as image:
                image.load()
                row.update(width=image.width, height=image.height,
                           mode=image.mode, format=image.format)
                rgb = image.convert('RGB')
                payload = f'{rgb.width}x{rgb.height}:RGB:'.encode() + rgb.tobytes()
                row['pixel_sha256'] = hashlib.sha256(payload).hexdigest()
                pixel_hashes[row['pixel_sha256']].append(row['path'])
        except (OSError, ValueError) as exc:
            row.update(status='unreadable', error=str(exc))
        rows.append(row)
    fields = ['image_id', 'path', 'group', 'sha256', 'pixel_sha256', 'width',
              'height', 'mode', 'format', 'status', 'error']
    with (output / 'manifest.csv').open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    valid = [row for row in rows if row['status'] == 'ok']
    groups = defaultdict(list)
    for row in valid:
        groups[row['group']].append(row)
    rng = random.Random(seed)
    target = min(sample_size, len(valid))
    quotas = {key: target * len(value) // max(1, len(valid)) for key, value in groups.items()}
    remaining = target - sum(quotas.values())
    order = sorted(groups, key=lambda key: (-(target * len(groups[key]) % max(1, len(valid))), key))
    for key in order[:remaining]:
        quotas[key] += 1
    sample = []
    for key in sorted(groups):
        sample.extend(rng.sample(groups[key], quotas[key]))
    sample.sort(key=lambda row: row['image_id'])
    with (output / 'review.csv').open('w', encoding='utf-8-sig', newline='') as file:
        review_fields = ['image_id', 'path', 'group', 'pose', 'expression', 'glasses',
                         'facial_hair', 'lighting', 'landmarks_ok', 'roi_ok', 'notes']
        writer = csv.DictWriter(file, fieldnames=review_fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(sample)
    for start in range(0, len(sample), 40):
        page = sample[start:start + 40]
        sheet = Image.new('RGB', (5 * 190, ((len(page) + 4) // 5) * 245), 'white')
        draw = ImageDraw.Draw(sheet)
        for index, row in enumerate(page):
            x, y = (index % 5) * 190, (index // 5) * 245
            with Image.open(ROOT / row['path']) as image:
                thumb = ImageOps.contain(image.convert('RGB'), (178, 218))
                sheet.paste(thumb, (x + 6, y))
            draw.text((x + 6, y + 220), f"ID {row['image_id']} {row['group']}", fill='black')
        sheet.save(output / f'review_{start // 40 + 1:02d}.jpg', quality=90)
    summary = dict(total_files=len(rows), readable=len(valid),
                   statuses=dict(Counter(row['status'] for row in rows)),
                   groups=dict(Counter(row['group'] for row in rows)),
                   dimensions=dict(Counter(f"{row['width']}x{row['height']}" for row in valid)),
                   modes=dict(Counter(row['mode'] for row in valid)),
                   formats=dict(Counter(row['format'] for row in valid)),
                   duplicate_file_groups=[paths for paths in hashes.values() if len(paths) > 1],
                   duplicate_pixel_groups=[paths for paths in pixel_hashes.values() if len(paths) > 1],
                   sample_size=len(sample), sample_groups=quotas, seed=seed,
                   review_status='pending_manual_review')
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'data_best')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts' / 'inventory')
    parser.add_argument('--sample-size', type=int, default=80)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.sample_size < 0:
        parser.error('--sample-size must be nonnegative')
    if not args.dataset.is_dir():
        parser.error('Dataset directory does not exist')
    summary = inventory(args.dataset, args.output, args.sample_size, args.seed)
    concise = {key: value for key, value in summary.items() if not key.startswith('duplicate_')}
    concise['duplicate_file_groups'] = len(summary['duplicate_file_groups'])
    concise['duplicate_pixel_groups'] = len(summary['duplicate_pixel_groups'])
    concise['unique_pixel_contents'] = summary['readable'] - sum(
        len(paths) - 1 for paths in summary['duplicate_pixel_groups'])
    print(json.dumps(concise, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
