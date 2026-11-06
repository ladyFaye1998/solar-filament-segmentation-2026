"""Check a downloaded public CSV without changing its bytes or implying a rerun."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from pycocotools import mask as coco_mask


def decoded_runs(encoded):
    runs, position = [], 0
    while position < len(encoded):
        value, shift = 0, 0
        while True:
            assert position < len(encoded), 'Truncated compressed RLE'
            code = ord(encoded[position]) - 48
            assert 0 <= code <= 63, 'Invalid compressed RLE character'
            position += 1
            value |= (code & 31) << shift
            shift += 5
            if not code & 32:
                if code & 16:
                    value |= -1 << shift
                break
        if len(runs) > 2:
            value += runs[-2]
        assert value >= 0, 'Negative run length'
        runs.append(value)
    return runs


def verify(csv_path, images, source):
    expected = {p.stem for p in images.glob('*.jpeg')}
    assert len(expected) == 180, 'Expected the official 180 test images'
    with csv_path.open(newline='', encoding='ascii') as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ['filament_id', 'segmentation_rle']
        rows = list(reader)
    assert rows and len({r['filament_id'] for r in rows}) == len(rows)
    grouped = defaultdict(list)
    for row in rows:
        assert None not in row and all(v is not None for v in row.values())
        stem, index = row['filament_id'].rsplit('_', 1)
        assert stem in expected and index.isdigit() and int(index) > 0
        grouped[stem].append(row)
    areas = []
    for stem, entries in grouped.items():
        occupied = np.zeros((2048, 2048), dtype=bool)
        for row in entries:
            assert sum(decoded_runs(row['segmentation_rle'])) == 2048 * 2048
            rle = {'size': [2048, 2048], 'counts': row['segmentation_rle']}
            mask = coco_mask.decode(rle).astype(bool)
            assert mask.any(), f'Empty instance: {row["filament_id"]}'
            assert not (mask & occupied).any(), f'Overlap: {row["filament_id"]}'
            encoded = coco_mask.encode(np.asfortranarray(mask.astype(np.uint8)))
            assert encoded['counts'].decode('ascii') == row['segmentation_rle']
            occupied |= mask
            areas.append(int(mask.sum()))
    return {
        'audited_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_notebook': source,
        'csv': str(csv_path),
        'csv_sha256': hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        'csv_bytes': csv_path.stat().st_size,
        'csv_rows': len(rows),
        'positive_images': len(grouped),
        'images_without_prediction_rows': sorted(expected - set(grouped)),
        'native_shape': [2048, 2048],
        'native_run_lengths_checked': True,
        'canonical_rle_roundtrip': True,
        'all_instances_positive': True,
        'unique_ids': True,
        'zero_instance_overlap': True,
        'area_percentiles': dict(zip(['min', 'p25', 'median', 'p75', 'max'],
                                     np.percentile(areas, [0, 25, 50, 75, 100]).tolist())),
        'independent_inference_reproduction': False,
        'provenance_limit': 'CSV integrity is verified; this audit alone cannot establish training-data provenance or final prize eligibility.'
    }


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv', required=True, type=Path)
    p.add_argument('--source', required=True)
    p.add_argument('--images', type=Path, default=Path('data/raw/MAGFiLO_1.0_Kaggle_2026/test/test_images'))
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    report = verify(args.csv, args.images, args.source)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
