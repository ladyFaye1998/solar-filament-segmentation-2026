"""Compressed-RLE implementation of the organizer's all-pair PQ semantics."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools import mask as coco_mask


def encode(mask: np.ndarray) -> dict:
    rle = coco_mask.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle['counts'] = rle['counts'].decode('ascii')
    return rle


def polygon_rle(annotation: dict, h=2048, w=2048) -> dict:
    seg = annotation['segmentation']
    if isinstance(seg, list):
        rle = coco_mask.merge(coco_mask.frPyObjects(seg, h, w))
    elif isinstance(seg['counts'], list):
        rle = coco_mask.frPyObjects(seg, h, w)
    else:
        rle = dict(seg)
    if isinstance(rle['counts'], bytes):
        rle['counts'] = rle['counts'].decode('ascii')
    return rle


def counts(gt: list[dict], pred: list[dict]) -> np.ndarray:
    """Return IoU sum, TP, FP and FN, including empty observations.

    Every qualifying pair is counted, matching the published notebook. Do not
    silently replace these rules with Hungarian or greedy one-to-one matching.
    """
    if not gt or not pred:
        return np.asarray([0., 0., len(pred), len(gt)], dtype=np.float64)
    overlap = coco_mask.iou(gt, pred, [0] * len(pred))
    hit = overlap > 0.5
    return np.asarray([overlap[hit].sum(), hit.sum(), (~hit.any(axis=0)).sum(),
                       (~hit.any(axis=1)).sum()], dtype=np.float64)


def summarize(total: np.ndarray) -> dict:
    iou, tp, fp, fn = map(float, total)
    denominator = tp + .5 * fp + .5 * fn
    return {'pq': iou / denominator if denominator else 0.,
            'sq': iou / tp if tp else 0.,
            'rq': tp / denominator if denominator else 0.,
            'sum_iou': iou, 'tp': int(tp), 'fp': int(fp), 'fn': int(fn)}


def load_ground_truth(annotation_path: Path, stems: set[str]) -> dict[str, list[dict]]:
    data = json.loads(annotation_path.read_text(encoding='utf-8'))
    selected = {i['id']: i for i in data['images'] if Path(i['file_name']).stem in stems}
    annotations = defaultdict(list)
    for annotation in data['annotations']:
        if annotation['image_id'] in selected:
            annotations[annotation['image_id']].append(polygon_rle(annotation))
    return {image_id: {'stem': Path(image['file_name']).stem,
                       'rles': annotations[image_id]} for image_id, image in selected.items()}


def evaluate(gt: dict, predictions: dict[str, list[dict]]) -> dict:
    expected = {x['stem'] for x in gt.values()}
    if set(predictions) != expected:
        raise ValueError(f'Prediction coverage mismatch: {len(set(predictions)^expected)} observations')
    totals = np.zeros(4, dtype=np.float64)
    per_observation = []
    for observation, entry in sorted(gt.items()):
        row_counts = counts(entry['rles'], predictions[entry['stem']])
        totals += row_counts
        per_observation.append({'annotator_image': observation, 'stem': entry['stem'],
                                **summarize(row_counts)})
    return {**summarize(totals), 'physical_images': len(expected),
            'annotator_image_records': len(gt), 'observations': per_observation}


def audit(predictions: dict[str, list[dict]], expected: set[str], shape=(2048,2048)) -> dict:
    assert set(predictions) == expected, 'Every image needs an inference record, including negatives'
    n_instances = 0
    negatives = []
    for stem, rles in predictions.items():
        occupied = np.zeros(shape, dtype=bool)
        if not rles:
            negatives.append(stem)
        for rle in rles:
            mask = coco_mask.decode(rle).astype(bool)
            assert mask.shape == shape and mask.any(), f'Invalid mask in {stem}'
            assert not (occupied & mask).any(), f'Overlapping instances in {stem}'
            occupied |= mask
            n_instances += 1
    return {'inferred_images': len(expected), 'instances': n_instances,
            'negative_images': sorted(negatives), 'native_shape': list(shape),
            'nonempty_instances': True, 'disjoint_instances': True}
