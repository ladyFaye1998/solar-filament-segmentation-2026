"""Alternative YOLO label policy: per-photo union of all annotator records with duplicates merged ('union').

solar.yolo_run.prepare writes one training image per annotator record ('all-annotators'), so a filament drawn by one of
three annotators is a positive in one copy and background in the other two, and the detector learns to hedge on
low-agreement (mostly faint) filaments. 'union' writes ONE image per training photo whose label holds every filament any
annotator drew: polygons from different records whose masks overlap with IoU >= 0.5 are one physical filament and are
represented once, by the cluster medoid (highest mean IoU to the other members). Validation is unchanged (medoid record
for Ultralytics' internal metrics; the official per-record PQ in solar.yolo_run.infer scores every held-out record).
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools import mask as coco_mask

from solar.metrics import polygon_rle
from solar.yolo_run import digest


def link_or_copy(src, dst):
    """symlink, else hard link, else copy (drives without link support, e.g. exFAT, need the copy)."""
    import shutil
    try:
        dst.symlink_to(Path(src).resolve())
    except OSError:
        try:
            os.link(src, dst)
        except OSError:
            shutil.copyfile(src, dst)


def merge_records(records):
    """records: list of lists of (annotation, rle). Returns one representative annotation per physical filament."""
    items = [(k, a, r) for k, rec in enumerate(records) for a, r in rec]
    if not items:
        return [], 0
    rles = [r for _, _, r in items]
    iou = np.asarray(coco_mask.iou(rles, rles, [0] * len(rles))).reshape(len(rles), len(rles))
    parent = list(range(len(items)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if items[i][0] != items[j][0] and iou[i, j] >= .5:
                a, b = find(i), find(j)
                if a != b:
                    parent[b] = a
    groups = defaultdict(list)
    for i in range(len(items)):
        groups[find(i)].append(i)
    keep = []
    for members in groups.values():
        if len(members) == 1:
            keep.append(items[members[0]][1]); continue
        sub = iou[np.ix_(members, members)]; score = (sub.sum(1) - 1) / (len(members) - 1)
        keep.append(items[members[int(np.argmax(score))]][1])
    return keep, len(items) - len(keep)


def prepare_union(root, partition, selections, output):
    annotation_path = root / 'train/MAGFiLO_1.0_Annotations_kaggle2026_train.json'
    data = json.loads(annotation_path.read_text()); split = json.loads(partition.read_text())
    train, val = set(split['training_ids']), set(split['validation_ids'])
    assert not {s[:8] for s in train} & {s[:8] for s in val}
    monitor = set(sorted(train)[::max(1, len(train) // 15)][:15]) if not val else set()  # deployment: NOT an evaluation
    medoids = {r['stem']: r['annotator_image'] for r in json.loads(selections.read_text())}
    by_image = defaultdict(list)
    for a in data['annotations']:
        by_image[a['image_id']].append(a)
    photos = defaultdict(list)
    for image in data['images']:
        photos[Path(image['file_name']).stem].append(image)
    manifest, merged_total = [], 0
    for stem, images in sorted(photos.items()):
        if stem not in train | val:
            continue
        subset = 'train' if stem in train else 'val'
        if stem in monitor:
            m_name = images[0]['file_name']; m_target = output / 'images' / 'val' / m_name; m_target.parent.mkdir(parents=True, exist_ok=True)
            if not m_target.exists():
                link_or_copy(root / 'train/train_images' / m_name, m_target)
        if subset == 'val':
            chosen = [i for i in images if i['id'] == medoids[stem]]; anns = by_image[chosen[0]['id']]; merged = 0
        else:
            recs = [[(a, polygon_rle(a)) for a in by_image[i['id']]] for i in images]
            anns, merged = merge_records(recs); merged_total += merged
        image = images[0]; name = image['file_name']
        target = output / 'images' / subset / name; target.parent.mkdir(parents=True, exist_ok=True)
        original = root / 'train/train_images' / name
        if not target.exists():
            link_or_copy(original, target)
        lines = []
        for a in anns:
            pts = np.asarray(a['segmentation'][0], dtype=float).reshape(-1, 2) / [image['width'], image['height']]
            assert len(pts) >= 3 and np.isfinite(pts).all() and ((pts >= 0) & (pts <= 1)).all()
            lines.append('0 ' + ' '.join(f'{x:.8f}' for x in pts.ravel()))
        label = output / 'labels' / subset / Path(name).with_suffix('.txt'); label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='ascii')
        if stem in monitor:
            m_label = output / 'labels' / 'val' / Path(name).with_suffix('.txt'); m_label.parent.mkdir(parents=True, exist_ok=True)
            m_label.write_text(label.read_text(encoding='ascii'), encoding='ascii')
        manifest.append({'stem': stem, 'annotator_image': f'union-{stem}' if subset == 'train' else image['id'], 'split': subset,
                         'records': len(images), 'instances': len(lines), 'label_sha256': digest(label)})
    output.mkdir(parents=True, exist_ok=True)
    config = output / 'data.yaml'
    config.write_text(f'path: {output.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: filament\n')
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    report = {'target_policy': 'union', 'training_images': sum(r['split'] == 'train' for r in manifest),
              'training_instances': sum(r['instances'] for r in manifest if r['split'] == 'train'),
              'duplicates_merged': merged_total, 'physical_validation_images': len(val), 'monitor_photos_not_independent': len(monitor), 'date_disjoint': True,
              'annotation_sha256': digest(annotation_path), 'partition_sha256': digest(partition), 'test_images_in_training': False}
    (output / 'audit.json').write_text(json.dumps(report, indent=2))
    return config, report


def prepare_records_deploy(root, partition, selections, output):
    """Deployment counterpart of solar.yolo_run.prepare('all-annotators') when the partition has no held-out photos:
    one training image per annotator record (all 707 photos) plus 15 monitor photos (medoid record, NOT independent)."""
    annotation_path = root / 'train/MAGFiLO_1.0_Annotations_kaggle2026_train.json'
    data = json.loads(annotation_path.read_text()); split = json.loads(partition.read_text())
    train = set(split['training_ids']); assert not split['validation_ids']
    monitor = set(sorted(train)[::max(1, len(train) // 15)][:15])
    medoids = {r['stem']: r['annotator_image'] for r in json.loads(selections.read_text())}
    by_image = defaultdict(list)
    for a in data['annotations']:
        by_image[a['image_id']].append(a)
    manifest = []
    for image in data['images']:
        stem = Path(image['file_name']).stem
        if stem not in train:
            continue
        lines = []
        for a in by_image[image['id']]:
            pts = np.asarray(a['segmentation'][0], dtype=float).reshape(-1, 2) / [image['width'], image['height']]
            assert len(pts) >= 3 and np.isfinite(pts).all() and ((pts >= 0) & (pts <= 1)).all()
            lines.append('0 ' + ' '.join(f'{x:.8f}' for x in pts.ravel()))
        body = '\n'.join(lines) + ('\n' if lines else '')
        name = f"{image['id']}.jpeg"
        for subset, fname in [('train', name)] + ([('val', image['file_name'])] if stem in monitor and image['id'] == medoids[stem] else []):
            target = output / 'images' / subset / fname; target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                link_or_copy(root / 'train/train_images' / image['file_name'], target)
            label = output / 'labels' / subset / Path(fname).with_suffix('.txt'); label.parent.mkdir(parents=True, exist_ok=True)
            label.write_text(body, encoding='ascii')
        manifest.append({'stem': stem, 'annotator_image': image['id'], 'split': 'train', 'instances': len(lines)})
    output.mkdir(parents=True, exist_ok=True)
    config = output / 'data.yaml'
    config.write_text(f'path: {output.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: filament\n')
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    report = {'target_policy': 'all-annotators', 'training_records': len(manifest), 'physical_training_images': len(train),
              'monitor_photos_not_independent': len(monitor), 'training_instances': sum(r['instances'] for r in manifest),
              'annotation_sha256': digest(annotation_path), 'partition_sha256': digest(partition), 'test_images_in_training': False}
    (output / 'audit.json').write_text(json.dumps(report, indent=2))
    return config, report
