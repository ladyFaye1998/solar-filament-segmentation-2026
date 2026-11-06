"""Reconstruct the 0.55 notebook's YOLOv8l architecture with auditable training.

This is an independent architecture reconstruction, not the inaccessible HDJoJo
checkpoint. All solar supervision comes from official competition annotations.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools import mask as coco_mask

from solar.metrics import audit, counts, encode, load_ground_truth, summarize


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare(root, partition, selections, output, policy):
    annotation_path = root / 'train/MAGFiLO_1.0_Annotations_kaggle2026_train.json'
    data = json.loads(annotation_path.read_text())
    split = json.loads(partition.read_text())
    train, val = set(split['training_ids']), set(split['validation_ids'])
    assert not {s[:8] for s in train} & {s[:8] for s in val}
    medoids = {r['stem']: r['annotator_image'] for r in json.loads(selections.read_text())}
    by_image = defaultdict(list)
    for annotation in data['annotations']:
        by_image[annotation['image_id']].append(annotation)
    manifest = []
    for image in data['images']:
        stem = Path(image['file_name']).stem
        if stem not in train | val:
            continue
        subset = 'train' if stem in train else 'val'
        # Internal YOLO AP uses one held-out annotator. Final PQ below always
        # scores every held-out annotation record, including misses.
        if (subset == 'val' or policy == 'medoid') and image['id'] != medoids[stem]:
            continue
        name = f'{image["id"]}.jpeg' if subset == 'train' and policy == 'all-annotators' else image['file_name']
        target = output / 'images' / subset / name
        target.parent.mkdir(parents=True, exist_ok=True)
        original = root / 'train/train_images' / image['file_name']
        if not target.exists():
            try:
                target.symlink_to(original.resolve())
            except OSError:
                os.link(original, target)
        lines = []
        for annotation in by_image[image['id']]:
            polygons = annotation['segmentation']
            assert isinstance(polygons, list) and len(polygons) == 1
            points = np.asarray(polygons[0], dtype=float).reshape(-1, 2)
            assert len(points) >= 3
            points /= [image['width'], image['height']]
            assert np.isfinite(points).all() and ((points >= 0) & (points <= 1)).all()
            lines.append('0 ' + ' '.join(f'{x:.8f}' for x in points.ravel()))
        label = output / 'labels' / subset / Path(name).with_suffix('.txt')
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='ascii')
        manifest.append({'stem': stem, 'annotator_image': image['id'], 'split': subset,
                         'instances': len(lines), 'label_sha256': digest(label)})
    assert {r['stem'] for r in manifest if r['split'] == 'train'} == train
    assert {r['stem'] for r in manifest if r['split'] == 'val'} == val
    output.mkdir(parents=True, exist_ok=True)
    config = output / 'data.yaml'
    config.write_text(f'path: {output.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: filament\n')
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    report = {'target_policy': policy, 'training_records': sum(r['split'] == 'train' for r in manifest),
              'physical_training_images': len(train), 'physical_validation_images': len(val),
              'annotation_sha256': digest(annotation_path), 'partition_sha256': digest(partition),
              'date_disjoint': True, 'test_images_in_training': False,
              'training_instances': sum(r['instances'] for r in manifest if r['split'] == 'train')}
    (output / 'audit.json').write_text(json.dumps(report, indent=2))
    return config, report


def select_instances(candidates, confidence=.3, box_iou=0., minimum_area=5):
    """Single-class box NMS followed by confidence-ordered mask ownership."""
    kept_boxes, output = [], []
    occupied = np.zeros((2048, 2048), dtype=bool)
    for candidate in sorted(candidates, key=lambda row: -row['score']):
        if candidate['score'] < confidence:
            continue
        box = np.asarray(candidate['box'], dtype=float)
        suppress = False
        for other in kept_boxes:
            wh = np.maximum(0, np.minimum(box[2:], other[2:]) - np.maximum(box[:2], other[:2]))
            intersection = float(wh.prod())
            union = float(np.maximum(0, box[2:] - box[:2]).prod() +
                          np.maximum(0, other[2:] - other[:2]).prod() - intersection)
            if intersection / max(union, 1e-9) > box_iou:
                suppress = True
                break
        if suppress:
            continue
        kept_boxes.append(box)
        mask = coco_mask.decode(candidate['rle']).astype(bool) & ~occupied
        if int(mask.sum()) < minimum_area:
            continue
        occupied |= mask
        output.append(encode(mask))
    return output


def write_csv(predictions, path):
    import csv
    with Path(path).open('w', newline='', encoding='ascii') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['filament_id', 'segmentation_rle'])
        for stem, rles in sorted(predictions.items()):
            for i, rle in enumerate(rles, 1):
                writer.writerow([f'{stem}_{i}', rle['counts']])


def infer(checkpoint, root, partition, output):
    import torch
    from ultralytics import YOLO
    model = YOLO(str(checkpoint))
    split = json.loads(partition.read_text())
    truth = load_ground_truth(root / 'train/MAGFiLO_1.0_Annotations_kaggle2026_train.json', set(split['validation_ids']))
    gt_by_stem = defaultdict(list)
    for record in truth.values():
        gt_by_stem[record['stem']].append(record['rles'])
    settings = [{'confidence': c, 'box_iou': iou, 'minimum_area': area}
                for c in [.2, .3, .4, .5] for iou in [0., .3, .6] for area in [5, 200]]
    validation = None
    for name, directory, stems in [
        ('val', root / 'train/train_images', sorted(split['validation_ids'])),
        ('test', root / 'test/test_images', sorted(p.stem for p in (root / 'test/test_images').glob('*.jpeg')))]:
        banks = {}
        for i, stem in enumerate(stems, 1):
            banks[stem] = {}
            # NMS is not nested across IoU thresholds: a proposal suppressed
            # at a loose threshold can survive a tighter threshold after its
            # suppressor is removed. Run each actual NMS policy independently.
            for nms_iou in ([0., .3, .6] if name == 'val' else [chosen['box_iou']]):
                result = model.predict(str(directory / f'{stem}.jpeg'), imgsz=2048, conf=.05,
                                       iou=nms_iou, max_det=100, device=0, verbose=False,
                                       retina_masks=False)[0]
                rows = []
                if result.masks is not None:
                    masks = result.masks.data.cpu().numpy()
                    scores = result.boxes.conf.cpu().numpy()
                    boxes = result.boxes.xyxy.cpu().numpy()
                    for mask, score, box in zip(masks, scores, boxes):
                        assert mask.shape == (2048, 2048)
                        if mask.any():
                            rows.append({'rle': encode(mask), 'score': float(score), 'box': box.tolist()})
                banks[stem][str(nms_iou)] = rows
            if i % 20 == 0 or i == len(stems):
                print(json.dumps({'event': 'yolo-inference', 'split': name, 'done': i, 'total': len(stems)}), flush=True)
        if name == 'val':
            (output / 'val-nms-candidates.json').write_text(json.dumps(banks))
            totals = np.zeros((len(settings), 4))
            for stem, bank in banks.items():
                for j, setting in enumerate(settings):
                    predictions = select_instances(bank[str(setting['box_iou'])],
                        confidence=setting['confidence'], box_iou=1., minimum_area=setting['minimum_area'])
                    for gt in gt_by_stem[stem]:
                        totals[j] += counts(gt, predictions)
            validation = [{'setting': setting, **summarize(total)} for setting, total in zip(settings, totals)]
            validation.sort(key=lambda row: -row['pq'])
            chosen = validation[0]['setting']
            (output / 'validation.json').write_text(json.dumps({
                'physical_images': len(stems), 'annotator_records': len(truth),
                'development_selection_only': True, 'results': validation,
                'caveat': 'The inherited development cohort has prior selection exposure; these 24 settings do not constitute an independent final test.'}, indent=2))
        candidates = {stem: bank[str(chosen['box_iou'])] for stem, bank in banks.items()}
        (output / f'{name}-candidates.json').write_text(json.dumps(candidates))
        predictions = {stem: select_instances(rows, confidence=chosen['confidence'], box_iou=1.,
                        minimum_area=chosen['minimum_area']) for stem, rows in candidates.items()}
        (output / f'{name}-predictions.json').write_text(json.dumps(predictions))
        (output / f'{name}-audit.json').write_text(json.dumps(audit(predictions, set(stems)), indent=2))
        if name == 'test':
            write_csv(predictions, output / 'submission.csv')
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return validation[0]


def run(root, partition, selections, output, policy, initial, warmup_hours=1.25, native_hours=.75):
    import torch
    from ultralytics import YOLO, settings
    settings.update({'wandb': False, 'mlflow': False, 'clearml': False,
                     'comet': False, 'tensorboard': False, 'sync': False})
    torch.set_num_threads(2)
    output.mkdir(parents=True, exist_ok=True)
    config, data_audit = prepare(root, partition, selections, output / 'dataset', policy)
    common = dict(data=str(config), device=0, batch=1, nbs=16, workers=2, cache=False,
                  optimizer='AdamW', weight_decay=.0005, cos_lr=True, patience=1000,
                  single_cls=True, overlap_mask=False, mask_ratio=4, amp=True,
                  seed=20261021, deterministic=True, hsv_h=0., hsv_s=0., hsv_v=.15,
                  degrees=180., translate=.05, scale=.15, shear=0., perspective=0.,
                  flipud=.5, fliplr=.5, mosaic=0., mixup=0., copy_paste=0.,
                  close_mosaic=0, plots=False, save=True, save_period=-1,
                  project=str((output / 'training').resolve()), exist_ok=True, verbose=False)
    protocol = {'architecture': 'yolov8l-seg', 'ultralytics': importlib.metadata.version('ultralytics'),
                'torch': torch.__version__, 'initial_checkpoint_sha256': digest(initial),
                'initial_checkpoint': 'Public Ultralytics COCO yolov8l-seg.pt; AGPL-3.0',
                'source_architecture_reference': 'hdjojo/solar-filament-seg-inference version 2 (0.55)',
                'independent_training_not_exact_checkpoint_reproduction': True,
                'supervision': 'Official competition annotations only; no test pseudo-labels',
                'dataset': data_audit, 'training_args': common,
                'stages': [{'resolution': 1536, 'epochs_nominal': 40, 'hours_max': warmup_hours, 'lr': .001},
                           {'resolution': 2048, 'epochs_nominal': 12, 'hours_max': native_hours, 'lr': .0003}],
                'time_limit_note': 'Ultralytics time takes precedence over the nominal epoch schedule; actual completed epochs are recorded.'}
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    started = time.monotonic()
    checkpoint = initial
    actual_stages = []
    for stage_index, stage in enumerate(protocol['stages']):
        stage_name = f'stage{stage_index + 1}_{stage["resolution"]}'
        model = YOLO(str(checkpoint))

        def progress(trainer):
            row = {'stage': stage_name, 'epoch_completed': trainer.epoch + 1,
                   'seconds_total': time.monotonic() - started,
                   'metrics': {str(k): float(v) for k, v in trainer.metrics.items()}}
            (output / 'progress.json').write_text(json.dumps(row, indent=2))
            print(json.dumps({'event': 'yolo-epoch', **row}), flush=True)

        model.add_callback('on_fit_epoch_end', progress)
        model.train(**common, name=stage_name, imgsz=stage['resolution'], epochs=stage['epochs_nominal'],
                    time=stage['hours_max'], lr0=stage['lr'], lrf=.1,
                    warmup_epochs=3. if stage_index == 0 else .5)
        checkpoint = output / 'training' / stage_name / 'weights/last.pt'
        assert checkpoint.is_file()
        actual_stages.append({'name': stage_name, 'checkpoint_sha256': digest(checkpoint),
                              'progress': json.loads((output / 'progress.json').read_text())})
        del model
        gc.collect()
        torch.cuda.empty_cache()
    result = infer(checkpoint, root, partition, output)
    complete = {'elapsed_seconds': time.monotonic() - started, 'checkpoint_sha256': digest(checkpoint),
                'actual_stages': actual_stages, 'best_development': result,
                'submission_sha256': digest(output / 'submission.csv')}
    (output / 'complete.json').write_text(json.dumps(complete, indent=2))
    print(json.dumps(complete, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--partition', type=Path, default=Path('configs/partition.json'))
    p.add_argument('--selections', type=Path, default=Path('configs/target-selection.json'))
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--policy', required=True, choices=['medoid', 'all-annotators'])
    p.add_argument('--initial', required=True, type=Path)
    p.add_argument('--warmup-hours', type=float, default=1.25)
    p.add_argument('--native-hours', type=float, default=.75)
    args = p.parse_args()
    run(args.root, args.partition, args.selections, args.output, args.policy, args.initial,
        args.warmup_hours, args.native_hours)
