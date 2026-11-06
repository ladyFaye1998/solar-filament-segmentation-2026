"""Deployment YOLOv8l-seg: all 707 training photos (+ optional GONG neighbour frames) with the validated long recipe.

There is no independent validation for this model; the operating point (confidence, box NMS IoU, minimum area) is
taken from the held-out selection of the 565-photo counterpart and passed explicitly. A small monitoring set of
training photos only feeds Ultralytics' internal metrics and is NOT an evaluation. Writes scored test candidate banks
(for later fusion research) and an audited 180-image CSV. Official training annotations only; test images are used
for inference only.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import json
import os
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from solar.metrics import audit, encode
from solar.yolo_run import digest, select_instances, write_csv
from solar.external_data_policy import ExternalFrameGuard, test_ids_from_root

STAMP = lambda s: dt.datetime.strptime(s[:14], '%Y%m%d%H%M%S')


def link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        try:
            dst.symlink_to(Path(src).resolve())
        except OSError:
            shutil.copyfile(src, dst)


def prepare_full(root, output, neighbour_dir=None, max_seconds=150, monitor=15, *, quality_excluded=None):
    if neighbour_dir and quality_excluded is None:
        raise ValueError('External label transfer requires an explicit reviewed quality-exclusion list')
    quality_excluded = set(quality_excluded or ())
    data = json.loads((root / 'train/MAGFiLO_1.0_Annotations_kaggle2026_train.json').read_text())
    by_image = defaultdict(list)
    for a in data['annotations']:
        by_image[a['image_id']].append(a)
    nb_by_site = defaultdict(list)
    if neighbour_dir:
        for p in Path(neighbour_dir).glob('*.jpeg'):
            nb_by_site[p.stem[14]].append(p.stem)
    records, added = 0, []
    stems = sorted({Path(i['file_name']).stem for i in data['images']})
    guard = ExternalFrameGuard(frozenset(stems), frozenset(), frozenset(test_ids_from_root(root))) if neighbour_dir else None
    rejected = {}
    monitor_stems = set(stems[::max(1, len(stems) // monitor)][:monitor])
    for image in data['images']:
        stem = Path(image['file_name']).stem
        lines = []
        for a in by_image[image['id']]:
            pts = np.asarray(a['segmentation'][0], dtype=float).reshape(-1, 2) / [image['width'], image['height']]
            assert len(pts) >= 3 and np.isfinite(pts).all() and ((pts >= 0) & (pts <= 1)).all()
            lines.append('0 ' + ' '.join(f'{x:.8f}' for x in pts.ravel()))
        body = '\n'.join(lines) + ('\n' if lines else '')
        name = f"{image['id']}.jpeg"
        link(root / 'train/train_images' / image['file_name'], output / 'images/train' / name)
        (output / 'labels/train').mkdir(parents=True, exist_ok=True)
        (output / 'labels/train' / f"{image['id']}.txt").write_text(body, encoding='ascii'); records += 1
        if stem in monitor_stems and not (output / 'images/val' / f'{stem}.jpeg').exists():
            link(root / 'train/train_images' / image['file_name'], output / 'images/val' / f'{stem}.jpeg')
            (output / 'labels/val').mkdir(parents=True, exist_ok=True)
            (output / 'labels/val' / f'{stem}.txt').write_text(body, encoding='ascii')
        for n in nb_by_site.get(stem[14], []):
            gap = abs((STAMP(n) - STAMP(stem)).total_seconds())
            if n == stem or gap > max_seconds:
                continue
            reason = guard.rejection(n, stem, max_seconds=max_seconds)
            if reason is None and n in quality_excluded:
                reason = 'preexisting_quality_exclusion'
            if reason:
                rejected[n] = reason
                continue
            nname = f"{image['id']}__{n}.jpeg"
            link(Path(neighbour_dir) / f'{n}.jpeg', output / 'images/train' / nname)
            (output / 'labels/train' / f"{image['id']}__{n}.txt").write_text(body, encoding='ascii')
            added.append(dict(neighbour=n, source_record=image['id'], seconds=gap))
    config = output / 'data.yaml'
    config.write_text(f'path: {output.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: filament\n')
    report = dict(training_photos=len(stems), training_records=records, neighbour_images_added=len(added),
                  distinct_neighbour_frames=len({a['neighbour'] for a in added}), monitor_photos_not_independent=len(monitor_stems),
                  temporal_policy=guard.metadata() if guard else None, rejected_external_frames=rejected)
    (output / 'audit.json').write_text(json.dumps(report, indent=2))
    (output / 'neighbour-manifest.json').write_text(json.dumps(added))
    return config, report


def run(root, output, initial, neighbour_dir, epochs, hours, lr0, confidence, box_iou, minimum_area):
    import torch
    from ultralytics import YOLO, settings
    settings.update({'wandb': False, 'mlflow': False, 'clearml': False, 'comet': False, 'tensorboard': False, 'sync': False})
    torch.set_num_threads(2)
    output.mkdir(parents=True, exist_ok=True)
    quality_excluded = None
    if neighbour_dir:
        quality_path = Path(__file__).resolve().parents[1] / 'configs/gong-neighbour-quality-20261026.json'
        quality_excluded = json.loads(quality_path.read_text())['excluded']
    config, data_audit = prepare_full(root, output / 'dataset', neighbour_dir,
                                     quality_excluded=quality_excluded)
    if neighbour_dir:
        data_audit['quality_exclusion_sha256'] = digest(quality_path)
        data_audit['reviewed_quality_exclusion_frames'] = len(quality_excluded)
    print(json.dumps({'event': 'dataset', **data_audit}), flush=True)
    common = dict(data=str(config), device=0, batch=1, nbs=16, workers=2, cache=False, optimizer='AdamW',
                  weight_decay=.0005, cos_lr=True, patience=1000, single_cls=True, overlap_mask=False, mask_ratio=4,
                  amp=True, seed=20261025, deterministic=True, hsv_h=0., hsv_s=0., hsv_v=.15, degrees=180.,
                  translate=.05, scale=.15, shear=0., perspective=0., flipud=.5, fliplr=.5, mosaic=0., mixup=0.,
                  copy_paste=0., close_mosaic=0, plots=False, save=True, save_period=1,
                  project=str((output / 'training').resolve()), exist_ok=True, verbose=False)
    (output / 'protocol.json').write_text(json.dumps(dict(dataset=data_audit, training_args=common, epochs=epochs, hours=hours,
        lr0=lr0, operating_point=dict(confidence=confidence, box_iou=box_iou, minimum_area=minimum_area),
        operating_point_source='565-photo counterpart held-out selection (no independent validation for this model)'), indent=2))
    started = time.monotonic(); model = YOLO(str(initial))

    def progress(trainer):
        row = {'epoch_completed': trainer.epoch + 1, 'seconds_total': time.monotonic() - started}
        (output / 'progress.json').write_text(json.dumps(row)); print(json.dumps({'event': 'yolo-full-epoch', **row}), flush=True)

    model.add_callback('on_fit_epoch_end', progress)
    model.train(**common, name='native2048', imgsz=2048, epochs=epochs, time=hours, lr0=lr0, lrf=.1, warmup_epochs=1.)
    checkpoint = output / 'training/native2048/weights/last.pt'; assert checkpoint.is_file()
    del model; gc.collect(); torch.cuda.empty_cache()
    model = YOLO(str(checkpoint)); test = root / 'test/test_images'
    stems = sorted(p.stem for p in test.glob('*.jpeg')); assert len(stems) == 180
    banks, preds = {}, {}
    for i, stem in enumerate(stems, 1):
        res = model.predict(str(test / f'{stem}.jpeg'), imgsz=2048, conf=.05, iou=box_iou, max_det=100, device=0,
                            verbose=False, retina_masks=False)[0]
        rows = []
        if res.masks is not None:
            for m, s, b in zip(res.masks.data.cpu().numpy(), res.boxes.conf.cpu().numpy(), res.boxes.xyxy.cpu().numpy()):
                if m.any():
                    rows.append({'rle': encode(m), 'score': float(s), 'box': b.tolist()})
        banks[stem] = rows
        preds[stem] = select_instances(rows, confidence=confidence, box_iou=1., minimum_area=minimum_area)
        if i % 30 == 0:
            print(json.dumps({'event': 'test-inference', 'done': i}), flush=True)
    (output / 'test-candidates.json').write_text(json.dumps(banks))
    (output / 'test-predictions.json').write_text(json.dumps(preds))
    write_csv(preds, output / 'submission.csv')
    report = audit(preds, set(stems))
    complete = dict(status='yolo_full707_trained_and_test_inferred', checkpoint_sha256=digest(checkpoint),
                    progress=json.loads((output / 'progress.json').read_text()), dataset=data_audit, test_audit=report,
                    submission_sha256=digest(output / 'submission.csv'), elapsed_seconds=time.monotonic() - started,
                    independent_validation=None, submissions_used=0)
    (output / 'complete.json').write_text(json.dumps(complete, indent=2)); print(json.dumps(complete, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True, type=Path); p.add_argument('--output', required=True, type=Path)
    p.add_argument('--initial', required=True, type=Path); p.add_argument('--neighbours', type=Path, default=None)
    p.add_argument('--epochs', type=int, default=24); p.add_argument('--hours', type=float, default=9.5)
    p.add_argument('--lr0', type=float, default=.001); p.add_argument('--confidence', type=float, default=.2)
    p.add_argument('--box-iou', type=float, default=0.); p.add_argument('--minimum-area', type=int, default=5)
    a = p.parse_args()
    run(a.root, a.output, a.initial, a.neighbours, a.epochs, a.hours, a.lr0, a.confidence, a.box_iou, a.minimum_area)
