"""Image-conditioned, orientation-normalized refinement of real instance seeds."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as coco_mask
from scipy import ndimage as ndi

from solar.metrics import audit, counts, encode, load_ground_truth, polygon_rle, summarize
from solar.yolo_run import write_csv


WIDTH, HEIGHT = 512, 128


def frame_statistics(gray):
    yy, xx = np.ogrid[:2048, :2048]
    values = gray[(yy - 1023.5) ** 2 + (xx - 1023.5) ** 2 <= 950 ** 2].astype(np.float32)
    return float(values.mean()), max(1., float(values.std()))


def crop_matrix(mask):
    yy, xx = np.nonzero(mask)
    assert len(xx)
    points = np.column_stack([xx, yy]).astype(np.float64)
    centre = points.mean(0)
    centred = points - centre
    _, basis = np.linalg.eigh(centred.T @ centred / len(points))
    u = basis[:, -1]
    if u[0] < 0:
        u = -u
    v = np.array([-u[1], u[0]])
    along, across = centred @ u, centred @ v
    centre += u * (along.min() + along.max()) / 2 + v * (across.min() + across.max()) / 2
    length = max(96., 1.25 * float(np.ptp(along)) + 32.)
    width = max(64., 1.4 * float(np.ptp(across)) + 32.)
    matrix = np.zeros((2, 3), np.float64)
    matrix[0, :2] = u * (WIDTH - 1) / length
    matrix[1, :2] = v * (HEIGHT - 1) / width
    matrix[:, 2] = np.array([(WIDTH - 1) / 2, (HEIGHT - 1) / 2]) - matrix[:, :2] @ centre
    return matrix


def crop_inputs(gray, seed, matrix, mean, std):
    image = cv2.warpAffine(gray, matrix, (WIDTH, HEIGHT), flags=cv2.INTER_LINEAR, borderValue=mean)
    prior = cv2.warpAffine(seed.astype(np.uint8), matrix, (WIDTH, HEIGHT), flags=cv2.INTER_NEAREST)
    inputs = np.stack([np.clip((image.astype(np.float32) - mean) / std, -8, 8), prior.astype(np.float32)])
    return inputs


def prepare_training(root, partition, selections, output):
    output.mkdir(parents=True, exist_ok=True)
    split = json.loads(partition.read_text())
    stems = set(split['training_ids'])
    validation = set(split['validation_ids'])
    assert not stems & validation
    assert not {s[:8] for s in stems} & {s[:8] for s in validation}
    selected = {r['stem']: r['annotator_image'] for r in json.loads(selections.read_text()) if r['split'] == 'train'}
    annotation_path = root / 'train/MAGFiLO_1.0_Annotations_kaggle2026_train.json'
    data = json.loads(annotation_path.read_text())
    records = defaultdict(list)
    image_ids = {}
    for image in data['images']:
        stem = Path(image['file_name']).stem
        if stem in stems:
            image_ids[image['id']] = stem
            records[stem].append(image['id'])
    annotations = defaultdict(list)
    for annotation in data['annotations']:
        if annotation['image_id'] in image_ids:
            annotations[annotation['image_id']].append(polygon_rle(annotation))
    manifest = []
    for i, stem in enumerate(sorted(stems), 1):
        gray = cv2.imread(str(root / 'train/train_images' / f'{stem}.jpeg'), cv2.IMREAD_GRAYSCALE)
        mean, std = frame_statistics(gray)
        for j, seed_rle in enumerate(annotations[selected[stem]]):
            path = output / f'{stem}_{j:03d}.npz'
            matched = []
            for image_id in records[stem]:
                others = annotations[image_id]
                if others:
                    ious = coco_mask.iou([seed_rle], others, [0] * len(others))[0]
                    best = int(np.argmax(ious))
                    if ious[best] > .15:
                        matched.append((image_id, others[best]))
            assert matched
            if not path.exists():
                seed = coco_mask.decode(seed_rle)
                matrix = crop_matrix(seed)
                inputs = crop_inputs(gray, seed, matrix, mean, std)
                target = np.mean([cv2.warpAffine(coco_mask.decode(r).astype(np.float32), matrix,
                                                (WIDTH, HEIGHT), flags=cv2.INTER_LINEAR)
                                  for _, r in matched], axis=0)
                np.savez_compressed(path, image=inputs[0].astype(np.float16),
                                    seed=inputs[1].astype(np.uint8),
                                    target=np.rint(target * 255).astype(np.uint8))
            manifest.append({'stem': stem, 'file': path.name, 'anchor_annotator': selected[stem],
                             'matched_annotators': [image_id for image_id, _ in matched]})
        if i % 40 == 0 or i == len(stems):
            print(json.dumps({'event': 'refiner-training-crops', 'done': i, 'total': len(stems)}), flush=True)
    report = {'physical_training_images': len(stems), 'instance_crops': len(manifest),
              'target': 'Mean morphology across matched official expert instances, conditional on presence',
              'annotation_sha256': hashlib.sha256(annotation_path.read_bytes()).hexdigest(),
              'partition_sha256': hashlib.sha256(partition.read_bytes()).hexdigest(),
              'test_or_validation_labels_used_for_training': False}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    (output / 'audit.json').write_text(json.dumps(report, indent=2))
    return report


def build_model(encoder, pretrained=False, in_channels=2):
    import segmentation_models_pytorch as smp
    return smp.Unet(encoder_name=encoder, encoder_weights='imagenet' if pretrained else None,
                    in_channels=in_channels, classes=1, decoder_channels=(128, 64, 32, 16, 8))


def initialize_from_checkpoint(model, checkpoint, encoder, append_zero_channel=False):
    """Strictly preserve the original model when adding one optional input."""
    assert checkpoint['encoder'] == encoder
    state = dict(checkpoint['model'])
    if append_zero_channel:
        current = model.state_dict()
        changed = [key for key in current if current[key].shape != state[key].shape]
        assert len(changed) == 1, 'Only the first input convolution may change shape'
        key = changed[0]
        old, new = state[key], current[key]
        assert old.ndim == new.ndim == 4 and old.shape[1] == 2 and new.shape[1] == 3
        assert old.shape[0] == new.shape[0] and old.shape[2:] == new.shape[2:]
        expanded = old.new_zeros(new.shape)
        expanded[:, :2] = old
        state[key] = expanded
    model.load_state_dict(state, strict=True)


def train(cache, output, encoder, steps=6000, seconds_limit=2700,
          initial=None, lr_max=.00015, seed=20261021, extra_cache=None, extra_enabled=True):
    import torch
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, Dataset
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    files = sorted(cache.glob('*.npz'))
    assert len(files) > 1000

    class Crops(Dataset):
        def __len__(self):
            return steps * 8

        def __getitem__(self, index):
            path = random.choice(files)
            with np.load(path) as z:
                gray, seed, target = z['image'].astype(np.float32), z['seed'], z['target'].astype(np.float32) / 255.
            if extra_cache is not None:
                with np.load(extra_cache / path.name) as z:
                    extra = z['fits'].astype(np.float32)
            # Imperfect seed masks simulate detector boundaries without using
            # any hidden-test labels or pseudo-label supervision.
            factor = random.choice([2, 4, 8])
            seed = cv2.resize(cv2.resize(seed, (WIDTH // factor, HEIGHT // factor), interpolation=cv2.INTER_NEAREST),
                              (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)
            radius = random.choice([-2, -1, 0, 1, 2])
            kernel = np.ones((2 * abs(radius) + 1, 2 * abs(radius) + 1), np.uint8)
            seed = cv2.dilate(seed, kernel) if radius > 0 else cv2.erode(seed, kernel) if radius < 0 else seed
            shift = np.array([[1, 0, random.randint(-5, 5)], [0, 1, random.randint(-3, 3)]], np.float32)
            seed = cv2.warpAffine(seed, shift, (WIDTH, HEIGHT), flags=cv2.INTER_NEAREST).astype(np.float32)
            if random.random() < .1:
                seed[:] = 0
            gain, offset = random.uniform(.85, 1.15), random.uniform(-.1, .1)
            gray = np.clip(gray * gain + offset, -8, 8)
            layers = np.stack([gray, seed, target])
            if extra_cache is not None:
                extra = np.clip(extra * gain + offset, -8, 8) if extra_enabled else np.zeros_like(extra)
                layers = np.concatenate([layers, extra[None]], axis=0)
            if random.random() < .5:
                layers = layers[:, :, ::-1]
            if random.random() < .5:
                layers = layers[:, ::-1, :]
            inputs = layers[[0, 1, 3]] if extra_cache is not None else layers[:2]
            return torch.from_numpy(inputs.copy()), torch.from_numpy(layers[2:3].copy())

    output.mkdir(parents=True, exist_ok=True)
    model = build_model(encoder, pretrained=initial is None, in_channels=3 if extra_cache is not None else 2).cuda()
    if initial is not None:
        checkpoint = torch.load(initial, map_location='cpu', weights_only=True)
        initialize_from_checkpoint(model, checkpoint, encoder, append_zero_channel=extra_cache is not None)
    ema = copy.deepcopy(model).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_max, weight_decay=.0001)
    scaler = torch.amp.GradScaler('cuda')
    loader = DataLoader(Crops(), batch_size=8, num_workers=2, pin_memory=True, persistent_workers=True)
    config = {'encoder': encoder, 'crop_shape': [HEIGHT, WIDTH], 'input_channels': ['normalized_gray', 'detector_seed'],
              'planned_steps': steps, 'seconds_limit': seconds_limit, 'target': 'conditional matched-expert soft mask',
              'seed': seed, 'lr_max': lr_max,
              'initial_checkpoint_sha256': hashlib.sha256(Path(initial).read_bytes()).hexdigest() if initial is not None else None,
              'data_audit': json.loads((cache / 'audit.json').read_text())}
    if extra_cache is not None:
        config.update(input_channels=['normalized_gray', 'detector_seed', 'original_fits' if extra_enabled else 'zero_control'],
                      appended_channel_initial_weights='Exactly zero; original logits preserved',
                      extra_channel_enabled=extra_enabled)
    (output / 'config.json').write_text(json.dumps(config, indent=2))
    started, losses, failures, successful, attempted = time.monotonic(), [], 0, 0, 0
    for step, (images, targets) in enumerate(loader, 1):
        if step > 1 and time.monotonic() - started >= seconds_limit:
            break
        attempted = step
        model.train()
        images, targets = images.cuda(non_blocking=True), targets.cuda(non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        lr = lr_max * min(1., step / 150) * (.1 + .9 * (1 + math.cos(math.pi * min(step / steps, 1))) / 2)
        for group in optimizer.param_groups:
            group['lr'] = lr
        with torch.autocast('cuda', dtype=torch.float16):
            logits = model(images)
        logits = logits.float()
        probability = logits.sigmoid()
        dice = 1 - (2 * (probability * targets).sum((1, 2, 3)) + 1) / (probability.sum((1, 2, 3)) + targets.sum((1, 2, 3)) + 1)
        loss = .5 * F.binary_cross_entropy_with_logits(logits, targets) + .5 * dice.mean()
        if not torch.isfinite(loss):
            failures += 1
            assert failures <= 3, 'Repeated nonfinite refiner loss'
            continue
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= previous_scale:
            successful += 1
            with torch.no_grad():
                for averaged, current in zip(ema.state_dict().values(), model.state_dict().values()):
                    if averaged.is_floating_point():
                        averaged.mul_(.995).add_(current, alpha=.005)
                    else:
                        averaged.copy_(current)
        losses.append(float(loss.detach()))
        if step % 200 == 0:
            row = {'step': step, 'successful_optimizer_updates': successful, 'loss': float(np.mean(losses)),
                   'elapsed_seconds': time.monotonic() - started}
            losses.clear()
            (output / 'progress.json').write_text(json.dumps(row, indent=2))
            print(json.dumps(row), flush=True)
        if step % 1000 == 0:
            torch.save({'encoder': encoder, 'model': ema.state_dict(), 'step': step}, output / 'latest.pt')
    torch.save({'encoder': encoder, 'model': ema.state_dict(), 'step': attempted,
                'successful_optimizer_updates': successful}, output / 'latest.pt')
    assert successful >= 100, 'Too few successful updates'
    return ema


def infer(model, seeds, images, output, extra_cache=None, extra_enabled=True, *, use_autocast=True):
    import torch
    output.mkdir(parents=True, exist_ok=True)
    model.eval()
    device = next(model.parameters()).device
    with torch.inference_mode():
        for i, (stem, rles) in enumerate(sorted(seeds.items()), 1):
            gray = cv2.imread(str(images / f'{stem}.jpeg'), cv2.IMREAD_GRAYSCALE)
            mean, std = frame_statistics(gray)
            if extra_cache is not None:
                with np.load(extra_cache / f'{stem}.npz') as z:
                    signal = z['fits'].astype(np.float32)
                if not extra_enabled:
                    signal[:] = 0
            matrices, probabilities = [], []
            for rle in rles:
                seed = coco_mask.decode(rle)
                matrix = crop_matrix(seed)
                inputs = crop_inputs(gray, seed, matrix, mean, std)
                if extra_cache is not None:
                    extra = cv2.warpAffine(signal, matrix, (WIDTH, HEIGHT), flags=cv2.INTER_LINEAR)
                    inputs = np.concatenate([inputs, extra[None]], axis=0)
                x = torch.from_numpy(inputs[None]).to(device)
                total = torch.zeros((1, 1, HEIGHT, WIDTH), device=device)
                for axes in [(), (2,), (3,), (2, 3)]:
                    view = torch.flip(x, axes) if axes else x
                    with torch.autocast(device.type, dtype=torch.float16,
                                        enabled=device.type == 'cuda' and use_autocast):
                        result = model(view).float().sigmoid()
                    total += torch.flip(result, axes) if axes else result
                probabilities.append((total[0, 0] / 4 * 255).round().clamp(0, 255).byte().cpu().numpy())
                matrices.append(matrix)
            np.savez_compressed(output / f'{stem}.npz',
                                probability=np.asarray(probabilities, dtype=np.uint8).reshape(-1, HEIGHT, WIDTH),
                                matrix=np.asarray(matrices, dtype=np.float64).reshape(-1, 2, 3))
            if i % 20 == 0 or i == len(seeds):
                print(json.dumps({'event': 'crop-refiner-inference', 'split': output.name, 'done': i, 'total': len(seeds)}), flush=True)


def refine_instances(seeds, probabilities, matrices, threshold=.45, prior_weight=0.):
    assert len(seeds) == len(probabilities) == len(matrices)
    occupied = np.zeros((2048, 2048), bool)
    output = []
    corners = np.array([[0, 0, 1], [WIDTH - 1, 0, 1], [WIDTH - 1, HEIGHT - 1, 1], [0, HEIGHT - 1, 1]])
    for rle, probability, matrix in zip(seeds, probabilities, matrices):
        inverse = cv2.invertAffineTransform(matrix)
        native_corners = corners @ inverse.T
        x0, y0 = np.maximum(0, np.floor(native_corners.min(0)).astype(int))
        x1, y1 = np.minimum(2048, np.ceil(native_corners.max(0)).astype(int) + 1)
        assert x1 > x0 and y1 > y0
        original = coco_mask.decode(rle)[y0:y1, x0:x1].astype(bool)
        value = probability.astype(np.float32) / 255
        if prior_weight:
            prior = cv2.warpAffine(coco_mask.decode(rle), matrix, (WIDTH, HEIGHT), flags=cv2.INTER_LINEAR)
            prior = cv2.GaussianBlur(prior.astype(np.float32), (5, 5), .8)
            value = (1 - prior_weight) * value + prior_weight * prior
        inverse[:, 2] -= [x0, y0]
        proposed = cv2.warpAffine(value, inverse, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR) > threshold
        proposed &= cv2.dilate(original.astype(np.uint8), np.ones((49, 49), np.uint8)).astype(bool)
        labels, n = ndi.label(proposed, np.ones((3, 3)))
        retained = np.unique(labels[original])
        retained = retained[retained != 0]
        proposed = np.isin(labels, retained) if len(retained) else np.zeros_like(original)
        proposed = ndi.binary_fill_holes(proposed)
        ratio = proposed.sum() / max(1, original.sum())
        if not .5 <= ratio <= 1.8:
            proposed = original
        proposed &= ~occupied[y0:y1, x0:x1]
        if proposed.sum() < 5:
            continue
        occupied[y0:y1, x0:x1] |= proposed
        mask = np.zeros((2048, 2048), np.uint8)
        mask[y0:y1, x0:x1] = proposed
        output.append(encode(mask))
    return output


def select_and_export(root, partition, shared, output, encoders):
    val_seeds = json.loads((shared / 'val-seeds.json').read_text())
    test_seeds = json.loads((shared / 'test-seeds.json').read_text())
    assert set(val_seeds) == set(json.loads(partition.read_text())['validation_ids'])
    truth = load_ground_truth(root / 'train/MAGFiLO_1.0_Annotations_kaggle2026_train.json', set(val_seeds))
    by_stem = defaultdict(list)
    for record in truth.values():
        by_stem[record['stem']].append(record['rles'])
    settings = [{'model': name, 'threshold': threshold, 'prior_weight': prior}
                for name in encoders + ['ensemble'] for threshold in [.4, .5] for prior in [0., .25]]
    settings.insert(0, {'model': 'identity'})
    totals = np.zeros((len(settings), 4))
    for stem, seeds in val_seeds.items():
        maps = {}
        for encoder in encoders:
            with np.load(output / encoder / 'maps/val' / f'{stem}.npz') as z:
                maps[encoder] = z['probability']
                matrices = z['matrix']
        maps['ensemble'] = np.rint(np.mean(list(maps.values()), axis=0)).astype(np.uint8)
        for j, setting in enumerate(settings):
            prediction = seeds if setting['model'] == 'identity' else refine_instances(
                seeds, maps[setting['model']], matrices, setting['threshold'], setting['prior_weight'])
            for gt in by_stem[stem]:
                totals[j] += counts(gt, prediction)
    results = [{'setting': setting, **summarize(total)} for setting, total in zip(settings, totals)]
    assert abs(results[0]['pq'] - .36360168073588384) < 1e-10
    results.sort(key=lambda row: -row['pq'])
    report = {'results': results, 'validation_detector': 'Our date-held-out RF-DETR, confidence 0.5',
              'test_detector': 'HDJoJo public v2 CSV, public PQ 0.55; inaccessible private weights',
              'transfer_caveat': 'Held-out refinement gain on RF-DETR does not establish a gain on the different HDJoJo detector. Exact YOLO validation is required before promotion.',
              'development_selection_only': True}
    (output / 'validation.json').write_text(json.dumps(report, indent=2))
    chosen = results[0]['setting']
    predictions = {}
    for stem, seeds in test_seeds.items():
        if chosen['model'] == 'identity':
            predictions[stem] = seeds
            continue
        requested = encoders if chosen['model'] == 'ensemble' else [chosen['model']]
        maps = []
        for encoder in requested:
            with np.load(output / encoder / 'maps/test' / f'{stem}.npz') as z:
                maps.append(z['probability'])
                matrices = z['matrix']
        probability = np.rint(np.mean(maps, axis=0)).astype(np.uint8)
        predictions[stem] = refine_instances(seeds, probability, matrices, chosen['threshold'], chosen['prior_weight'])
    (output / 'test-predictions.json').write_text(json.dumps(predictions))
    (output / 'test-audit.json').write_text(json.dumps(audit(predictions, set(test_seeds)), indent=2))
    write_csv(predictions, output / 'candidate-unvalidated-transfer.csv')
    (output / 'complete.json').write_text(json.dumps({'best_development': results[0],
        'base_public_score': .55, 'candidate_public_score': None, 'independent_base_reproduction': False,
        'promotion_requires_exact_yolo_validation': True}, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--shared', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--encoder', required=True)
    args = p.parse_args()
    model = train(args.shared / 'training-crops', args.output, args.encoder)
    for subset in ['val', 'test']:
        seeds = json.loads((args.shared / f'{subset}-seeds.json').read_text())
        image_dir = args.root / ('train/train_images' if subset == 'val' else 'test/test_images')
        infer(model, seeds, image_dir, args.output / 'maps' / subset)
    (args.output / 'complete.json').write_text(json.dumps({'encoder': args.encoder, 'inference_complete': True}))
