"""Test a frozen learned boundary rule on independently trained YOLO seeds."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from solar.crop_refiner import build_model, infer, refine_instances
from solar.metrics import audit, counts, load_ground_truth, summarize
from solar.yolo_run import write_csv


def paired_report(stems, per_image):
    dates = sorted({stem[:8] for stem in stems})
    grouped = np.array([per_image[[s[:8] == date for s in stems]].sum(0) for date in dates])
    rng = np.random.default_rng(20261022)
    boot = grouped[rng.integers(0, len(dates), (3000, len(dates)))].sum(1)
    pq = boot[:, :, 0] / np.maximum(boot[:, :, 1] + .5 * boot[:, :, 2] + .5 * boot[:, :, 3], 1e-12)
    scores = [summarize(total) for total in per_image.sum(0)]
    return {
        'baseline': scores[0], 'learned_refinement': scores[1],
        'delta_pq': scores[1]['pq'] - scores[0]['pq'],
        'paired_date_bootstrap_95_interval': np.quantile(pq[:, 1] - pq[:, 0], [.025, .975]).tolist(),
        'physical_images': len(stems), 'capture_dates': len(dates),
        'caveat': 'Frozen rule transferred from RF-DETR. Historically reused development cohort; conditional uncertainty is not selection-adjusted.',
    }


def run(root, source, checkpoint, encoder, output):
    import cv2
    import torch
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    started = time.monotonic()
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert state['encoder'] == encoder
    model = build_model(encoder).cuda()
    model.load_state_dict(state['model'], strict=True)
    output.mkdir(parents=True, exist_ok=True)
    protocol = {
        'encoder': encoder, 'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'threshold': .5, 'prior_weight': 0., 'rule_selected_on': 'Earlier held-out RF-DETR proposals',
        'primary_encoder': 'timm-efficientnet-b0', 'secondary_encoder': 'resnet34 sensitivity only',
        'new_training': False, 'test_labels_used': False,
    }
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    results = []
    for variant in ['medoid', 'all-annotators']:
        destination = output / variant
        destination.mkdir(parents=True, exist_ok=True)
        seeds = json.loads((source / variant / 'val-predictions.json').read_text())
        stems = sorted(seeds)
        assert len(stems) == 132
        truth = load_ground_truth(root / 'train/MAGFiLO_1.0_Annotations_kaggle2026_train.json', set(stems))
        by_stem = defaultdict(list)
        for record in truth.values():
            by_stem[record['stem']].append(record['rles'])
        infer(model, seeds, root / 'train/train_images', destination / 'maps/val')
        per_image = np.zeros((len(stems), 2, 4))
        refined = {}
        for i, stem in enumerate(stems):
            with np.load(destination / 'maps/val' / f'{stem}.npz') as z:
                refined[stem] = refine_instances(seeds[stem], z['probability'], z['matrix'], .5, 0.)
            for j, prediction in enumerate([seeds[stem], refined[stem]]):
                for record in by_stem[stem]:
                    per_image[i, j] += counts(record, prediction)
        report = {'variant': variant, **protocol, **paired_report(stems, per_image)}
        previous = json.loads((source / variant / 'validation.json').read_text())['results'][0]['pq']
        assert abs(previous - report['baseline']['pq']) < 1e-10
        (destination / 'validation.json').write_text(json.dumps(report, indent=2))
        (destination / 'val-predictions.json').write_text(json.dumps(refined))
        np.savez_compressed(destination / 'per-image-counts.npz', counts=per_image, stems=np.array(stems))
        results.append(report)
        print(json.dumps(report), flush=True)
        # Preserve test inference only for rules that improve this detector.
        # Public evaluation and the comparison to the native correction remain separate.
        if report['delta_pq'] <= 0:
            continue
        tests = json.loads((source / variant / 'test-predictions.json').read_text())
        expected = {p.stem for p in (root / 'test/test_images').glob('*.jpeg')}
        assert len(expected) == 180 and set(tests) == expected
        infer(model, tests, root / 'test/test_images', destination / 'maps/test')
        predictions = {}
        for stem, rles in tests.items():
            with np.load(destination / 'maps/test' / f'{stem}.npz') as z:
                predictions[stem] = refine_instances(rles, z['probability'], z['matrix'], .5, 0.)
        (destination / 'test-predictions.json').write_text(json.dumps(predictions))
        (destination / 'test-audit.json').write_text(json.dumps(audit(predictions, expected), indent=2))
        write_csv(predictions, destination / 'submission.csv')
    complete = {'results': results, 'elapsed_seconds': time.monotonic() - started,
                'new_public_score': None, 'hdjojo_checkpoint_reproduced': False}
    (output / 'complete.json').write_text(json.dumps(complete, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['root', 'source', 'checkpoint', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--encoder', required=True)
    args = parser.parse_args()
    run(args.root, args.source, args.checkpoint, args.encoder, args.output)
