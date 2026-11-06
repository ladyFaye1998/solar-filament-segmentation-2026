"""Owned, award-eligible test export: owned detector masks refined with the incumbent's contour-refiner structure.

Seeds come from an owned detector's audited test predictions (full707 Mask R-CNN four-flip, or an owned YOLO bank at a
fixed operating point). Every seed is refined exactly as the 0.56 incumbent refines HDJoJo's masks: probability maps of
the two full707 B0 refiners (seeds 20261022/20261023, averaged) and the 565-photo ResNet34 refiner, mixed .75/.25,
contours(seeds, mix, matrix, .5) (refine at 0.5, IoU<0.2 fallback, 50/50 signed-distance blend), then the 5% detached
island prune. Owned weights only; test images used for inference only. Writes predictions, CSV and an audit summary.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def run(seeds_path, refiners, root, output):
    import cv2, torch
    from solar.crop_refiner import build_model, infer
    from solar.incumbent_probability_cut import contours
    from solar.incumbent_component_pruning import prune
    from solar.metrics import audit
    from solar.yolo_run import digest, write_csv
    torch.set_num_threads(4); cv2.setNumThreads(2)
    output.mkdir(parents=True, exist_ok=True)
    seeds = json.loads(Path(seeds_path).read_text())
    test = root / 'test/test_images'; stems = sorted(p.stem for p in test.glob('*.jpeg')); assert len(stems) == 180
    seeds = {s: seeds.get(s, []) for s in stems}
    maps = {}
    for name, path in refiners.items():
        st = torch.load(path, map_location='cpu', weights_only=True)
        model = build_model(st.get('encoder', 'timm-efficientnet-b0'), pretrained=False); model.load_state_dict(st['model'] if 'model' in st else st)
        maps[name] = output / f'maps-{name}'; started = time.time()
        infer(model.eval(), {s: r for s, r in seeds.items() if r}, test, maps[name])
        print(json.dumps({'event': 'refiner-maps', 'refiner': name, 'seconds': round(time.time() - started)}), flush=True)
    preds = {}
    for s in stems:
        if not seeds[s]:
            preds[s] = []; continue
        loaded = {}
        for name, d in maps.items():
            with np.load(d / f'{s}.npz', allow_pickle=False) as z:
                loaded[name] = (z['probability'].astype(np.float32), z['matrix'])
        m = loaded['r34'][1]
        b0 = (loaded['b0a'][0] + loaded['b0b'][0]) / 2
        mixed = np.rint(.75 * np.rint(b0) + .25 * loaded['r34'][0]).astype(np.uint8)
        preds[s] = [prune(r) for r in contours(seeds[s], mixed, m, .5)]
        preds[s] = [{'size': list(r['size']), 'counts': r['counts'] if isinstance(r['counts'], str) else r['counts'].decode()} for r in preds[s]]
    (output / 'test-predictions.json').write_text(json.dumps(preds, separators=(',', ':')))
    write_csv(preds, output / 'submission.csv')
    summary = dict(seeds=str(seeds_path), seed_instances=sum(map(len, seeds.values())), refiners={k: str(v) for k, v in refiners.items()},
                   refiner_sha256={k: digest(v) for k, v in refiners.items()}, recipe='incumbent contour structure (.75 full707 B0 pair + .25 R34, t .5, 50/50 blend, 5% prune)',
                   test_audit=audit(preds, set(stems)), submission_sha256=digest(output / 'submission.csv'), instances=sum(map(len, preds.values())))
    (output / 'complete.json').write_text(json.dumps(summary, indent=2)); print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seeds', required=True, type=Path); p.add_argument('--root', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--b0a', required=True, type=Path); p.add_argument('--b0b', required=True, type=Path); p.add_argument('--r34', required=True, type=Path)
    a = p.parse_args()
    run(a.seeds, {'b0a': a.b0a, 'b0b': a.b0b, 'r34': a.r34}, a.root, a.output)
