"""Held-out mask-voting ensemble of independently trained YOLOv8l-seg runs (same 565/132 split, official per-record PQ).

Per photo, candidates (score >= 0.05, box NMS IoU 0 per model) from K models are clustered greedily in score order:
a candidate joins the highest-scoring existing cluster whose seed mask has IoU >= `link` with it and that holds no member
from the same model. Cluster score = sum of member scores / K (absent models count 0). Fused mask = score-weighted
pixel vote of the members >= 0.5. Clusters above a score threshold are made pixel-disjoint in score order (min 5 px).
"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np
from pycocotools import mask as mu
from solar.metrics import encode, polygon_rle
from solar.yolo_run import select_instances


def clusters(banks, link):
    pool = [(r['score'], k, r['rle']) for k, rows in enumerate(banks) for r in rows]
    pool.sort(key=lambda x: -x[0]); out = []
    if not pool:
        return out
    rles = [p[2] for p in pool]; iou = np.asarray(mu.iou(rles, rles, [0] * len(rles))).reshape(len(rles), len(rles))
    for i, (s, k, r) in enumerate(pool):
        for c in out:
            if k not in c['models'] and iou[i, c['seed']] >= link:
                c['members'].append(i); c['models'].add(k); break
        else:
            out.append(dict(seed=i, members=[i], models={k}))
    for c in out:
        c['scores'] = [pool[i][0] for i in c['members']]; c['rles'] = [pool[i][2] for i in c['members']]
    return out


def fuse_dense(c, K):
    w = np.asarray(c['scores'], np.float32)
    if len(w) == 1:
        return c['rles'][0], float(w.sum() / K)
    acc = None
    for wi, r in zip(w, c['rles']):
        m = mu.decode(r).astype(np.float32) * wi; acc = m if acc is None else acc + m
    return encode((acc / w.sum() >= .5).astype(np.uint8)), float(w.sum() / K)


def fuse(c, K):
    """Use exact sparse voting for small ensembles, with a bounded dense fallback."""
    if len(c['scores']) <= 12:
        from solar.sparse_vote import weighted_rle_vote
        return weighted_rle_vote(c, K)
    return fuse_dense(c, K)


def main(paths, out, link):
    data = json.loads((ROOT / 'data/raw/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json').read_text())
    by = defaultdict(list)
    for a in data['annotations']:
        by[a['image_id']].append(polygon_rle(a))
    banks = [json.loads(Path(p).read_text()) for p in paths]; K = len(banks)
    stems = sorted(banks[0]); assert all(set(b) == set(stems) for b in banks)
    recs = defaultdict(list)
    for im in data['images']:
        s = Path(im['file_name']).stem
        if s in banks[0]:
            recs[s].append(by[im['id']])
    def counts(g, P):
        if not P or not g:
            return np.array([0, 0, len(P), len(g)], float)
        iou = np.asarray(mu.iou(g, P, [0] * len(P))).reshape(len(g), len(P)); h = iou > .5
        return np.array([iou[h].sum(), h.sum(), (~h.any(0)).sum(), (~h.any(1)).sum()])
    grid = [.1, .15, .2, .25, .3, .35, .4, .5]
    per = {t: [] for t in grid}; single = {i: [] for i in range(K)}
    for n, s in enumerate(stems, 1):
        cl = clusters([b[s] for b in banks], link)
        fused = [dict(zip(('rle', 'score'), fuse(c, K))) for c in cl]
        boxes = []
        for f in fused:
            x, y, w, h = mu.toBbox(f['rle']); f['box'] = [float(x), float(y), float(x + w), float(y + h)]
        for t in grid:
            P = select_instances(fused, confidence=t, box_iou=1., minimum_area=5); c = np.zeros(4)
            for g in recs[s]:
                c += counts(g, P)
            per[t].append(c)
        for i in range(K):
            P = select_instances(banks[i][s], confidence=.2, box_iou=1., minimum_area=5); c = np.zeros(4)
            for g in recs[s]:
                c += counts(g, P)
            single[i].append(c)
        if n % 20 == 0:
            print(json.dumps({'done': n}), flush=True)
    pq = lambda t: t[0] / (t[1] + .5 * t[2] + .5 * t[3])
    best_single = max(range(K), key=lambda i: pq(np.sum(single[i], 0)))
    ref = np.array(single[best_single]); rng = np.random.default_rng(0); idx = [rng.integers(0, len(stems), len(stems)) for _ in range(2000)]
    rep = {'single_at_0.2': {Path(paths[i]).parent.name: round(pq(np.sum(single[i], 0)), 4) for i in range(K)}, 'ensemble': {}}
    for t in grid:
        a = np.array(per[t]); d = [pq(a[j].sum(0)) - pq(ref[j].sum(0)) for j in idx]
        rep['ensemble'][str(t)] = dict(pq=round(pq(a.sum(0)), 4), delta_vs_best_single=round(pq(a.sum(0)) - pq(ref.sum(0)), 4),
                                       ci95=[round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)],
                                       tp_fp_fn=a.sum(0)[1:].astype(int).tolist())
    rep.update(models=[str(p) for p in paths], link=link, best_single=Path(paths[best_single]).parent.name)
    Path(out).write_text(json.dumps(rep, indent=2)); print(json.dumps(rep, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--bank', action='append', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--link', type=float, default=.5); a = ap.parse_args(); main(a.bank, a.out, a.link)
