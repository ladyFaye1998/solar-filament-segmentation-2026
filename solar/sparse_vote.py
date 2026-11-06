"""Exact float32 confidence-weighted mask voting using compressed COCO operations.

For small ensembles, a pixel wins precisely when it lies in at least one minimal
subset whose ordered float32 confidence sum reaches the original half-vote cut.
This avoids allocating a native-resolution float array for every member.
"""
import numpy as np
from pycocotools import mask as mu


def weighted_rle_vote(cluster, models):
    weights = np.asarray(cluster['scores'], np.float32)
    if not 1 <= len(weights) <= 12 or len(weights) != len(cluster['rles']):
        raise ValueError('Sparse subset voting supports 1 to 12 member masks')
    if models <= 0 or not np.isfinite(weights).all() or not (weights > 0).all():
        raise ValueError('Positive finite member weights and model count required')
    if len(weights) == 1:
        return cluster['rles'][0], float(weights.sum() / models)
    denominator = weights.sum()
    winning = []
    for bits in range(1, 1 << len(weights)):
        if any(bits & smaller == smaller for smaller in winning):
            continue
        value = np.float32(0)
        for index, weight in enumerate(weights):
            if bits & (1 << index):
                value = np.float32(value + weight)
        if np.float32(value / denominator) >= np.float32(.5):
            winning.append(bits)
    intersections = [mu.merge([rle for index, rle in enumerate(cluster['rles']) if bits & (1 << index)], intersect=True)
                     for bits in winning]
    rle = mu.merge(intersections, intersect=False)
    rle = dict(size=list(rle['size']), counts=rle['counts'].decode('ascii') if isinstance(rle['counts'], bytes) else rle['counts'])
    return rle, float(denominator / models)
