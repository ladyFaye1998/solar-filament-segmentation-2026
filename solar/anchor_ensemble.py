"""Combine corresponding contours while preserving every 0.55 seed identity."""
import numpy as np
from pycocotools import mask as coco_mask

from solar.blend import blend_mask
from solar.metrics import encode


def combine(base, refined, weight=.5,preserve_original_ties=False):
    assert len(base) == len(refined), 'Refinement must preserve seed identities'
    if weight == 1.:
        return list(base)
    shape = (2048, 2048)
    owner = np.full(shape, -1, dtype=np.int16)
    for i, rle in enumerate(base):
        mask = coco_mask.decode(rle).astype(bool)
        assert mask.shape == shape and mask.any()
        assert (owner[mask] < 0).all(), 'Base instances must be disjoint'
        owner[mask] = i
    occupied = np.zeros(shape, bool)
    output = []
    for i, (original, learned) in enumerate(zip(base, refined)):
        assert coco_mask.iou([original], [learned], [0])[0, 0] >= .2, 'Refiner seed ordering changed'
        proposal = blend_mask(original, learned, weight,preserve_original_ties=preserve_original_ties)
        mask = coco_mask.decode(proposal).astype(bool)
        # Keep a neighboring seed's original pixels reserved for that identity.
        mask &= ((owner < 0) | (owner == i)) & ~occupied
        if mask.sum() < 5:
            mask = coco_mask.decode(original).astype(bool)
            assert not (mask & occupied).any()
        occupied |= mask
        output.append(encode(mask))
    assert len(output) == len(base)
    return output
