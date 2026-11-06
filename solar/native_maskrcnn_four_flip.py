"""Four-flip inference of the owned native Mask R-CNN with fixed fusion.

Identity, horizontal, vertical and both-axis flips use the same frozen CNN.
Candidates are restored to original pixels before anchor-IoU association.
"""
from pathlib import Path
import hashlib
import time

import numpy as np
from solar.native_four_flip_geometry import flip_rle, fuse, exclusive_bank
from solar.native_maskrcnn_evaluation import torch_decode
from solar.metrics import audit

VIEWS = ('identity', 'horizontal', 'vertical', 'both')


def tensor_view(value, view):
    import torch
    assert value.ndim == 3 and value.shape[0] == 3 and 0 <= view < 4
    axes = ([2] if view & 1 else []) + ([1] if view & 2 else [])
    return torch.flip(value, axes) if axes else value


def infer_four_flips(model, image_root, stems, device='cuda:0', score_cutoff=.5,
                     capture_input_hashes=False, progress=None):
    import torch
    from PIL import Image
    from solar.rfdetr_checkpoint import state_digest
    assert not model.training and not model.roi_heads.training
    assert len(stems) == len(set(stems)) and .05 <= score_cutoff <= 1
    initial = state_digest(model.state_dict()); root = Path(image_root).resolve()
    primary, fused_banks, restored_views, identity_raw, inspection = {}, {}, {}, {}, {}
    forward_calls = 0
    for number, stem in enumerate(stems):
        started = time.monotonic()
        path = (root / (stem + '.jpeg')).resolve(); assert path.parent == root
        with Image.open(path) as image:
            assert image.size == (2048, 2048)
            rgb = np.asarray(image.convert('RGB')).copy()
        value = torch.from_numpy(rgb).permute(2, 0, 1).float().to(device) / 255.
        views, hashes = [], []
        for view in range(4):
            transformed = tensor_view(value, view)
            if capture_input_hashes:
                # Hash observed input pixels after reversing normalization;
                # rounding recovers exact uint8 values, independent of float
                # division implementation on CPU versus CUDA.
                observed = torch.round(transformed * 255).to(torch.uint8).permute(1, 2, 0).contiguous().cpu().numpy()
                hashes.append(hashlib.sha256(observed.tobytes()).hexdigest())
                del observed
            with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.float16,
                    enabled=str(device).startswith('cuda')):
                predicted = model([transformed])[0]
            _, raw, _ = torch_decode(predicted, score_cutoff=.05)
            if view == 0:
                identity_raw[stem] = raw
            restored = []
            for row in raw:
                inverse = flip_rle(row['rle'], view)
                assert flip_rle(inverse, view) == row['rle']
                restored.append(dict(score=row['score'], rle=inverse,
                    detector_index=row['detector_index'], source_view=view))
            views.append(restored)
            forward_calls += 1
            del predicted, transformed
        del value
        fused = fuse(views, shape=(2048, 2048), maximum=100)
        owned = exclusive_bank(fused, shape=(2048, 2048), minimum_area=5)
        primary[stem] = [row['rle'] for row in owned if row['score'] >= score_cutoff]
        fused_banks[stem] = fused; restored_views[stem] = views
        inspection[stem] = dict(view_candidates=[len(rows) for rows in views],
            fused_candidates=len(fused), primary_candidates=len(primary[stem]),
            input_pixel_sha256_by_view=hashes if capture_input_hashes else None,
            elapsed_seconds=time.monotonic() - started)
        if progress:
            progress(number + 1, len(stems), stem, len(primary[stem]))
    assert state_digest(model.state_dict()) == initial
    assert forward_calls == 4 * len(stems)
    return primary, fused_banks, restored_views, identity_raw, dict(
        neural_state_unchanged=True, actual_neural_forward_calls=forward_calls,
        views=list(VIEWS), native_shape=[2048, 2048], score_cutoff=score_cutoff,
        min_native_area=5, top_fused_candidates=100, anchor_matching_IoU=.5,
        fusion='Highest-score anchor; one detection per view; majority mask with anchor ties; sum member confidences/4.',
        fixed_confidence_ownership_before_threshold=True,
        per_photo=inspection, audit=audit(primary, set(stems)))
