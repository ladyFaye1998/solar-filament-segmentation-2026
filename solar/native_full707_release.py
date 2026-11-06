"""Separate full707 inference-weight schema; never claim174 heldout exclusion."""
import hashlib
import json
from pathlib import Path

MODEL_SOURCE = 'solar/torchvision_native_maskrcnn.py'
ARCHITECTURE = dict(resolution=2048, num_classes=2, mask_ROI_pool_size=56,
                    mask_output_size=112, sampled_ROIs=128)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def export_full707_checkpoint(checkpoint_path, recipe_path, destination, expected_sha256):
    import torch
    from solar.rfdetr_checkpoint import state_digest
    assert digest(checkpoint_path) == expected_sha256
    recipe = json.loads(Path(recipe_path).read_text())
    assert recipe['schema'] == 'owned_native_MaskRCNN_full707_training_recipe_v1'
    source = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    assert source['schema'] == 'owned_native_MaskRCNN_V2_full707_training_v1'
    assert source['recipe_sha256'] == digest(recipe_path)
    assert source['source_hashes'] == recipe['frozen_sources']
    assert source['index_sha256'] == recipe['index_sha256']
    assert source['total_optimizer_updates'] == 10605
    assert source['training_photos'] == 707 and source['training_expert_records'] == 1154
    assert source['excluded_validation_photos'] == 0
    step = source['successful_optimizer_steps']
    assert 0 < step <= 10605 and source['completed_microbatches'] == 4 * step
    assert state_digest(source['model']) == source['model_state_sha256']
    assert all(isinstance(value, torch.Tensor) for value in source['model'].values())
    assert all(bool(torch.isfinite(value).all()) for value in source['model'].values() if value.is_floating_point())
    assert digest(MODEL_SOURCE) == recipe['frozen_sources'][MODEL_SOURCE]
    payload = dict(schema='owned_native_MaskRCNN_full707_inference_v1', model=source['model'],
        model_state_sha256=source['model_state_sha256'], architecture=ARCHITECTURE,
        model_source_sha256=recipe['frozen_sources'][MODEL_SOURCE],
        source_checkpoint_sha256=expected_sha256, recipe_sha256=source['recipe_sha256'],
        training_index_sha256=source['index_sha256'], successful_optimizer_steps=step,
        target_optimizer_steps=10605, is_fixed_final_checkpoint=step == 10605,
        training_photos=707, training_expert_records=1154, excluded_validation_photos=0,
        independent_validation_available=False, previous174_validation_labels_now_TRAIN=True,
        initialization=dict(url=recipe['pretrained_url'], sha256=recipe['pretrained_checkpoint_sha256']),
        validation_score=None, public_score=None, ground_truth_or_answers_included=False)
    destination = Path(destination)
    assert not destination.exists()
    torch.save(payload, destination)
    return dict(schema=payload['schema'], file_sha256=digest(destination), bytes=destination.stat().st_size,
        model_state_sha256=source['model_state_sha256'], source_checkpoint_sha256=expected_sha256,
        successful_optimizer_steps=step, is_fixed_final_checkpoint=step == 10605,
        training_photos=707, excluded_validation_photos=0, independent_validation_available=False)


def load_full707_weights(path, expected_sha256, device='cuda:0', require_fixed_final=True):
    import torch
    from solar.rfdetr_checkpoint import state_digest
    from solar.torchvision_native_maskrcnn import build_model
    assert digest(path) == expected_sha256
    payload = torch.load(path, map_location='cpu', weights_only=True)
    assert payload['schema'] == 'owned_native_MaskRCNN_full707_inference_v1'
    assert payload['architecture'] == ARCHITECTURE
    assert payload['model_source_sha256'] == digest(MODEL_SOURCE)
    assert payload['training_photos'] == 707 and payload['training_expert_records'] == 1154
    assert payload['excluded_validation_photos'] == 0 and not payload['independent_validation_available']
    assert payload['previous174_validation_labels_now_TRAIN'] and not payload['ground_truth_or_answers_included']
    assert payload['target_optimizer_steps'] == 10605 and 0 < payload['successful_optimizer_steps'] <= 10605
    assert payload['is_fixed_final_checkpoint'] == (payload['successful_optimizer_steps'] == 10605)
    if require_fixed_final:
        assert payload['is_fixed_final_checkpoint'], 'Full707 software-proof weights are not final10605.'
    assert state_digest(payload['model']) == payload['model_state_sha256']
    model, _ = build_model(pretrained=False, resolution=2048, sampled_rois=128)
    model.load_state_dict(payload['model'], strict=True)
    assert state_digest(model.state_dict()) == payload['model_state_sha256']
    metadata = {key: value for key, value in payload.items() if key != 'model'}
    del payload
    return model.to(device).eval(), metadata
