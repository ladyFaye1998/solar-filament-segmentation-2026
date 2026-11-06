"""Pinned RF-DETR EMA export and complete-state fingerprints."""
import hashlib
import json
from pathlib import Path


def state_digest(value):
    import torch
    result = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            result.update(json.dumps(['tensor', str(tensor.dtype), list(tensor.shape)]).encode())
            result.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            result.update(b'dict')
            for key in sorted(item, key=lambda k: (type(k).__name__, repr(k))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            result.update(type(item).__name__.encode())
            for child in item:
                visit(child)
        else:
            result.update(json.dumps([type(item).__name__, item], sort_keys=True).encode())
    visit(value)
    return result.hexdigest()


def export_final_ema(trainer, module, ema, path):
    import torch
    from rfdetr.training.callbacks.best_model import BestModelCallback
    weights = ema.get_ema_model_state_dict()
    assert weights and set(weights) == set(module.model.state_dict())
    assert all(torch.isfinite(tensor).all() for tensor in weights.values())
    description = BestModelCallback._model_description(
        *BestModelCallback._resolve_model_description(trainer, module, weights))
    payload = {'model': weights, **description}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    reloaded = torch.load(path, map_location='cpu', weights_only=True)
    expected = state_digest(weights)
    assert state_digest(reloaded['model']) == expected
    return {'model_state_sha256': expected, 'model_name': description['model_name'],
            'checkpoint_bytes': path.stat().st_size,
            'model_config': reloaded['model_config']}
