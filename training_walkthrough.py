"""Optional training from official labels; never evaluate on competition test pixels.

Run this file from the release bundle. Historical recipes are retained as evidence;
new training is a new fit, with no promise of byte-identical weights or public PQ.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

BUNDLE = Path(__file__).resolve().parent
sys.path.insert(0, str(BUNDLE))
ANNOTATION_SHA = '5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2), encoding='utf-8')


@contextmanager
def copy_fallback():
    """Keep original label preparation usable across drives without hard links."""
    original = os.link
    def link(source, destination, *args, **kwargs):
        try:
            return original(source, destination, *args, **kwargs)
        except OSError:
            return shutil.copy2(source, destination)
    os.link = link
    try:
        yield
    finally:
        os.link = original


def check_data(root):
    path = Path(root) / 'train/MAGFiLO_1.0_Annotations_kaggle2026_train.json'
    assert sha(path) == ANNOTATION_SHA, 'Use the official competition TRAIN annotation file'
    return path


def yolo_dataset(root, output, recipe, gong=None):
    """Prepare one expert record per example, or the historical union policy."""
    from solar.yolo_run import prepare
    from solar.yolo_labels import prepare_union, prepare_records_deploy
    from solar.yolo_full_run import prepare_full, link
    configs = BUNDLE / 'configs'
    check_data(root)
    if recipe['policy'] == 'historical-neighbours':
        if gong is None:
            raise ValueError('Set FILAMENT_GONG/--gong to the permitted same-site GONG JPEG cache')
        # Exact admitted historical rows. Do not silently substitute the later
        # quality-filtered dataset or additional cached observations.
        from solar.external_data_policy import ExternalFrameGuard, test_ids_from_root
        annotations = read(check_data(root))
        by_id = {row['id']:row for row in annotations['images']}
        stems = {Path(row['file_name']).stem for row in annotations['images']}
        guard = ExternalFrameGuard(frozenset(stems), frozenset(), frozenset(test_ids_from_root(root)))
        allowed = read(BUNDLE / 'training-provenance/gong-historical-frames.json')
        for frame, digest in allowed.items():
            assert sha(Path(gong) / (frame + '.jpeg')) == digest, frame
        with copy_fallback():
            config, audit = prepare_full(root, output, neighbour_dir=None)
        rows = read(BUNDLE / 'training-provenance/gong-historical-manifest.json')
        for row in rows:
            record, frame = by_id[row['source_record']], row['neighbour']
            source = Path(record['file_name']).stem
            assert guard.rejection(frame, source, max_seconds=150) is None, frame
            name = f"{row['source_record']}__{frame}"
            link(Path(gong) / (frame + '.jpeg'), output / 'images/train' / (name + '.jpeg'))
            shutil.copy2(output / 'labels/train' / (str(row['source_record']) + '.txt'), output / 'labels/train' / (name + '.txt'))
        audit.update(neighbour_images_added=len(rows), distinct_neighbour_frames=len(allowed),
                     historical_manifest_used=True, later_quality_exclusions_intentionally_not_applied=True,
                     quality_limitation='68 admitted historical frames fail the later quality filter',
                     temporal_policy=guard.metadata())
        assert len(rows) == 2683 and len(allowed) == 1649
        write(output / 'audit.json', audit)
        return config, audit
    partition = configs / ('partition-full707.json' if recipe['training_photos'] == 707 else 'partition.json')
    with copy_fallback():
        if recipe['training_photos'] == 707:
            return prepare_records_deploy(root, partition, configs / 'target-selection.json', output)
        if recipe['policy'] == 'union':
            return prepare_union(root, partition, configs / 'target-selection.json', output)
        return prepare(root, partition, configs / 'target-selection.json', output, 'all-annotators')


def generic_initializer(architecture, cache, expected=None):
    """Download only the declared generic COCO checkpoint, from the official host."""
    import torch
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / (architecture + '.pt')
    url = 'https://github.com/ultralytics/assets/releases/download/v8.4.0/' + path.name
    if not path.exists():
        temporary = path.with_suffix('.download')
        torch.hub.download_url_to_file(url, str(temporary), progress=True)
        if expected is not None:
            assert sha(temporary) == expected, 'Generic initializer hash differs'
        temporary.replace(path)
    if expected is not None:
        assert sha(path) == expected, 'Generic initializer hash differs'
    return path, dict(url=url, sha256=sha(path), historical_hash_available=expected is not None)


def train_yolo(root, output, recipe, gong=None):
    import torch
    from ultralytics import YOLO, settings
    settings.update({'wandb':False, 'mlflow':False, 'clearml':False, 'comet':False, 'tensorboard':False, 'sync':False})
    config, audit = yolo_dataset(root, output / 'dataset', recipe, gong)
    initial, initialization = generic_initializer(recipe['architecture'], output.parent / 'generic-initializers', recipe.get('initial_sha256'))
    args = dict(recipe['training_args'])
    args.update(data=str(config.resolve()), device=0, project=str((output / 'training').resolve()),
                name='recipe-fit', imgsz=recipe['resolution'], epochs=recipe['nominal_epochs'], time=recipe['hours_max'])
    # This path trains only. It does not call the older modules' automatic
    # deployment/test-export steps after fitting.
    write(output / 'new-fit-protocol.json', dict(recipe=recipe, initialization=initialization,
        actual_args=args, dataset=audit, test_pixels_opened=False, exact_weight_reproduction_claimed=False))
    model = YOLO(str(initial))
    model.train(**args)
    checkpoint = output / 'training/recipe-fit/weights/last.pt'
    assert checkpoint.is_file()
    write(output / 'training-result.json', dict(checkpoint=str(checkpoint), sha256=sha(checkpoint),
        new_fit=True, public_score=None, test_inference_performed=False))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return checkpoint


def native_index(root, output):
    from solar.maskrcnn_training_data import build_index
    from solar.native_full707_index import build_full707_index
    source = build_index(check_data(root), BUNDLE / 'configs/month-replication-fold0.json')
    body = json.dumps(source, separators=(',', ':')).encode()
    source_sha = hashlib.sha256(body).hexdigest()
    assert source_sha == '49e018490224b92fff333e68f57c5a5f1ebcd7b559b74b7262df8675cae395f1'
    full = build_full707_index(source, source_sha)
    path = output / 'full707-index.json'
    path.write_text(json.dumps(full, separators=(',', ':')), encoding='utf-8')
    recipe = read(BUNDLE / 'configs/owned-native-full707-v2-60epoch-20261024.json')
    assert sha(path) == recipe['index_sha256']
    return path


def refiner_cache(root, work, full=False):
    from solar.crop_refiner import prepare_training
    configs = BUNDLE / 'configs'
    cache = work / ('crops-full707' if full else 'crops-565')
    check_data(root)
    prepare_training(root, configs / ('full-refit-partition.json' if full else 'partition.json'),
                     configs / ('target-selection-full707.json' if full else 'target-selection.json'), cache)
    return cache


def train_refiner(root, output, name):
    import torch
    from solar.crop_refiner import train
    cache = refiner_cache(root, output.parent)
    if name == 'r34':
        model = train(cache, output, 'resnet34', seed=20261021)
    else:
        initial_dir = output.parent / 'b0-initial-565'
        initial = initial_dir / 'latest.pt'
        if not initial.exists():
            model = train(cache, initial_dir, 'timm-efficientnet-b0', seed=20261021)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        full = refiner_cache(root, output.parent, full=True)
        model = train(full, output, 'timm-efficientnet-b0', steps=3000, seconds_limit=1800,
                      initial=initial, lr_max=.00005, seed=20261022 if name == 'b0a' else 20261023)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    checkpoint = output / 'latest.pt'
    write(output / 'training-result.json', dict(checkpoint=str(checkpoint), sha256=sha(checkpoint),
        new_fit=True, private_detector_training_predictions_used=False, public_score=None))
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--component', required=True)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--gong', type=Path, default=os.environ.get('FILAMENT_GONG'))
    parser.add_argument('--session-seconds', type=float, default=36000)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    root, output = args.data.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    recipes = read(BUNDLE / 'training-provenance/training-recipes.json')
    os.chdir(BUNDLE)
    if args.component in recipes:
        recipe = recipes[args.component]
        if args.prepare_only:
            config, audit = yolo_dataset(root, output / 'dataset', recipe, args.gong)
            write(output / 'prepared.json', dict(config=str(config), audit=audit, training_executed=False))
        else:
            train_yolo(root, output, recipe, args.gong)
    elif args.component == 'maskrcnn707-fourflip':
        index = native_index(root, output)
        if not args.prepare_only:
            from solar.native_full707_training import run
            run(index, root / 'train/train_images', BUNDLE / 'configs/owned-native-full707-v2-60epoch-20261024.json',
                output / 'session', args.session_seconds, args.resume)
    elif args.component in ('b0a', 'b0b', 'r34'):
        if args.prepare_only:
            refiner_cache(root, output, full=args.component != 'r34')
        else:
            train_refiner(root, output, args.component)
    else:
        raise ValueError('Unknown component: ' + args.component)


if __name__ == '__main__':
    main()
