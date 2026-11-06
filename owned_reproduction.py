"""Image-to-mask reproduction of the owned 0.39 ensemble; no prediction tables.

Run from a bundle made by scripts/build_owned_reproduction.py. The only inference
inputs are the supplied JPEGs and checksummed model weights. Optional reference
predictions are read AFTER inference, solely to measure numerical reproduction.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).with_name("models.json"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--refiner-device", default="cpu")
    parser.add_argument("--refiner-fp32", action="store_true")
    parser.add_argument("--stems", nargs="*")
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    base = Path(__file__).resolve().parent
    os.chdir(base)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    images = args.data.resolve() / "test/test_images"
    full = sorted(p.stem for p in images.glob("*.jpeg"))
    if len(full) != 180:
        raise ValueError("A production export requires all 180 official test JPEGs")
    stems = sorted(args.stems) if args.stems else full
    if not stems or len(stems) != len(set(stems)) or not set(stems) <= set(full):
        raise ValueError("Invalid probe image IDs")
    for name, expected in manifest["code_sha256"].items():
        if digest(base / name) != expected:
            raise ValueError(f"Bundle source changed: {name}")
    for member in manifest["detectors"] + list(manifest["refiners"].values()):
        if digest(base / member["path"]) != member["sha256"]:
            raise ValueError(f"Model checksum mismatch: {member['path']}")
    import torch
    import cv2
    import numpy as np
    from pycocotools import mask as mu
    from solar.metrics import encode, audit, counts, summarize
    from solar.yolo_run import select_instances, write_csv
    from scripts.eval_yolo_ensemble import clusters, fuse
    import ultralytics.nn.tasks as tasks
    from ultralytics import YOLO
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    started = time.monotonic()
    banks = []
    events = []
    original_fuse = tasks.BaseModel.fuse
    tasks.BaseModel.fuse = lambda self, verbose=True, **kwargs: self
    try:
        for member in manifest["detectors"]:
            if member["family"] != "yolo":
                continue
            model = YOLO(str(base / member["path"]))
            bank = {}
            for number, stem in enumerate(stems):
                options = dict(imgsz=member["resolution"], conf=.05, iou=0., max_det=100,
                               device=args.device, verbose=False, retina_masks=member["resolution"] != 2048)
                result = model.predict(str(images / f"{stem}.jpeg"), **options)[0]
                if number == 0:
                    inner = getattr(getattr(model.predictor.model, "backend", None), "model", model.predictor.model)
                    bns = [m for m in inner.modules() if isinstance(m, torch.nn.BatchNorm2d)]
                    if not bns:
                        raise ValueError("BatchNorm was fused; cannot reproduce per-image normalization")
                    for layer in bns:
                        layer.train()
                        layer.momentum = 0.
                    result = model.predict(str(images / f"{stem}.jpeg"), **options)[0]
                rows = []
                if result.masks is not None:
                    for mask, score, box in zip(result.masks.data.cpu().numpy(), result.boxes.conf.cpu().numpy(), result.boxes.xyxy.cpu().numpy()):
                        if mask.shape != (2048, 2048):
                            raise ValueError("Detector did not return native-resolution masks")
                        if mask.any():
                            rows.append(dict(rle=encode(mask), score=float(score), box=box.tolist()))
                bank[stem] = rows
            banks.append(bank)
            event = dict(member=member["name"], images=len(bank), seconds=time.monotonic() - started)
            events.append(event)
            print(json.dumps(event), flush=True)
            # The backend and BatchNorm list also own modules. Keeping them
            # alive retains the last YOLO on the GPU during native Mask R-CNN.
            # The in-progress frozen replay retains its original code copy.
            del model, result, inner, bns, layer
            gc.collect()
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
    finally:
        tasks.BaseModel.fuse = original_fuse
    from solar.native_full707_release import load_full707_weights
    from solar.native_maskrcnn_four_flip import infer_four_flips
    member = next(m for m in manifest["detectors"] if m["family"] == "maskrcnn")
    model, _ = load_full707_weights(base / member["path"], member["sha256"], device=args.device)
    _, fused, _, _, inspection = infer_four_flips(model, images, stems, device=args.device, score_cutoff=.65,
        progress=lambda n, total, stem, count: print(json.dumps(dict(stage="maskrcnn", done=n, total=total)), flush=True) if n % 20 == 0 or n == total else None)
    banks.append({stem: [row for row in fused[stem] if row['score'] >= .05] for stem in stems})
    del model
    gc.collect()
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    seeds = {}
    for stem in stems:
        candidates = []
        for cluster in clusters([b[stem] for b in banks], .5):
            rle, score = fuse(cluster, len(banks))
            x, y, w, h = mu.toBbox(rle)
            candidates.append(dict(rle=rle, score=score, box=[float(x), float(y), float(x+w), float(y+h)]))
        seeds[stem] = select_instances(candidates, confidence=.2, box_iou=1., minimum_area=5)
    (output / "seed-predictions.json").write_text(json.dumps(seeds), encoding="utf-8")
    from solar.crop_refiner import build_model, infer
    from solar.incumbent_probability_cut import contours
    from solar.incumbent_component_pruning import prune
    maps = {}
    if args.refiner_fp32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    for name, member in manifest["refiners"].items():
        state = torch.load(base / member["path"], map_location="cpu", weights_only=True)
        model = build_model(state.get("encoder", "timm-efficientnet-b0"), pretrained=False)
        model.load_state_dict(state["model"] if "model" in state else state)
        model = model.to(args.refiner_device)
        # CPU is the original export. The explicit GPU-fp32 option is checked
        # separately for floating-point and final-mask parity.
        maps[name] = output / f"maps-{name}"
        options = dict(use_autocast=not args.refiner_fp32) if args.refiner_device != "cpu" else {}
        infer(model.eval(), {s:r for s,r in seeds.items() if r}, images, maps[name], **options)
        del model, state
        gc.collect()
    predictions = {}
    for stem in stems:
        if not seeds[stem]:
            predictions[stem] = []
            continue
        values = {}
        for name, directory in maps.items():
            with np.load(directory / f"{stem}.npz", allow_pickle=False) as archive:
                values[name] = (archive["probability"].astype(np.float32), archive["matrix"])
        b0 = (values["b0a"][0] + values["b0b"][0]) / 2
        mixed = np.rint(.75 * np.rint(b0) + .25 * values["r34"][0]).astype(np.uint8)
        predictions[stem] = [prune(r) for r in contours(seeds[stem], mixed, values["r34"][1], .5)]
    prediction_path = output / "predictions.json"
    prediction_path.write_text(json.dumps(predictions, separators=(",", ":")), encoding="utf-8")
    report = dict(integrity=audit(predictions, set(stems)), elapsed_seconds=time.monotonic()-started,
                  neural_image_inference_completed=True, images=len(stems), probe_only=bool(args.stems),
                  detectors=len(banks), refiners=len(maps), events=events, maskrcnn_inspection=inspection,
                  refiner_device=args.refiner_device, refiner_fp32=args.refiner_fp32,
                  prediction_sha256=digest(prediction_path), new_submissions=0)
    if not args.stems:
        from scripts.audit_submission import verify
        write_csv(predictions, output / "submission.csv")
        report["csv_audit"] = verify(output / "submission.csv", prediction_path, images)
    if args.reference:
        reference = json.loads(args.reference.read_text(encoding="utf-8"))
        total = sum((counts(reference[s], predictions[s]) for s in stems), np.zeros(4))
        report["reproduction_comparison"] = dict(exact_prediction_equality=all(reference[s] == predictions[s] for s in stems),
            mask_agreement=summarize(total), meaning="Agreement with the previous inference, not ground-truth accuracy",
            reference_read_after_all_predictions_completed=True)
    (output / "complete.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
