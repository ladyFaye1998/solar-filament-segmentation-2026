"""Native-resolution U-Net ensemble benchmark with real OOF validation.

Public benchmark weights: Dariush Afshar, solar-filament-unet-weights,
CC BY-NC-SA 4.0. Predictions are recomputed from each actual image.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy import ndimage as ndi

from solar.metrics import encode, audit, evaluate, load_ground_truth


D4 = [(0,False),(0,True),(2,False),(2,True),(1,False),(1,True),(3,False),(3,True)]
RULE = {'low':.35, 'peak':.99, 'area':200, 'mean':.85}


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def transform(x, rotation, reflection):
    if reflection:
        x = torch.flip(x, (3,))
    return torch.rot90(x, rotation, (2,3)) if rotation else x


def invert(x, rotation, reflection):
    if rotation:
        x = torch.rot90(x, -rotation, (2,3))
    return torch.flip(x, (3,)) if reflection else x


@torch.inference_mode()
def predict(model, image: np.ndarray, device: torch.device, views_per_pass=2):
    h,w = image.shape
    assert (h,w)==(2048,2048)
    yy,xx = np.ogrid[:h,:w]
    disk = (yy-(h-1)/2)**2 + (xx-(w-1)/2)**2 <= 950**2
    values = image[disk].astype(np.float32)/255.
    mean, std = float(values.mean()), max(float(values.std()),1e-3)
    x = (torch.from_numpy(image.copy()).to(device).float()/255.-mean)/std
    x = x[None,None]
    total = torch.zeros((h,w),dtype=torch.float32,device=device)
    for start in range(0,8,views_per_pass):
        operations = D4[start:start+views_per_pass]
        batch = torch.cat([transform(x,*op) for op in operations])
        batch = batch.contiguous(memory_format=torch.channels_last)
        with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):
            logits = model(batch)
        probabilities = logits.float().sigmoid()
        for index,op in enumerate(operations):
            total += invert(probabilities[index:index+1],*op)[0,0]
    return (total/8*255).round().clamp(0,255).to(torch.uint8).cpu().numpy()


def components(probability: np.ndarray, rule=RULE) -> list[dict]:
    labels,n = ndi.label(probability > round(rule['low']*255), structure=np.ones((3,3)))
    if not n:
        return []
    flat = labels.ravel()
    index = np.flatnonzero(flat)
    label_values,values = flat[index],probability.ravel()[index]
    area = np.bincount(label_values,minlength=n+1)
    sums = np.bincount(label_values,weights=values,minlength=n+1)
    peak = np.zeros(n+1,dtype=np.uint8)
    np.maximum.at(peak,label_values,values)
    mean = sums/np.maximum(area,1)/255.
    keep = np.flatnonzero((area>=rule['area']) & (mean>=rule['mean']) & (peak>round(rule['peak']*255)))
    return [encode(labels==i) for i in keep if i]


def write_submission(predictions, path):
    with Path(path).open('w',newline='',encoding='ascii') as stream:
        writer = csv.writer(stream,lineterminator='\n')
        writer.writerow(['filament_id','segmentation_rle'])
        for stem,rles in sorted(predictions.items()):
            for index,rle in enumerate(rles,1):
                writer.writerow([f'{stem}_{index}',rle['counts']])


def run(root: Path, weights: Path, output: Path, partition: Path, views_per_pass=2):
    import segmentation_models_pytorch as smp
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = True
    output.mkdir(parents=True,exist_ok=True)
    receipt = json.loads((weights/'solar_unet_receipt.json').read_text())
    models = receipt['ensemble']['models']
    with (weights/'folds.csv').open() as stream:
        folds = {r['stem']:int(r['fold']) for r in csv.DictReader(stream)}
    dev_stems = set(json.loads(partition.read_text())['validation_ids'])
    paths = {'test': sorted((root/'test/test_images').glob('*.jpeg')),
             'val': [root/'train/train_images'/f'{s}.jpeg' for s in sorted(dev_stems)]}
    assert len(paths['test'])==180 and len(paths['val'])==132
    assert torch.cuda.is_available(), 'This run explicitly requires a GPU'
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    verified = []
    for entry in models:
        path = weights/entry['file']
        assert path.stat().st_size==entry['bytes'] and digest(path)==entry['sha256'], entry['file']
        verified.append({'file':entry['file'],'sha256':entry['sha256']})
    start = time.monotonic()
    cache = output/'cache'

    def worker(index):
        device = devices[index]
        torch.cuda.set_device(device)
        for entry in models[index::len(devices)]:
            name = Path(entry['file']).stem
            model = smp.Unet(encoder_name=entry['encoder'],encoder_weights=None,in_channels=1,classes=1)
            model.load_state_dict(torch.load(weights/entry['file'],map_location='cpu',weights_only=True),strict=True)
            model = model.to(device).to(memory_format=torch.channels_last).eval()
            for split,images in paths.items():
                selected = images if split=='test' else [p for p in images if folds[p.stem]==entry['fold']]
                destination = cache/split/name
                destination.mkdir(parents=True,exist_ok=True)
                for i,path in enumerate(selected):
                    target = destination/f'{path.stem}.npz'
                    if target.exists():
                        continue
                    image = cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
                    probability = predict(model,image,device,views_per_pass)
                    np.savez_compressed(target,probability=probability)
                    if (i+1)%20==0 or i+1==len(selected):
                        print(json.dumps({'model':name,'split':split,'done':i+1,'total':len(selected),
                                          'seconds':round(time.monotonic()-start,1)}),flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()

    with ThreadPoolExecutor(len(devices)) as pool:
        list(pool.map(worker,range(len(devices))))
    predictions = {}
    for split,images in paths.items():
        predictions[split] = {}
        maps = output/'maps'/split
        maps.mkdir(parents=True,exist_ok=True)
        for image in images:
            entries = models if split=='test' else [m for m in models if m['fold']==folds[image.stem]]
            total = np.zeros((2048,2048),dtype=np.uint16)
            for entry in entries:
                source = cache/split/Path(entry['file']).stem/f'{image.stem}.npz'
                with np.load(source) as archive:
                    total += archive['probability'].astype(np.uint16)
            mean = np.rint(total.astype(np.float32)/len(entries)).astype(np.uint8)
            np.savez_compressed(maps/f'{image.stem}.npz',probability=mean)
            predictions[split][image.stem] = components(mean)
        (output/f'{split}-predictions.json').write_text(json.dumps(predictions[split]))
        (output/f'{split}-audit.json').write_text(json.dumps(audit(predictions[split],{p.stem for p in images}),indent=2))
    truth = load_ground_truth(root/'train/MAGFiLO_1.0_Annotations_kaggle2026_train.json',dev_stems)
    result = evaluate(truth,predictions['val'])
    result['validation_caveat'] = 'Public weights use physical-image folds, not this project date folds. Two held-out models per DEV image; ten models averaged for test. Original author selected this keep rule using these labels.'
    (output/'validation.json').write_text(json.dumps(result,indent=2))
    write_submission(predictions['test'],output/'submission.csv')
    report = {'pipeline':'native-resolution soft-target U-Net ensemble','source':'dariushafshar/solar-filament-unet-weights',
              'license':'CC-BY-NC-SA-4.0','weights':verified,'parameters':RULE,'tta_views':8,
              'views_per_forward':views_per_pass,'devices':[torch.cuda.get_device_name(d) for d in devices],
              'torch':torch.__version__,'elapsed_seconds':time.monotonic()-start,
              'validation_pq':result['pq'],'submission_sha256':digest(output/'submission.csv')}
    (output/'run.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('artifacts/soft-unet'))
    parser.add_argument('--partition',type=Path,default=Path('configs/partition.json'))
    parser.add_argument('--views-per-pass',type=int,default=2)
    args=parser.parse_args()
    run(args.root,args.weights,args.output,args.partition,args.views_per_pass)
