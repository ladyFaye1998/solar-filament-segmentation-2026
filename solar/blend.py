"""Ensemble contours of matched instances using signed pixel distances."""
import argparse
import json
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_util
from scipy import ndimage as ndi

from solar.metrics import encode,counts,summarize,load_ground_truth,audit


def blend_mask(first,second,weight,preserve_original_ties=False):
    if weight==1:
        return first
    if weight==0:
        return second
    shape=tuple(first['size'])
    assert tuple(second['size'])==shape
    a,b=mask_util.toBbox(first),mask_util.toBbox(second)
    x0=max(0,int(min(a[0],b[0]))-2)
    y0=max(0,int(min(a[1],b[1]))-2)
    x1=min(shape[1],int(max(a[0]+a[2],b[0]+b[2]))+2)
    y1=min(shape[0],int(max(a[1]+a[3],b[1]+b[3]))+2)
    masks=[mask_util.decode(rle)[y0:y1,x0:x1].astype(bool) for rle in [first,second]]
    signed=[]
    for mask in masks:
        padded=np.pad(mask,1,constant_values=False)
        signed.append((ndi.distance_transform_edt(padded)-ndi.distance_transform_edt(~padded))[1:-1,1:-1])
    field=weight*signed[0]+(1-weight)*signed[1]
    mixed=field>0
    if preserve_original_ties:
        # At equal weights, opposite boundary distances cancel exactly. Sending
        # every tied pixel to background erodes a one-pixel translated mask.
        # Keep the original detector's decision where the blend is undecided.
        mixed |= np.isclose(field,0.,rtol=0.,atol=1e-12)&masks[0]
    assert not (mixed&~(masks[0]|masks[1])).any(),'Ensemble must stay within predicted foreground union'
    assert np.all(mixed[masks[0]&masks[1]]),'Ensemble must retain shared foreground'
    original_area=float(mask_util.area(first))
    if not .5*original_area<=mixed.sum()<=1.8*original_area:
        return first
    result=np.zeros(shape,dtype=np.uint8)
    result[y0:y1,x0:x1]=mixed
    return encode(result)


def blend_instances(first,second,weight=.75,min_iou=.35,minimum_area=200):
    if weight==1 or not first or not second:
        return list(first)
    overlap=mask_util.iou(first,second,[0]*len(second))
    pairs={}
    used=set()
    for i,j in sorted(zip(*np.where(overlap>=min_iou)),key=lambda pair:-overlap[pair]):
        if i not in pairs and j not in used:
            pairs[i]=j
            used.add(j)
    proposed=[blend_mask(rle,second[pairs[index]],weight) if index in pairs else rle
              for index,rle in enumerate(first)]
    result=[]
    for rle in proposed:
        conflicts=np.flatnonzero(mask_util.iou([rle],result,[0]*len(result))[0]>0) if result else []
        if len(conflicts):
            mask=mask_util.decode(rle).astype(bool)
            for other in conflicts:
                mask &= ~mask_util.decode(result[other]).astype(bool)
            if mask.sum()<minimum_area:
                continue
            rle=encode(mask)
        result.append(rle)
    return result


def run(first_path,second_path,output,split,parameters=None):
    first=json.loads(first_path.read_text())
    second=json.loads(second_path.read_text())
    assert set(first)==set(second)
    output.mkdir(parents=True,exist_ok=True)
    if split=='test':
        choice=json.loads(parameters.read_text())['results'][0]
        prediction={stem:blend_instances(first[stem],second[stem],choice['weight'],choice['min_iou']) for stem in first}
        (output/'test-predictions.json').write_text(json.dumps(prediction))
        (output/'test-audit.json').write_text(json.dumps(audit(prediction,set(first)),indent=2))
        from solar.soft_unet import write_submission
        write_submission(prediction,output/'submission.csv')
        return
    truth=load_ground_truth(Path('data/raw/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json'),set(first))
    references={stem:[] for stem in first}
    for entry in truth.values():
        references[entry['stem']].append(entry['rles'])
    rules=[(1.,.35)]+[(weight,iou) for weight in [.5,.7,.85] for iou in [.2,.35,.5]]
    totals=np.zeros((len(rules),4))
    for index,stem in enumerate(sorted(first),1):
        for k,(weight,iou) in enumerate(rules):
            prediction=blend_instances(first[stem],second[stem],weight,iou)
            for reference in references[stem]:
                totals[k]+=counts(reference,prediction)
        if index%20==0:
            print('Contour ensemble images evaluated',index,flush=True)
    results=sorted([{'weight':weight,'min_iou':iou,**summarize(totals[k])} for k,(weight,iou) in enumerate(rules)],key=lambda row:-row['pq'])
    report={'first':str(first_path),'second':str(second_path),'results':results,
            'selection_scope':'Reused 132-image development cohort; not an independent final score'}
    (output/'selection.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(results,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--first',type=Path,required=True)
    parser.add_argument('--second',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--split',choices=['val','test'],default='val')
    parser.add_argument('--parameters',type=Path)
    args=parser.parse_args()
    run(args.first,args.second,args.output,args.split,args.parameters)
