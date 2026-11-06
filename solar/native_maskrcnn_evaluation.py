"""Native instance decoding and held-out evaluation of the owned detector.

Postprocessing is independent of Torch so geometry and organizer-count semantics
can be checked without local neural inference or fitting.
"""
from __future__ import annotations
from collections import defaultdict
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from solar.metrics import encode,audit,counts,summarize,evaluate


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def decode_native(scores,labels,boxes,probability,shape=(2048,2048),score_cutoff=.5,mask_cutoff=.5,minimum_area=5):
    """Keep raw proposals and a canonical disjoint primary panoptic prediction.

    `probability(i)` returns one native-sized mask at a time; dense probability
    tensors are not exported. Confidence order gives ownership of shared pixels.
    """
    scores=np.asarray(scores,dtype=np.float64);labels=np.asarray(labels)
    boxes=np.asarray(boxes,dtype=np.float64)
    assert scores.ndim==labels.ndim==1 and boxes.shape==(len(scores),4)
    assert len(labels)==len(scores) and np.isfinite(scores).all() and np.isfinite(boxes).all()
    assert np.all((scores>=0)&(scores<=1)) and np.all(labels==1)
    height,width=shape
    if len(boxes):
        assert np.all(boxes[:,[0,2]]>=0) and np.all(boxes[:,[0,2]]<=width)
        assert np.all(boxes[:,[1,3]]>=0) and np.all(boxes[:,[1,3]]<=height)
        assert np.all(boxes[:,2]>=boxes[:,0]) and np.all(boxes[:,3]>=boxes[:,1])
    claimed=np.zeros(shape,bool);primary=[];candidates=[];trace=[]
    for i in np.argsort(-scores,kind='stable'):
        if scores[i]<.05:continue
        value=np.asarray(probability(int(i)))
        assert value.shape==shape and np.isfinite(value).all() and np.all((value>=0)&(value<=1))
        # Strict comparison avoids creating foreground from exact-zero logits.
        mask=value>mask_cutoff;raw_area=int(mask.sum())
        if raw_area<minimum_area:continue
        row=dict(detector_index=int(i),score=float(scores[i]),label=1,box_xyxy=boxes[i].tolist(),
            raw_area=raw_area,mask_mean_probability=float(value[mask].astype(np.float32).mean()),rle=encode(mask))
        candidates.append(row)
        if scores[i]<score_cutoff:continue
        owned=mask&~claimed;owned_area=int(owned.sum())
        if owned_area<minimum_area:continue
        claimed|=owned;primary.append(encode(owned))
        trace.append(dict(detector_index=int(i),score=float(scores[i]),raw_area=raw_area,
            owned_area=owned_area,removed_fraction=1-owned_area/raw_area))
    return primary,candidates,trace


def torch_decode(prediction,score_cutoff=.5):
    scores=prediction['scores'].detach().cpu().numpy()
    labels=prediction['labels'].detach().cpu().numpy()
    boxes=prediction['boxes'].detach().cpu().numpy()
    assert tuple(prediction['masks'].shape)==(len(scores),1,2048,2048)
    return decode_native(scores,labels,boxes,
        lambda i:prediction['masks'][i,0].detach().cpu().numpy(),score_cutoff=score_cutoff)


def load_owned_model(checkpoint_path,recipe_path,index_path,device='cuda:0'):
    import torch
    from solar.torchvision_native_maskrcnn import build_model
    from solar.rfdetr_checkpoint import state_digest
    recipe=json.loads(Path(recipe_path).read_text());index=json.loads(Path(index_path).read_text())
    assert digest(index_path)==recipe['index_sha256']
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    assert checkpoint['schema']=='owned_native_MaskRCNN_V2_training_v1'
    assert checkpoint['recipe_sha256']==digest(recipe_path)
    assert checkpoint['source_hashes']==recipe['frozen_sources']
    assert checkpoint['index_sha256']==recipe['index_sha256'] and checkpoint['total_optimizer_updates']==7995
    step=checkpoint['successful_optimizer_steps'];assert 0<step<=7995 and checkpoint['completed_microbatches']==4*step
    train={r['stem'] for r in index['records'] if r['split']=='train'}
    validation={r['stem'] for r in index['records'] if r['split']=='validation'}
    assert len(train)==533 and len(validation)==174 and train.isdisjoint(validation)
    assert set(checkpoint['seen_training_photos'])<=train
    assert {s[:6] for s in train}.isdisjoint(s[:6] for s in validation)
    for name,expected in recipe['frozen_sources'].items():assert digest(Path(name))==expected
    model,_=build_model(pretrained=False,resolution=2048,sampled_rois=128)
    model.load_state_dict(checkpoint['model'],strict=True)
    assert state_digest(model.state_dict())==checkpoint['model_state_sha256']
    model=model.to(device).eval()
    protocol=dict(checkpoint_sha256=digest(checkpoint_path),model_state_sha256=checkpoint['model_state_sha256'],
        recipe_sha256=digest(recipe_path),index_sha256=digest(index_path),training_source_hashes=checkpoint['source_hashes'],
        successful_optimizer_steps=step,completed_microbatches=4*step,training_target_updates=7995,
        is_fixed_final_checkpoint=step==7995,training_photos=533,training_expert_records=882,
        excluded_validation_photos=174,validation_expert_records=272,
        model_source_sha256=digest(Path(__file__)),test_inputs_used=False,test_labels_used=False)
    del checkpoint
    return model,recipe,index,protocol


def infer_photos(model,image_root,stems,device='cuda:0',score_cutoff=.5,progress=None):
    import torch
    from PIL import Image
    from solar.rfdetr_checkpoint import state_digest
    assert len(stems)==len(set(stems))
    image_root=Path(image_root).resolve();initial=state_digest(model.state_dict())
    predictions={};banks={};traces={};timings=[]
    for number,stem in enumerate(stems):
        started=time.monotonic();path=(image_root/(stem+'.jpeg')).resolve()
        assert path.parent==image_root
        with Image.open(path) as image:
            assert image.size==(2048,2048)
            rgb=np.asarray(image.convert('RGB')).copy()
        value=torch.from_numpy(rgb).permute(2,0,1).float().to(device)/255.
        with torch.inference_mode(),torch.autocast(device_type='cuda',dtype=torch.float16,enabled=str(device).startswith('cuda')):
            raw=model([value])[0]
        primary,pool,trace=torch_decode(raw,score_cutoff)
        predictions[stem]=primary;banks[stem]=pool;traces[stem]=trace
        timings.append(time.monotonic()-started)
        del raw,value
        if progress:progress(number+1,len(stems),stem,len(primary))
    assert state_digest(model.state_dict())==initial,'Inference changed trained neural weights or buffers'
    inspection=audit(predictions,set(stems))
    return predictions,banks,traces,dict(audit=inspection,neural_state_unchanged=True,
        image_seconds=timings,total_inference_seconds=float(sum(timings)))


def heldout_quality(index,predictions):
    records={r['expert_record_id']:dict(stem=r['stem'],rles=r['rles'])
        for r in index['records'] if r['split']=='validation'}
    assert len(records)==272 and len({r['stem'] for r in records.values()})==174
    assert set(predictions)=={r['stem'] for r in records.values()}
    result=evaluate(records,predictions)
    stems=sorted(predictions);per_photo=np.zeros((len(stems),4),np.float64)
    by_photo=defaultdict(list)
    for row in records.values():by_photo[row['stem']].append(row['rles'])
    for i,stem in enumerate(stems):
        for expert in by_photo[stem]:per_photo[i]+=counts(expert,predictions[stem])
    # Grouping by photo changes floating-point addition order relative to
    # the organizer's expert-record order. Integer counts remain exact.
    aggregate=summarize(per_photo.sum(0))
    assert all(aggregate[name]==result[name] for name in ['tp','fp','fn'])
    np.testing.assert_allclose(aggregate['sum_iou'],result['sum_iou'],rtol=0,atol=1e-9)
    np.testing.assert_allclose(aggregate['pq'],result['pq'],rtol=0,atol=1e-12)
    return result,stems,per_photo


def run_heldout(checkpoint_path,recipe_path,index_path,image_root,output):
    import torch
    assert torch.cuda.is_available() and 'T4' in torch.cuda.get_device_name(0)
    torch.set_num_threads(2);torch.cuda.reset_peak_memory_stats()
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    model,recipe,index,protocol=load_owned_model(checkpoint_path,recipe_path,index_path)
    stems=sorted({r['stem'] for r in index['records'] if r['split']=='validation'})
    assert len(stems)==174
    (output/'protocol.json').write_text(json.dumps(dict(protocol,precision='float16',
        primary_score_cutoff=.5,mask_cutoff=.5,mask_comparison='>',minimum_native_area=5,
        exclusive_pixel_priority='descending box score, stable original index',
        no_TTA=True,trained_neural_state_fixed=True),indent=2))
    predictions,banks,traces,inspection=infer_photos(model,image_root,stems,progress=lambda n,total,s,k:
        print('OWNED_NATIVE_HELDOUT',n,total,s,k,flush=True))
    quality,ordered,values=heldout_quality(index,predictions)
    for name,value in [('val-predictions.json',predictions),('raw-candidate-banks.json',banks),('ownership-traces.json',traces)]:
        (output/name).write_text(json.dumps(value,separators=(',',':')))
    np.savez_compressed(output/'per-photo-counts.npz',stems=ordered,counts=values)
    result=dict(status='owned_native_MaskRCNN_full174_all272_primary_quality_complete',
        at_utc=datetime.now(timezone.utc).isoformat(),checkpoint=protocol,primary=quality,
        native_inspection=inspection,all272_experts_scored=True,heldout_accuracy_measured=True,
        source_sha256=digest(Path(__file__)),prediction_sha256=digest(output/'val-predictions.json'),
        candidate_pool_sha256=digest(output/'raw-candidate-banks.json'),
        peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),
        cohort_caveat='Previously reused official development cohort, not an untouched lockbox; this checkpoint excludes every174heldout photo.',
        current_public_best_beaten=False,submission_admitted=False,test_inputs_used=False,test_labels_used=False,
        competition_submission_enabled=False,submissions_used=0)
    (output/'complete.json').write_text(json.dumps(result,indent=2))
    return result
