"""Full707refit native2048 study with NO independent validation, resumable at optimizer boundaries.

All sources and the actual training index are checkpoint-bound. This module
never evaluates or enumerates competition test images and never exports CSVs.
"""
from __future__ import annotations
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image
from pycocotools import mask as mask_util
import torch
from torch.utils.data import Dataset,DataLoader
from solar.native_full707_data import Full707PhotoExpertSequence as PhotoExpertSequence,augment
from solar.torchvision_native_maskrcnn import build_model
from solar.rfdetr_checkpoint import state_digest


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


class TrainingPhotos(Dataset):
    def __init__(self,index,image_root,start,seed):
        self.sequence=PhotoExpertSequence(index,seed,60)
        self.image_root=Path(image_root).resolve();self.start=start;self.seed=seed
    def __len__(self):return len(self.sequence)-self.start
    def __getitem__(self,position):
        global_index=self.start+position;record=self.sequence.record(global_index)
        path=(self.image_root/record['filename']).resolve();assert path.parent==self.image_root
        with Image.open(path) as source:
            assert source.size==(2048,2048)
            image=np.asarray(source.convert('RGB'))
        masks=np.stack([mask_util.decode(rle) for rle in record['rles']])
        boxes=np.asarray([[x,y,x+w,y+h] for x,y,w,h in record['boxes_xywh']],np.float32)
        image,masks,boxes=augment(image,masks,boxes,self.seed,global_index)
        image=torch.from_numpy(image).permute(2,0,1).float()/255.
        target=dict(boxes=torch.from_numpy(boxes),labels=torch.ones(len(boxes),dtype=torch.int64),
            masks=torch.from_numpy(masks),area=torch.tensor(record['areas'],dtype=torch.float32),
            iscrowd=torch.zeros(len(boxes),dtype=torch.int64),image_id=torch.tensor([record['source_id']]))
        return image,target,dict(global_index=global_index,stem=record['stem'],expert_record_id=record['expert_record_id'])


def single(batch):
    assert len(batch)==1
    return batch[0]


def learning_rate(step,recipe):
    warmup=recipe['warmup_optimizer_steps'];base=recipe['learning_rate'];minimum=recipe['minimum_learning_rate']
    if step<warmup:return recipe['warmup_learning_rate']+(base-recipe['warmup_learning_rate'])*step/warmup
    progress=(step-warmup)/(recipe['total_optimizer_updates']-warmup)
    return minimum+.5*(base-minimum)*(1+math.cos(math.pi*progress))


def run(index_path,image_root,recipe_path,output,total_session_seconds,resume_path=None,stop_after_updates=None):
    started=time.monotonic();output=Path(output);output.mkdir(parents=True,exist_ok=False)
    recipe_path=Path(recipe_path);recipe=json.loads(recipe_path.read_text());index_path=Path(index_path)
    assert digest(index_path)==recipe['index_sha256']
    index=json.loads(index_path.read_text())
    assert recipe['epochs']==60 and recipe['total_microbatches']==42420 and recipe['total_optimizer_updates']==10605
    assert recipe['resolution']==2048 and recipe['mask_output_resolution']==112
    assert recipe['schema']=='owned_native_MaskRCNN_full707_training_recipe_v1'
    assert index['schema']=='owned_native_full707_input_index_v1'
    assert recipe['training_photos']==707 and recipe['training_expert_records']==1154
    assert recipe['validation_photos_excluded']==0 and not index['independent_validation_available_for_this_refit']
    stop_target=10605 if stop_after_updates is None else int(stop_after_updates)
    assert 0<stop_target<=10605
    packages={n:importlib.metadata.version(n) for n in ['torch','torchvision','numpy','Pillow','pycocotools']}
    assert packages['torch']=='2.11.0+cu128' and packages['torchvision']=='0.26.0+cu128'
    source_hashes={name:digest(Path(name)) for name in recipe['frozen_sources']}
    assert source_hashes==recipe['frozen_sources']
    recipe_sha=digest(recipe_path);device=torch.device('cuda:0')
    assert torch.cuda.is_available() and 'T4' in torch.cuda.get_device_name(0)
    torch.set_num_threads(2);torch.manual_seed(recipe['seed'])
    torch.cuda.manual_seed_all(recipe['seed']);torch.cuda.reset_peak_memory_stats()
    model,preserved=build_model(pretrained=resume_path is None,resolution=2048,sampled_rois=128)
    model=model.to(device)
    optimizer=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=recipe['learning_rate'],
        momentum=.9,weight_decay=.0005)
    scaler=torch.amp.GradScaler('cuda',init_scale=recipe['AMP_initial_scale'])
    completed=0;skipped=0;seen_photos=set();seen_experts=set();resume_proof=None
    if resume_path is not None:
        payload=torch.load(resume_path,map_location='cpu',weights_only=True)
        assert payload['schema']=='owned_native_MaskRCNN_V2_full707_training_v1'
        assert payload['recipe_sha256']==recipe_sha and payload['source_hashes']==source_hashes
        assert payload['index_sha256']==recipe['index_sha256'] and payload['total_optimizer_updates']==10605
        assert payload['completed_microbatches']==4*payload['successful_optimizer_steps']
        model.load_state_dict(payload['model'],strict=True);optimizer.load_state_dict(payload['optimizer'])
        scaler.load_state_dict(payload['scaler'])
        assert state_digest(model.state_dict())==payload['model_state_sha256']
        assert state_digest(optimizer.state_dict())==payload['optimizer_state_sha256']
        assert state_digest(scaler.state_dict())==payload['scaler_state_sha256']
        completed=payload['successful_optimizer_steps'];skipped=payload['skipped_overflow_attempts']
        seen_photos=set(payload['seen_training_photos']);seen_experts=set(payload['seen_training_expert_records'])
        torch.set_rng_state(payload['torch_rng']);torch.cuda.set_rng_state_all(payload['cuda_rng'])
        resume_proof=dict(checkpoint_sha256=digest(Path(resume_path)),
            model_optimizer_scaler_exact=True,resumed_optimizer_step=completed,
            resumed_global_microbatch=4*completed,source_bindings_exact=True)
        del payload
    sequence=PhotoExpertSequence(index,recipe['seed'],60)
    assert len(sequence)==42420
    dataset=TrainingPhotos(index,image_root,4*completed,recipe['seed'])
    loader=DataLoader(dataset,batch_size=1,shuffle=False,num_workers=0,collate_fn=single,
        generator=torch.Generator().manual_seed(recipe['seed']+4*completed))
    iterator=iter(loader)
    protocol=dict(at_utc=datetime.now(timezone.utc).isoformat(),recipe_sha256=recipe_sha,
        source_hashes=source_hashes,packages=packages,training_photos=707,training_expert_records=1154,
        excluded_validation_photos=0,photo_expert_sequence='Each photo exactly once per epoch, experts cycle evenly;60epochs.',
        total_optimizer_updates=10605,effective_batch=4,physical_batch=1,native_resolution=2048,
        predicted_mask_resolution=112,pretrained_preserved_components=preserved,resume_proof=resume_proof,
        independent_validation_available=False,requested_stop_update=stop_target,
        model_training_started=True,test_inputs_used=False,test_labels_used=False,submissions_enabled=False)
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    events=[];loss_rows=[];average_step=2.;last_checkpoint=None;model.train()

    def save():
        nonlocal last_checkpoint
        torch.cuda.synchronize()
        state=dict(schema='owned_native_MaskRCNN_V2_full707_training_v1',model=model.state_dict(),
            training_photos=707,training_expert_records=1154,excluded_validation_photos=0,
            optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
            successful_optimizer_steps=completed,completed_microbatches=4*completed,total_optimizer_updates=10605,
            recipe_sha256=recipe_sha,source_hashes=source_hashes,index_sha256=recipe['index_sha256'],
            model_state_sha256=state_digest(model.state_dict()),optimizer_state_sha256=state_digest(optimizer.state_dict()),
            scaler_state_sha256=state_digest(scaler.state_dict()),torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all(),skipped_overflow_attempts=skipped,
            seen_training_photos=sorted(seen_photos),seen_training_expert_records=sorted(seen_experts),packages=packages)
        temporary=output/'last.pt.partial';checkpoint=output/'last.pt'
        torch.save(state,temporary);temporary.replace(checkpoint)
        last_checkpoint=dict(path=str(checkpoint),sha256=digest(checkpoint),bytes=checkpoint.stat().st_size,
            successful_optimizer_steps=completed,completed_microbatches=4*completed,
            model_state_sha256=state['model_state_sha256'])
        (output/'checkpoint.json').write_text(json.dumps(last_checkpoint,indent=2))

    assert completed<=stop_target
    while completed<stop_target:
        if time.monotonic()-started+45+3*average_step>=total_session_seconds:break
        batch=[next(iterator) for _ in range(4)]
        assert [row[2]['global_index'] for row in batch]==list(range(4*completed,4*completed+4))
        for group in optimizer.param_groups:group['lr']=learning_rate(completed,recipe)
        successful=False;update_started=time.monotonic()
        for attempt in range(8):
            optimizer.zero_grad(set_to_none=True)
            buffers={n:b.detach().clone() for n,b in model.named_buffers()}
            cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all();before_scale=scaler.get_scale()
            micro=[]
            try:
                for image,target,meta in batch:
                    image=image.to(device);target={k:v.to(device) for k,v in target.items()}
                    with torch.autocast(device_type='cuda',dtype=torch.float16):
                        losses=model([image],[target]);loss=sum(losses.values())/4
                    values={k:float(v.detach()) for k,v in losses.items()}
                    assert all(np.isfinite(v) for v in values.values()),f'Nonfinite loss at{meta}'
                    scaler.scale(loss).backward();micro.append(values)
                scaler.unscale_(optimizer)
                finite=bool(torch.stack([torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]).all())
                if not finite:
                    before_weights=state_digest({n:p for n,p in model.named_parameters()})
                    before_optimizer=state_digest(optimizer.state_dict())
                    scaler.step(optimizer);scaler.update();assert scaler.get_scale()<before_scale
                    assert state_digest({n:p for n,p in model.named_parameters()})==before_weights
                    assert state_digest(optimizer.state_dict())==before_optimizer
                    with torch.no_grad():
                        for n,b in model.named_buffers():b.copy_(buffers[n])
                    torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state_all(cuda_rng)
                    skipped+=1
                    events.append(dict(step=completed,attempt=attempt,scale_before=before_scale,scale_after=scaler.get_scale(),
                        trainable_weights_optimizer_unchanged=True,buffers_and_RNG_restored=True))
                    (output/'AMP-skips.json').write_text(json.dumps(events,indent=2))
                    continue
                norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5.));assert np.isfinite(norm) and norm>0
                scaler.step(optimizer);scaler.update();assert scaler.get_scale()>=before_scale
                successful=True;break
            except Exception:
                # No optimizer update precedes any of the assertions above.
                with torch.no_grad():
                    for n,b in model.named_buffers():b.copy_(buffers[n])
                raise
        assert successful,'Eight AMP backoffs without a finite update; stop rather than changing the frozen recipe'
        completed+=1
        for _,_,meta in batch:seen_photos.add(meta['stem']);seen_experts.add(meta['expert_record_id'])
        torch.cuda.synchronize();seconds=time.monotonic()-update_started
        average_step=.8*average_step+.2*seconds
        row=dict(step=completed,photo_epochs=4*completed/707,learning_rate=optimizer.param_groups[0]['lr'],
            loss=sum(sum(v.values()) for v in micro)/4,mask_loss=sum(v['loss_mask'] for v in micro)/4,
            gradient_norm_before_clip=norm,AMP_scale=scaler.get_scale(),seconds=seconds,
            unique_training_photos=len(seen_photos),unique_training_expert_records=len(seen_experts))
        loss_rows.append(row)
        if completed%10==0 or completed<=2:
            (output/'progress.json').write_text(json.dumps(row,indent=2));print('OWNED_NATIVE_FULL707_TRAIN',json.dumps(row),flush=True)
        if completed%100==0:save()
    save()
    (output/'training-log.json').write_text(json.dumps(loss_rows,indent=2))
    result=dict(status='owned_native_full707_bounded_training_segment_checkpoint_saved',
        recipe_sha256=recipe_sha,source_hashes=source_hashes,packages=packages,
        successful_optimizer_steps=completed,completed_microbatches=4*completed,target_optimizer_updates=10605,
        target_training_complete=completed==10605,photo_epochs=4*completed/707,
        unique_training_photos=len(seen_photos),unique_training_expert_records=len(seen_experts),
        checkpoint=last_checkpoint,skipped_overflow_attempts=skipped,resume_proof=resume_proof,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        session_elapsed_seconds=time.monotonic()-started,session_successful_updates=len(loss_rows),
        training_loss_last=loss_rows[-1] if loss_rows else None,
        training_photos=707,training_expert_records=1154,excluded_validation_photos=0,
        independent_validation_available=False,requested_stop_update=stop_target,requested_stop_reached=completed==stop_target,
        heldout_accuracy_measured=False,test_inputs_used=False,test_labels_used=False,
        competition_submission_enabled=False,paid_compute=False,winning_solution_verified=False)
    (output/'complete.json').write_text(json.dumps(result,indent=2))
    return result
