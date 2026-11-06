"""One fixed probability cut using cached, unchanged incumbent refiners."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from pycocotools import mask as mu

from solar.anchor_ensemble import combine
from solar.crop_refiner import refine_instances
from solar.metrics import counts,load_ground_truth
from solar.refiner_transfer import paired_report
from solar.yolo_run import write_csv

THRESHOLD=.4


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2))


def contours(seeds,probability,matrix,threshold):
    learned=refine_instances(seeds,probability,matrix,threshold=threshold,prior_weight=0.)
    if len(learned)!=len(seeds):learned=list(seeds)
    else:
        for i,(a,b) in enumerate(zip(seeds,learned)):
            if mu.iou([a],[b],[0])[0,0]<.2:learned[i]=a
    return combine(seeds,learned,.5)


def run(root,old_maps,full,inputs,out):
    import cv2
    from scripts.audit_submission import verify
    cv2.setNumThreads(1)
    plan=read('configs/incumbent-probability-cut-20261023.json')
    assert plan['threshold']==THRESHOLD and plan['weight_original']==.5
    for p,h in plan['source_sha256'].items():assert sha(p)==h,p
    split=read('configs/partition.json');stems=sorted(split['validation_ids'])
    assert len(stems)==132 and len(split['training_ids'])==565
    assert not {s[:8] for s in stems}&{s[:8] for s in split['training_ids']}
    assert not out.exists();out.mkdir(parents=True)
    truth=defaultdict(list)
    for r in load_ground_truth(root/'train/MAGFiLO_1.0_Annotations_kaggle2026_train.json',set(stems)).values():truth[r['stem']].append(r['rles'])
    assert len(truth)==132 and sum(map(len,truth.values()))==211
    families={}
    for f in ['medoid','all-annotators']:
        seeds=read(inputs/f'{f}-seeds.json');learned=read(inputs/f'{f}-learned.json')
        assert set(seeds)==set(learned)==set(stems)
        values=np.zeros((132,2,4));pred={}
        for i,s in enumerate(stems):
            with np.load(old_maps[f]/f'{s}.npz',allow_pickle=False) as z:
                probability,matrix=z['probability'],z['matrix']
            replay=refine_instances(seeds[s],probability,matrix,threshold=.5,prior_weight=0.)
            assert replay==learned[s],f'Exact old native refinement replay failed: {f}/{s}'
            baseline=combine(seeds[s],replay,.5)
            pred[s]=contours(seeds[s],probability,matrix,THRESHOLD)
            assert len(pred[s])==len(seeds[s])
            for j,masks in enumerate([baseline,pred[s]]):
                for gt in truth[s]:values[i,j]+=counts(gt,masks)
            if (i+1)%30==0:print('PROBABILITY_CUT_PROGRESS',f,i+1,132,flush=True)
        with np.load(inputs/f'{f}-counts.npz',allow_pickle=False) as z:
            lookup=dict(zip(z['stems'].tolist(),z['counts'][:,2]))
            np.testing.assert_array_equal(values[:,0],np.asarray([lookup[s] for s in stems]))
        report=paired_report(stems,values);report.update(original_count_vectors_replayed_exactly=True,old_native_refiner_masks_replayed_exactly=True)
        families[f]=report;write(out/f'{f}-quality.json',report);write(out/f'{f}-val-predictions.json',pred)
        np.savez_compressed(out/f'{f}-per-image-counts.npz',stems=stems,counts=values)
        print('PROBABILITY_CUT_QUALITY',f,report['delta_pq'],flush=True)
        if report['delta_pq']<=0:break
    positive=len(families)==2 and all(r['delta_pq']>0 for r in families.values())
    quality=dict(status='probability_cut_positive' if positive else 'probability_cut_rejected',
        improvement_indication=positive,threshold=THRESHOLD,plan_sha256=sha('configs/incumbent-probability-cut-20261023.json'),
        families=families,physical_images=132,expert_records=211,caveat=plan['caveat'],
        test_inputs_used=False,CSV_exported=False,new_neural_inference=False,submissions_used=0)
    write(out/'quality.json',quality)
    if not positive:
        write(out/'complete.json',quality);return
    base=read(inputs/'original-test-seeds.json');reference=read(inputs/'incumbent-test-predictions.json')
    expected={p.stem for p in (root/'test/test_images').glob('*.jpeg')}
    assert len(expected)==180 and set(base)==set(reference)==expected
    predicted={};replayed={}
    for i,s in enumerate(sorted(expected),1):
        probs=[];matrices=[]
        for seed in [20261022,20261023]:
            with np.load(full/f'seed-{seed}/maps/test'/f'{s}.npz',allow_pickle=False) as z:
                probs.append(z['probability'].astype(np.float32));matrices.append(z['matrix'])
        np.testing.assert_array_equal(matrices[0],matrices[1])
        probability=np.rint(np.mean(probs,axis=0)).astype(np.uint8)
        replayed[s]=contours(base[s],probability,matrices[0],.5)
        assert replayed[s]==reference[s],f'Original incumbent replay failed: {s}'
        predicted[s]=contours(base[s],probability,matrices[0],THRESHOLD)
        assert len(predicted[s])==len(reference[s])
        if i%30==0:print('PROBABILITY_CUT_TEST',i,180,flush=True)
    original=out/'original-replayed.csv';write_csv(replayed,original)
    assert sha(original)=='08ebf13920d232b8f9b94c116bd540a37488fa0088b1e8c1812f45e7a08a0d54'
    records=out/'test-predictions.json';write(records,predicted)
    csv=out/'candidate.csv';write_csv(predicted,csv);integrity=verify(csv,records,root/'test/test_images')
    assert integrity['instances']==1342 and integrity['csv_sha256']!=sha(original)
    result=dict(status='probability_cut_CSV_audited',quality_sha256=sha(out/'quality.json'),plan_sha256=quality['plan_sha256'],
        source_sha256=sha(Path(__file__)),improvement_indication=True,original_all180_masks_and_CSV_replayed_exactly=True,
        original_all132_native_refiner_masks_and_count_vectors_replayed_exactly=True,
        original_detection_counts_and_identities_preserved=True,audit=integrity,
        new_training=False,new_neural_inference=False,training_performed=False,test_inputs_used_for_cached_inference_only=True,
        CSV_exported=True,submissions_used=0,competition_submission_enabled=False,expected_public_pq=None,caveat=plan['caveat'])
    write(out/'complete.json',result);print('PROBABILITY_CUT_CANDIDATE_READY',integrity['csv_sha256'],flush=True)
