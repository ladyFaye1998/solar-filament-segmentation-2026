"""Run the fixed neural chain, optionally in two independent CPU partitions.

Partitions are computed from JPEGs in this invocation. Their new masks are merged
only after every neural execution completes; no archived answers are accepted.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

BUNDLE=Path(__file__).resolve().parent


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--shards',type=int,choices=(1,2),default=1)
    parser.add_argument('--stems',nargs='*')
    parser.add_argument('--wall-seconds',type=float,default=39600)
    args=parser.parse_args()
    data,output=args.data.resolve(),args.output.resolve()
    command=[sys.executable,'-u',str(BUNDLE/'owned_reproduction.py'),'--data',str(data),'--device',args.device]
    if args.device.startswith('cuda'):command+=['--refiner-device',args.device,'--refiner-fp32']
    if args.shards==1:
        command+=['--output',str(output)]
        if args.stems:command+=['--stems',*args.stems]
        subprocess.run(command,cwd=BUNDLE,check=True)
        return
    assert args.device=='cpu','Parallel partitions are restricted to CPU; no automatic GPU memory overcommit'
    assert (os.cpu_count() or 1)>=4,'Two CPU partitions need at least four logical CPUs'
    import psutil
    assert psutil.virtual_memory().available>=12*1024**3,'Two CPU partitions need at least 12 GiB free host RAM'
    assert 120<=args.wall_seconds<=39600
    images=data/'test/test_images'
    full=sorted(p.stem for p in images.glob('*.jpeg'));assert len(full)==180
    stems=sorted(args.stems) if args.stems else full
    assert len(stems)==len(set(stems)) and set(stems)<=set(full) and len(stems)>=2
    output.mkdir(parents=True,exist_ok=False)
    jobs=[];logs=[];started=time.monotonic()
    try:
        for index in range(2):
            selected=stems[index::2]
            path=output/'parts'/f'part-{index:02d}'
            log=(output/f'part-{index:02d}.log').open('w',buffering=1);logs.append(log)
            own=subprocess.Popen(command+['--output',str(path),'--stems',*selected],cwd=BUNDLE,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=os.name!='nt',
                env=dict(os.environ,YOLO_CONFIG_DIR=str(output/f'settings-{index:02d}')))
            jobs.append((own,path,selected))
        while any(process.poll() is None for process,_,_ in jobs):
            assert time.monotonic()-started<args.wall_seconds,'CPU inference wall bound reached; no complete export'
            assert psutil.virtual_memory().available>768*1024**2,'Critical host RAM pressure; aborting only owned partitions'
            for process,_,_ in jobs:
                assert process.poll() in (None,0),'A neural partition failed; no complete export'
            print(json.dumps(dict(stage='cpu-neural-partitions',seconds=time.monotonic()-started,
                returncodes=[p.poll() for p,_,_ in jobs])),flush=True)
            time.sleep(30)
        assert all(p.returncode==0 for p,_,_ in jobs)
        predictions={};receipts=[]
        for _,path,selected in jobs:
            receipt=json.loads((path/'complete.json').read_text())
            assert receipt['images']==len(selected) and receipt['probe_only']
            assert receipt['neural_image_inference_completed'] and receipt['detectors']==8 and receipt['refiners']==3
            values=json.loads((path/'predictions.json').read_text())
            assert set(values)==set(selected) and not set(values)&set(predictions)
            assert digest(path/'predictions.json')==receipt['prediction_sha256']
            predictions.update(values);receipts.append(dict(path=str(path/'complete.json'),sha256=digest(path/'complete.json')))
        assert set(predictions)==set(stems)
        sys.path.insert(0,str(BUNDLE))
        from solar.metrics import audit
        from solar.yolo_run import write_csv
        integrity=audit(predictions,set(stems))
        prediction_path=output/'predictions.json'
        prediction_path.write_text(json.dumps(predictions,separators=(',',':')),encoding='utf-8')
        result=dict(integrity=integrity,images=len(stems),probe_only=bool(args.stems),detectors=8,refiners=3,
            neural_image_inference_completed=True,execution='Two fresh independent CPU partitions',
            partition_receipts=receipts,elapsed_seconds=time.monotonic()-started,
            model_manifest_sha256=digest(BUNDLE/'models.json'),prediction_sha256=digest(prediction_path),
            refiner_device='cpu',refiner_fp32=False,new_submissions=0,public_score=None,
            archived_answers_or_prior_predictions_used=False)
        if not args.stems:
            from scripts.audit_submission import verify
            write_csv(predictions,output/'submission.csv')
            result['csv_audit']=verify(output/'submission.csv',prediction_path,images)
        (output/'complete.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps(result,indent=2),flush=True)
    finally:
        for process,_,_ in jobs:
            if process.poll() is None:
                if os.name=='nt':subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],capture_output=True)
                else:os.killpg(process.pid,signal.SIGTERM)
        for process,_,_ in jobs:
            try:process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                if os.name=='nt':process.kill()
                else:os.killpg(process.pid,signal.SIGKILL)
                process.wait()
        for log in logs:log.close()


if __name__=='__main__':main()
