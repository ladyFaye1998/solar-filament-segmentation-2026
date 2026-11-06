"""Verify the actual CSV against complete image inference records before upload."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from pycocotools import mask as mask_util
from solar.metrics import audit,encode


def verify(csv_path,prediction_path,test_images):
    predictions=json.loads(prediction_path.read_text())
    expected={path.stem for path in test_images.glob('*.jpeg')}
    assert len(expected)==180
    with csv_path.open(newline='',encoding='ascii') as stream:
        reader=csv.DictReader(stream)
        assert reader.fieldnames==['filament_id','segmentation_rle']
        rows=list(reader)
    expected_rows=[{'filament_id':f'{stem}_{index}','segmentation_rle':rle['counts']}
                   for stem,rles in sorted(predictions.items()) for index,rle in enumerate(rles,1)]
    assert rows==expected_rows,'CSV differs from recorded real inference predictions'
    assert len({row['filament_id'] for row in rows})==len(rows)
    for row in rows:
        stem,index=row['filament_id'].rsplit('_',1)
        assert stem in expected and int(index)>0
        rle={'size':[2048,2048],'counts':row['segmentation_rle']}
        mask=mask_util.decode(rle)
        assert mask.any() and encode(mask)==rle,'Invalid/noncanonical RLE'
    report={**audit(predictions,expected),'csv_rows':len(rows),
            'csv_sha256':hashlib.sha256(csv_path.read_bytes()).hexdigest(),
            'predictions_sha256':hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
            'csv_matches_inference_records':True,'csv_roundtrips_at_native_resolution':True}
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv',type=Path,required=True)
    parser.add_argument('--predictions',type=Path,required=True)
    parser.add_argument('--images',type=Path,default=Path('data/raw/MAGFiLO_1.0_Kaggle_2026/test/test_images'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=verify(args.csv,args.predictions,args.images)
    args.output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
