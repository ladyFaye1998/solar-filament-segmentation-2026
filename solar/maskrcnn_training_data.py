"""Official-TRAIN-only, native-mask input for TensorFlow Model Garden.

TensorFlow is imported only when serializing TF Examples. Each annotator is a
separate example; no instance set ever combines different experts' labels.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shutil

import ijson
import numpy as np
from PIL import Image
from pycocotools import mask as mask_util

from solar.metrics import polygon_rle

ANNOTATION_SHA = '5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1'
PARTITION_SHA = 'c672a33754e4be9e053e3131960e37481b8a08fa44581dfd3ed646cc2b04e462'
ANNOTATION_NAME = 'MAGFiLO_1.0_Annotations_kaggle2026_train.json'


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def build_index(annotation, partition_path):
    """Stream the official polygons once; retain compact RLEs and source IDs."""
    annotation, partition_path = Path(annotation), Path(partition_path)
    assert digest(annotation) == ANNOTATION_SHA
    assert digest(partition_path) == PARTITION_SHA
    split = json.loads(partition_path.read_text())
    train, validation = set(split['training_ids']), set(split['validation_ids'])
    assert len(train) == 533 and len(validation) == 174 and not train & validation
    assert {s[:6] for s in train}.isdisjoint(s[:6] for s in validation)
    with annotation.open('rb') as stream:
        images = {row['id']: row for row in ijson.items(stream, 'images.item', use_float=True)}
    assert len(images) == 1154
    multiplicity = Counter(Path(row['file_name']).stem for row in images.values())
    assert set(multiplicity) == train | validation
    records = {}
    for source_id, expert_id in enumerate(sorted(images), 1):
        image = images[expert_id]
        filename = image['file_name']
        assert Path(filename).name == filename and '\\' not in filename and '/' not in filename
        stem = Path(filename).stem
        assert image['height'] == image['width'] == 2048
        records[expert_id] = dict(source_id=source_id, expert_record_id=expert_id,
            stem=stem, filename=filename, height=2048, width=2048,
            split='train' if stem in train else 'validation',
            experts_for_photo=multiplicity[stem], photo_weight=1.0/multiplicity[stem],
            annotation_ids=[], original_chirality_categories=[], rles=[], boxes_xywh=[], areas=[])
    annotation_ids = set()
    with annotation.open('rb') as stream:
        for annotation in ijson.items(stream, 'annotations.item', use_float=True):
            assert annotation['id'] not in annotation_ids
            annotation_ids.add(annotation['id'])
            record = records[annotation['image_id']]
            assert annotation['iscrowd'] == 0 and annotation['category_id'] in (1, 2, 3)
            rle = polygon_rle(annotation, record['height'], record['width'])
            area = int(mask_util.area(rle))
            box = mask_util.toBbox(rle).astype(float).tolist()
            x, y, width, height = box
            assert area > 0 and 0 <= x < x+width <= 2048 and 0 <= y < y+height <= 2048
            record['annotation_ids'].append(annotation['id'])
            record['original_chirality_categories'].append(annotation['category_id'])
            record['rles'].append(rle)
            record['boxes_xywh'].append(box)
            record['areas'].append(area)
    assert len(annotation_ids) == 8199
    assert sum(r['split'] == 'train' for r in records.values()) == 882
    assert sum(r['split'] == 'validation' for r in records.values()) == 272
    # Official categories encode chirality, all belonging to foreground filament.
    # The three original values remain in the manifest for provenance.
    return dict(schema='official_solar_MaskRCNN_input_index_v1',
        annotation_sha256=ANNOTATION_SHA, partition_sha256=PARTITION_SHA,
        training_photos=533, validation_photos=174,
        training_expert_records=882, validation_expert_records=272,
        annotation_instances=8199, native_shape=[2048, 2048],
        num_classes_including_background=2, foreground_label=1,
        category_mapping={'1': 1, '2': 1, '3': 1},
        calendar_month_disjoint=True, expert_records_combined=False,
        test_inputs_used=False, test_labels_used=False,
        records=list(records.values()))


def binary_png(rle):
    """PNG contains 0/1, since Model Garden casts decoded bytes to float32."""
    native = mask_util.decode(rle)
    assert native.dtype == np.uint8 and native.shape == tuple(rle['size'])
    assert np.all(native <= 1) and native.any()
    buffer = io.BytesIO()
    Image.fromarray(native).save(buffer, format='PNG')
    encoded = buffer.getvalue()
    decoded = np.asarray(Image.open(io.BytesIO(encoded)))
    assert np.array_equal(decoded, native)
    roundtrip = mask_util.encode(np.asfortranarray(decoded))
    assert roundtrip['size'] == rle['size']
    assert roundtrip['counts'].decode('ascii') == rle['counts']
    return encoded


def feature_values(record, encoded_jpeg):
    """Typed Model Garden features, before TensorFlow protobuf serialization."""
    with Image.open(io.BytesIO(encoded_jpeg)) as image:
        assert image.format == 'JPEG' and image.size == (record['width'], record['height'])
        assert image.mode in ('L', 'RGB')
    width, height = record['width'], record['height']
    n = len(record['rles'])
    assert len(record['boxes_xywh']) == len(record['areas']) == n
    assert record['source_id'] > 0 and record['experts_for_photo'] >= 1
    boxes = record['boxes_xywh']
    masks = [binary_png(rle) for rle in record['rles']]
    return {
        'image/height': ('int64', [height]),
        'image/width': ('int64', [width]),
        'image/filename': ('bytes', [record['filename'].encode()]),
        'image/source_id': ('bytes', [str(record['source_id']).encode()]),
        'image/key/sha256': ('bytes', [hashlib.sha256(encoded_jpeg).hexdigest().encode()]),
        'image/encoded': ('bytes', [encoded_jpeg]),
        'image/format': ('bytes', [b'jpg']),
        'image/object/bbox/xmin': ('float', [x/width for x, y, w, h in boxes]),
        'image/object/bbox/xmax': ('float', [(x+w)/width for x, y, w, h in boxes]),
        'image/object/bbox/ymin': ('float', [y/height for x, y, w, h in boxes]),
        'image/object/bbox/ymax': ('float', [(y+h)/height for x, y, w, h in boxes]),
        'image/object/class/label': ('int64', [1]*n),
        'image/object/class/text': ('bytes', [b'filament']*n),
        'image/object/is_crowd': ('int64', [0]*n),
        'image/object/area': ('float', list(record['areas'])),
        'image/object/mask': ('bytes', masks),
        'solar/expert_record_id': ('bytes', [record['expert_record_id'].encode()]),
        'solar/physical_stem': ('bytes', [record['stem'].encode()]),
        'solar/photo_weight': ('float', [record['photo_weight']]),
    }


def tensorflow_example(features):
    import tensorflow as tf
    constructors = {'int64': tf.train.Int64List, 'float': tf.train.FloatList,
                    'bytes': tf.train.BytesList}
    return tf.train.Example(features=tf.train.Features(feature={
        key: tf.train.Feature(**{kind+'_list': constructors[kind](value=values)})
        for key, (kind, values) in features.items()}))


def balanced_groups(index):
    """Sampling a group then an expert gives each physical photo equal weight."""
    groups = defaultdict(list)
    for record in index['records']:
        if record['split'] == 'train':
            groups[record['experts_for_photo']].append(record)
    counts = {str(k): len({r['stem'] for r in rows}) for k, rows in sorted(groups.items())}
    assert sum(counts.values()) == 533
    weights = {k: n/533 for k, n in counts.items()}
    for k, rows in groups.items():
        assert len(rows) == k*counts[str(k)]
        # Independent check of each photo's marginal sampling probability.
        probability = weights[str(k)]*k/len(rows)
        assert abs(probability-1/533) < 1e-15
    return groups, weights, counts


def export_tfrecords(dataset_root, partition_path, output, shards_per_train_group=8):
    """Run on roomy remote CPU storage; all writes are under a new output path."""
    import tensorflow as tf
    dataset_root, output = Path(dataset_root), Path(output)
    assert not output.exists(), 'Preserve prior exports and partial diagnostics'
    assert shutil.disk_usage(output.parent).free >= 8*1024**3, 'Full export requires 8GiB free disk'
    index = build_index(dataset_root/'train'/ANNOTATION_NAME, partition_path)
    groups, weights, photo_counts = balanced_groups(index)
    output.mkdir()
    (output/'index.json').write_text(json.dumps(index, separators=(',', ':')))
    image_root = (dataset_root/'train/train_images').resolve()
    cohorts = [(f'train-experts-{k}', rows, shards_per_train_group) for k, rows in sorted(groups.items())]
    cohorts.append(('validation', [r for r in index['records'] if r['split']=='validation'], 8))
    files = {}
    for name, records, shards in cohorts:
        paths = [output/f'{name}-{i:05d}-of-{shards:05d}.tfrecord' for i in range(shards)]
        writers = [tf.io.TFRecordWriter(str(p)) for p in paths]
        try:
            for i, record in enumerate(records):
                image = (image_root/record['filename']).resolve()
                assert image.parent == image_root
                features = feature_values(record, image.read_bytes())
                writers[i % shards].write(tensorflow_example(features).SerializeToString())
                if i % 50 == 0:
                    print('OFFICIAL_TRAIN_ADAPTER', name, i, len(records), flush=True)
        finally:
            for writer in writers:
                writer.close()
        for p in paths:
            files[p.name] = dict(bytes=p.stat().st_size, sha256=digest(p))
    config = dict(input_path={str(k): str(output/f'train-experts-{k}-*.tfrecord') for k in groups},
                  weights=weights, group_physical_photos=photo_counts,
                  validation_path=str(output/'validation-*.tfrecord'),
                  requires_repeated_shuffled_training_datasets=True,
                  pinned_framework_decoder_validation_required=True)
    (output/'sampling.json').write_text(json.dumps(config, indent=2))
    report = dict(status='native_TFRecord_export_complete_decoder_proof_still_required',
        at_utc=datetime.now(timezone.utc).isoformat(), tensorflow=tf.__version__,
        annotation_sha256=ANNOTATION_SHA, partition_sha256=PARTITION_SHA,
        source_sha256=digest(Path(__file__)), index_sha256=digest(output/'index.json'),
        training_photos=533, training_records=882, validation_photos=174,
        validation_records=272, instances=8199, binary_native_mask_roundtrips=True,
        separate_expert_records=True, physical_photo_balancing=True,
        files=files, test_inputs_used=False, model_fitted=False, submissions_used=0)
    (output/'complete.json').write_text(json.dumps(report, indent=2))
    return report
