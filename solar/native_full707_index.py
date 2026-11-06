"""Reproduce the separately verified full707 index from the original index."""


def build_full707_index(source, source_index_sha256):
    assert source['schema'] == 'official_solar_MaskRCNN_input_index_v1'
    assert source['training_photos'] == 533 and source['validation_photos'] == 174
    assert len(source['records']) == 1154 and source['annotation_instances'] == 8199
    records = []
    for original in source['records']:
        assert original['split'] in ['train', 'validation']
        record = dict(original)
        record.update(previous_split=original['split'], split='train')
        records.append(record)
    return dict(schema='owned_native_full707_input_index_v1',
        source_index_sha256=source_index_sha256, original_month_partition_sha256=source['partition_sha256'],
        annotation_sha256=source['annotation_sha256'], training_photos=707, training_expert_records=1154,
        validation_photos=0, validation_expert_records=0, annotation_instances=8199,
        native_shape=[2048, 2048], num_classes_including_background=2, foreground_label=1,
        category_mapping=source['category_mapping'], expert_records_combined=False,
        independent_validation_available_for_this_refit=False,
        previous174_validation_labels_are_now_training_labels=True,
        test_inputs_used=False, test_labels_used=False, records=records)
