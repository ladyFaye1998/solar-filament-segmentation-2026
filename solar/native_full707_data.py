"""Full official707-photo refit selection; no heldout-accuracy claims.

This deliberately requires a separate index schema. Geometry/photometry reuse
the frozen native training functions; no alternative-expert GT sets are merged.
"""
from collections import defaultdict
from functools import lru_cache
import hashlib

import numpy as np
from solar.native_maskrcnn_data import augment, geometry


class Full707PhotoExpertSequence:
    def __init__(self, index, seed=20261024, epochs=60):
        assert index['schema'] == 'owned_native_full707_input_index_v1'
        assert index['training_photos'] == 707 and index['training_expert_records'] == 1154
        assert index['validation_photos'] == index['validation_expert_records'] == 0
        assert not index['independent_validation_available_for_this_refit']
        self.seed, self.epochs = seed, epochs
        assert epochs == 60
        self.groups = defaultdict(list)
        for record in index['records']:
            assert record['split'] == 'train' and record['previous_split'] in ['train', 'validation']
            self.groups[record['stem']].append(record)
        self.photos = sorted(self.groups)
        assert len(self.photos) == 707 and sum(map(len, self.groups.values())) == 1154
        for records in self.groups.values():
            records.sort(key=lambda row: row['expert_record_id'])
            assert len(records) in [1, 2, 3] and epochs % len(records) == 0
        self.offsets = {stem: int(hashlib.sha256(f'{seed}:{stem}'.encode()).hexdigest()[:8], 16) % len(self.groups[stem])
                        for stem in self.photos}

    def __len__(self):
        return len(self.photos) * self.epochs

    @lru_cache(maxsize=3)
    def order(self, epoch):
        assert 0 <= epoch < self.epochs
        return np.random.default_rng(self.seed + epoch).permutation(len(self.photos))

    def record(self, global_index):
        assert 0 <= global_index < len(self)
        epoch, position = divmod(global_index, len(self.photos))
        stem = self.photos[int(self.order(epoch)[position])]
        records = self.groups[stem]
        return records[(epoch + self.offsets[stem]) % len(records)]
