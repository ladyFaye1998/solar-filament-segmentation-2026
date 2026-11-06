"""Enforce organizer ruling 748045 before external frames receive train labels.

The one-hour test exclusion applies across every GONG site. Development dates
are also excluded to preserve the project's validation partition. Whitelisting
provides archive access; it does not waive either exclusion.
"""
from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re

GONG_STEM = re.compile(r"^\d{14}[BCLMTU]h$")
AUTHORITY = "https://www.kaggle.com/competitions/filament-segmentation-2026/discussion/748045#3536644"


def observation(stem: str) -> datetime:
    if not GONG_STEM.fullmatch(stem):
        raise ValueError(f"Invalid science-grade GONG observation ID: {stem!r}")
    return datetime.strptime(stem[:14], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)


def test_ids_from_root(root: Path) -> set[str]:
    stems = {p.stem for p in (Path(root) / "test/test_images").glob("*.jpeg")}
    if len(stems) != 180:
        raise ValueError("External training requires the full 180-image test timestamp exclusion list")
    for stem in stems:
        observation(stem)
    return stems


@dataclass(frozen=True)
class ExternalFrameGuard:
    training_ids: frozenset[str]
    validation_ids: frozenset[str]
    test_ids: frozenset[str]

    def __post_init__(self):
        if not self.training_ids or not self.test_ids:
            raise ValueError("Training sources and test timestamp exclusions must be explicit and nonempty")
        if self.training_ids & (self.validation_ids | self.test_ids):
            raise ValueError("Training source IDs overlap validation/test IDs")
        if {s[:8] for s in self.training_ids} & {s[:8] for s in self.validation_ids}:
            raise ValueError("Training and development dates overlap")
        for stem in self.training_ids | self.validation_ids | self.test_ids:
            observation(stem)
        object.__setattr__(self, "_test_seconds", sorted(observation(s).timestamp() for s in self.test_ids))
        object.__setattr__(self, "_validation_seconds", sorted(observation(s).timestamp() for s in self.validation_ids))
        object.__setattr__(self, "_validation_dates", {s[:8] for s in self.validation_ids})

    @staticmethod
    def _within(seconds: float, protected: list[float], window: float = 3600) -> bool:
        index = bisect_left(protected, seconds - window)
        return index < len(protected) and protected[index] <= seconds + window

    def rejection(self, frame: str, source: str, *, max_seconds: float = 150,
                  require_same_site: bool = True) -> str | None:
        try:
            seconds = observation(frame).timestamp()
            source_seconds = observation(source).timestamp()
        except ValueError:
            return "invalid_observation_id"
        if source not in self.training_ids:
            return "source_is_not_official_training_partition"
        if self._within(seconds, self._test_seconds):
            return "within_one_hour_of_test_across_sites"
        if frame[:8] in self._validation_dates or self._within(seconds, self._validation_seconds):
            return "development_date_or_one_hour_buffer"
        if frame == source:
            return "not_an_additional_observation"
        if require_same_site and frame[14] != source[14]:
            return "different_site_requires_registered_loader"
        if abs(seconds - source_seconds) > max_seconds:
            return "too_far_from_training_source"
        return None

    def metadata(self) -> dict:
        return dict(authority=AUTHORITY, test_images=len(self.test_ids),
                    test_exclusion_seconds=3600, exclusion_applies_across_sites=True,
                    development_dates_excluded=True, training_source_required=True)
