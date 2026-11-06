# Solar filament instance segmentation

This private preview preserves the participant-trained eleven-weight baseline.
A fresh full 180-image neural replay scored public PQ **0.39** on 6 November
2026 (submission 58067341). The separate historical 0.57 entry depends on an
unavailable private detector and is not reproduced by this source.

Run `reproduce_owned.ipynb` with the official competition data and this source.
The notebook loads included weights or fetches missing weights automatically
from `weights-location.json`, verifies their sizes and SHA-256 hashes, and runs
JPEG-only inference. Release assets and this repository are currently private;
unauthenticated automatic downloading has not been verified. A complete clean
default notebook run is still pending. See `RELEASE_STATUS.json` for the exact
scope of the existing verification.

The default runs all 180 test images. Set `FILAMENT_DATA` and `FILAMENT_BUNDLE`
if needed. GPU refinement uses float32 without autocast or TF32. The notebook
also includes preprocessing, optional training and expert-record evaluation.
The report is a four-page review draft, not a final selected award method.

Source is AGPL-3.0; preserve `LICENSE`, `THIRD_PARTY_NOTICES.md` and `licenses/`.
Competition images, annotations, GONG frames, credentials and prediction tables
are not bundled. Historical training metadata and data-selection identifiers
are included for provenance. Exact retraining byte identity is not promised.

The frozen Kaggle V3 bundle is preserved separately. This hosting variant only
changes the README and weight locations, with corresponding manifest hashes.
New native-model and GONG experiments are still running and are not selected
or included here. Final model choice, public access and competition disclosure,
complete clean execution, final report and the organizer form remain pending.
