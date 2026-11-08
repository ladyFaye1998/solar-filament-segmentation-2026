# Solar filament instance segmentation

This review candidate preserves the participant-trained eleven-weight baseline.
A fresh full 180-image neural replay scored public PQ **0.39** on 6 November
2026, submission **58067341**. The separate historical 0.57 entry depends on an
unavailable private detector and is not reproduced by this source.

Run `reproduce_owned.ipynb` with Python 3.12, the official competition data and
this source. Its default installer uses the pinned requirements. Set
`FILAMENT_DATA` and `FILAMENT_BUNDLE` if needed. The default processes all 180
supplied JPEGs. GPU refinement uses float32 with autocast and TF32 disabled.

The exact five code cells and all eleven weights completed a clean default
installation and full notebook execution in a fresh isolated environment.
The observed run took 46.6 minutes and reproduced the scored CSV byte for byte:
`83139e60ad3392fbdf4c06e1a287fdfe78c79dc23d7c31114934f97c3119bbff`.
See `verification/clean-notebook.json`. This candidate updates markdown,
documentation, verification records and the reviewed four-page report; its code
cells, numerical source, requirements, model membership and weights are unchanged.
No new full neural run was needed or performed for these documentation changes.

The notebook loads included weights or fetches missing weights automatically
from `weights-location.json`, verifying every size and SHA-256. The repository
and its eleven weight assets are currently private. The clean execution used
included weights; anonymous HTTPS downloading remains unverified. This candidate
does not certify public access, final selection or award eligibility.

Optional notebook cells provide preprocessing, training and official-label
evaluation. Historical training recipes and external-frame identifiers are
retained for provenance. Retraining produces new weights; byte identity is not
promised. The historical 565/132 split is distinct from the later 524/183
whole-month comparisons. Full-707 deployment models cannot establish held-out
validation on those same training photographs.

Source is AGPL-3.0; preserve `LICENSE`, `THIRD_PARTY_NOTICES.md` and `licenses/`.
Competition images, annotations, GONG frames, credentials and prediction tables
are not bundled. Data must be obtained separately under their source terms.

The frozen Kaggle V3 bundle and existing private GitHub release are preserved.
The accompanying report is the visually reviewed four-page organizer-template
draft for this baseline. Later experiments are not selected or included here.
Final model choice, public access and competition disclosure, final report
approval, the organizer form and any prize documents remain pending.
