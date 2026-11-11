# Solar filament instance segmentation

This package reproduces Kaggle submission **58067341**, which scored **0.39
public panoptic quality** in the Solar Filament Segmentation Challenge 2026.
The [four-page technical report](technical-report.pdf) documents the model,
training data, postprocessing and validation.

## Reproduce the predictions

1. Obtain the official competition data and extract `MAGFiLO_1.0_Kaggle_2026`.
2. Open `reproduce_owned.ipynb` from this repository in a Python 3.12 environment.
   Set `FILAMENT_DATA` to the extracted data directory when running outside
   Kaggle. Set `FILAMENT_BUNDLE` to the repository directory if the notebook
   runs from another working directory.
3. Run the notebook. Its default installer uses the pinned `requirements.txt`
   and `requirements-training.txt`. The bootstrap fetches eleven checkpoints
   from this repository's `owned-baseline-20261106` release and checks each
   file's size and SHA-256. The default inference processes all 180 test JPEGs
   and writes `submission.csv` in the printed output directory.

The verified run used CUDA with float32 refinement, autocast disabled and TF32
disabled. It completed in 46.6 minutes in a fresh environment and produced
1,270 disjoint instances. Its CSV matches Kaggle's stored submission byte for
byte. The expected CSV SHA-256 is
`83139e60ad3392fbdf4c06e1a287fdfe78c79dc23d7c31114934f97c3119bbff`.
See `verification/clean-notebook.json` and `verification/scored-replay.json`.

The notebook includes optional training and official-label evaluation cells.
Training recipes, partitions and external GONG frame identifiers are recorded
in `training-provenance/` and `configs/`. The historical 565/132 split and the
later 524/183 whole-month comparisons are distinct evaluation cohorts.

Source is AGPL-3.0. See `LICENSE`, `THIRD_PARTY_NOTICES.md` and `licenses/`.
Obtain competition data and external GONG observations under their respective
source terms.
