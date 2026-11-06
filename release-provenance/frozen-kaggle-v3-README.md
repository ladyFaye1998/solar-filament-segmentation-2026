# Solar filament instance ensemble: local review candidate

Open reproduce_owned.ipynb and run all cells with the official competition data
attached, or set FILAMENT_DATA and FILAMENT_BUNDLE. The notebook loads the eleven
checksummed included weights automatically. Source-only hosting can use real
published URLs in weights-location.json; those URLs are not published yet.

The default is full 180-image inference from JPEGs. CPU is supported but slower.
GPU refinement is float32 without autocast or TF32. See the notebook for optional
preprocessing, training and expert-record evaluation. Every output path is new
and writable; attached source and weights may be read-only.
FILAMENT_SHARDS=2 enables two independent CPU image partitions on a host with
four logical CPUs and at least 12 GiB free RAM; they merge only fresh predictions.

This pipeline scored public PQ 0.39. The separate 0.57 entry's private detector
is not included. A staged full neural replay exists; an integrated two-image probe
is not a full notebook reproduction. Newly generated CSVs are not automatically
submitted or assigned the historical public score.

Source: AGPL-3.0. Preserve THIRD_PARTY_NOTICES.md and licenses/. Competition data,
external frames, credentials and archived answer tables are not bundled. The
technical-report.pdf is a four-page review candidate in the official template.
Final model choice, clean complete execution, public hosting and the organizer
form remain release work.
