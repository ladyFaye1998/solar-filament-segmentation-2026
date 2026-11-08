# Source and third-party notices

This participant implementation integrates Ultralytics and is distributed under
GNU AGPL version 3; see LICENSE. Preserve upstream notices. The complete source,
training scripts, recipes and dependency versions accompany the weights.

Upstream software (not authored by the participant):

- Ultralytics 8.4.183 and generic YOLOv8 COCO initializers: AGPL-3.0.
  https://github.com/ultralytics/ultralytics/tree/v8.4.183
- Torchvision 0.26: BSD-3-Clause. Generic COCO Mask R-CNN initialization comes
  from the official PyTorch model host.
  https://github.com/pytorch/vision/tree/v0.26.0
- Segmentation Models PyTorch 0.5.0: MIT, Pavel Yakubovskiy.
  https://github.com/qubvel-org/segmentation_models.pytorch/tree/v0.5.0
- timm 1.0.31: Apache-2.0, Ross Wightman and contributors.
  https://github.com/huggingface/pytorch-image-models/tree/v1.0.31

Copies of these upstream license texts are under licenses/. Installed libraries
retain their own bundled notices and licenses. The source license does not grant
rights to third-party data. Competition images/annotations and extra GONG frames
are supplied separately under their source terms; they are not redistributed here.
The MAGFiLO source publication is doi:10.1038/s41597-024-03876-y.

GONG data are obtained by the NSO Integrated Synoptic Program, managed by the
National Solar Observatory, operated by AURA under a cooperative agreement with
NSF and with contribution from NOAA. The network is hosted by Big Bear Solar
Observatory, High Altitude Observatory, Learmonth Solar Observatory, Udaipur
Solar Observatory, Instituto de Astrofisica de Canarias and Cerro Tololo
Interamerican Observatory.

The separate HDJoJo-derived 0.57 entry is not included. Its private detector
weights have not been recovered. This bundle contains the participant-trained
eleven-model 0.39 pipeline. The inherited historical 565/132 partition is retained
for provenance and is not represented as a newly unbiased whole-month holdout.

This candidate has not been publicly released or submitted as a final
award package. The unchanged notebook code cells completed the clean default
installer and all 180 images with included weights; anonymous public weight
fetching, final selection and the organizer form remain review gates.
