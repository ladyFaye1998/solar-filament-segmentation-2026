"""Owned full-resolution Mask R-CNN V2 with generic COCO initialization.

Uses public Torchvision models and ROI target projection (BSD-3-Clause).
Only the final class/box and mask-logit predictors are replaced. The pretrained
backbone, FPN, RPN, box feature head, mask feature head and deconvolution remain.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image
from pycocotools import mask as mask_util
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import roi_heads
from solar.rfdetr_checkpoint import state_digest

PRETRAIN_URL='https://download.pytorch.org/models/maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth'


def binary_mask_loss(logits, target):
    logits, target=logits.float(), target.float()
    if target.numel()==0:
        return logits.sum()*0
    bce=F.binary_cross_entropy_with_logits(logits,target)
    probability=logits.sigmoid()
    numerator=2*(probability*target).sum(dim=(-2,-1))+1.
    denominator=probability.sum(dim=(-2,-1))+target.sum(dim=(-2,-1))+1.
    return bce+(1-numerator/denominator).mean()


def maskrcnn_bce_dice(mask_logits, proposals, gt_masks, gt_labels, mask_matched_idxs):
    size=mask_logits.shape[-1]
    labels=torch.cat([label[index] for label,index in zip(gt_labels,mask_matched_idxs)],dim=0)
    targets=torch.cat([roi_heads.project_masks_on_boxes(mask,proposal,index,size)
        for mask,proposal,index in zip(gt_masks,proposals,mask_matched_idxs)],dim=0)
    if targets.numel()==0:
        return mask_logits.sum()*0
    selected=mask_logits[torch.arange(labels.shape[0],device=labels.device),labels]
    return binary_mask_loss(selected,targets)


def install_mask_loss():
    roi_heads.maskrcnn_loss=maskrcnn_bce_dice


def build_model(pretrained, resolution=2048, sampled_rois=128):
    weights=MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1 if pretrained else None
    if pretrained:
        assert weights.url==PRETRAIN_URL
    model=maskrcnn_resnet50_fpn_v2(weights=weights,weights_backbone=None,progress=False,
        trainable_backbone_layers=5,min_size=resolution,max_size=resolution,
        box_batch_size_per_image=sampled_rois,box_positive_fraction=.25,
        rpn_pre_nms_top_n_train=1000,rpn_post_nms_top_n_train=512,
        rpn_pre_nms_top_n_test=1000,rpn_post_nms_top_n_test=512,
        box_detections_per_img=100)
    preserved=dict(backbone=state_digest(model.backbone.state_dict()),
        rpn=state_digest(model.rpn.state_dict()),mask_features=state_digest(model.roi_heads.mask_head.state_dict()),
        mask_deconvolution=state_digest(model.roi_heads.mask_predictor.conv5_mask.state_dict()))
    in_features=model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor=FastRCNNPredictor(in_features,2)
    channels=model.roi_heads.mask_predictor.mask_fcn_logits.in_channels
    predictor=nn.Conv2d(channels,2,1)
    nn.init.kaiming_normal_(predictor.weight,mode='fan_out',nonlinearity='relu')
    nn.init.zeros_(predictor.bias)
    model.roi_heads.mask_predictor.mask_fcn_logits=predictor
    model.roi_heads.mask_roi_pool.output_size=(56,56)
    assert preserved==dict(backbone=state_digest(model.backbone.state_dict()),
        rpn=state_digest(model.rpn.state_dict()),mask_features=state_digest(model.roi_heads.mask_head.state_dict()),
        mask_deconvolution=state_digest(model.roi_heads.mask_predictor.conv5_mask.state_dict()))
    install_mask_loss()
    return model,preserved


def training_fixture(record,image_root):
    image_root=Path(image_root).resolve()
    path=(image_root/record['filename']).resolve()
    assert path.parent==image_root
    with Image.open(path) as source:
        assert source.size==(2048,2048)
        rgb=np.asarray(source.convert('RGB')).copy()
    image=torch.from_numpy(rgb).permute(2,0,1).float()/255.
    boxes=torch.tensor([[x,y,x+w,y+h] for x,y,w,h in record['boxes_xywh']],dtype=torch.float32)
    masks=torch.from_numpy(np.stack([mask_util.decode(rle) for rle in record['rles']])).to(torch.uint8)
    target=dict(boxes=boxes,labels=torch.ones(len(boxes),dtype=torch.int64),masks=masks,
        image_id=torch.tensor([record['source_id']]),area=torch.tensor(record['areas'],dtype=torch.float32),
        iscrowd=torch.zeros(len(boxes),dtype=torch.int64))
    assert len(boxes)==len(masks)>0 and masks.shape[-2:]==(2048,2048)
    return image,target


def synthetic_negative():
    return torch.full((3,2048,2048),.5),dict(boxes=torch.zeros((0,4),dtype=torch.float32),
        labels=torch.zeros(0,dtype=torch.int64),masks=torch.zeros((0,2048,2048),dtype=torch.uint8),
        image_id=torch.tensor([99999]),area=torch.zeros(0),iscrowd=torch.zeros(0,dtype=torch.int64))
