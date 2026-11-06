"""Prune tiny detached islands without removing any incumbent detection."""
import numpy as np
from pycocotools import mask as mu
from scipy import ndimage as ndi

from solar.metrics import encode


def prune(rle):
    native=mu.decode(rle)
    x,y,w,h=map(int,mu.toBbox(rle));local=native[y:y+h,x:x+w].astype(bool)
    labels,n=ndi.label(local,np.ones((3,3),dtype=np.uint8))
    if n<=1:return rle
    areas=np.bincount(labels.ravel());areas[0]=0
    keep=(areas>=.05*areas.max());keep[0]=False
    cleaned=keep[labels]
    if cleaned.sum()==local.sum() or cleaned.sum()<max(5,.5*local.sum()):return rle
    assert np.all(~cleaned|local)
    native[y:y+h,x:x+w]=cleaned
    return encode(native)


def transform(rles):
    return [prune(r) for r in rles]
