"""Pure-array training selection and geometry; no model or test data access."""
from collections import defaultdict
from functools import lru_cache
import hashlib
import numpy as np


class PhotoExpertSequence:
    def __init__(self,index,seed=20261024,epochs=60):
        self.seed,self.epochs=seed,epochs
        self.groups=defaultdict(list)
        for record in index['records']:
            if record['split']=='train':self.groups[record['stem']].append(record)
        self.photos=sorted(self.groups)
        assert len(self.photos)==533 and sum(map(len,self.groups.values()))==882
        for records in self.groups.values():records.sort(key=lambda r:r['expert_record_id'])
        assert {p[:6] for p in self.photos}.isdisjoint(r['stem'][:6] for r in index['records'] if r['split']=='validation')
        self.offsets={p:int(hashlib.sha256(f'{seed}:{p}'.encode()).hexdigest()[:8],16)%len(self.groups[p]) for p in self.photos}
        assert epochs%6==0

    def __len__(self):return len(self.photos)*self.epochs

    @lru_cache(maxsize=3)
    def order(self,epoch):
        assert 0<=epoch<self.epochs
        return np.random.default_rng(self.seed+epoch).permutation(len(self.photos))

    def record(self,global_index):
        assert 0<=global_index<len(self)
        epoch,position=divmod(global_index,len(self.photos))
        stem=self.photos[int(self.order(epoch)[position])]
        records=self.groups[stem]
        return records[(epoch+self.offsets[stem])%len(records)]


def geometry(image,masks,boxes,hflip=False,vflip=False,quarter_turns=0):
    """Pixel-exact D4 transforms with exclusive XYXY native box coordinates."""
    boxes=np.asarray(boxes,dtype=np.float32).copy()
    height,width=image.shape[:2]
    if hflip:
        image=image[:,::-1];masks=masks[:,:,::-1]
        boxes[:,[0,2]]=width-boxes[:,[2,0]]
    if vflip:
        image=image[::-1];masks=masks[:,::-1,:]
        boxes[:,[1,3]]=height-boxes[:,[3,1]]
    for _ in range(quarter_turns%4):
        image=np.rot90(image,1,axes=(0,1));masks=np.rot90(masks,1,axes=(1,2))
        boxes=np.stack([boxes[:,1],width-boxes[:,2],boxes[:,3],width-boxes[:,0]],axis=1)
        height,width=width,height
    return np.ascontiguousarray(image),np.ascontiguousarray(masks),np.ascontiguousarray(boxes)


def augment(image,masks,boxes,seed,global_index):
    generator=np.random.default_rng(seed+global_index)
    image,masks,boxes=geometry(image,masks,boxes,
        bool(generator.integers(2)),bool(generator.integers(2)),int(generator.integers(4)))
    gamma=float(generator.uniform(.8,1.2));contrast=float(generator.uniform(.8,1.2));gain=float(generator.uniform(.9,1.1))
    values=np.arange(256,dtype=np.float32)/255.
    values=np.clip(((values**gamma-.5)*contrast+.5)*gain,0,1)
    lookup=np.rint(values*255).astype(np.uint8)
    return np.ascontiguousarray(lookup[image]),masks,boxes
