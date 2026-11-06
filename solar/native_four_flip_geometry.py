"""Sparse native four-flip geometry derived unchanged from checked operators.

No detector, training, file access, target labels or CSV generation.
"""
import numpy as np
from pycocotools import mask as mu
from scripts.audit_public_csv import decoded_runs


def canonical(rle):
    counts = rle['counts']
    return {'size':list(rle['size']), 'counts':counts.decode('ascii') if isinstance(counts,bytes) else counts}


def flip_rle(rle, view):
    """Map foreground intervals in column-major order, preserving every pixel."""
    assert 0 <= view < 4
    rle = canonical(rle)
    height, width = rle['size']; total = height*width
    assert height > 0 and width > 0
    if view == 0:
        return rle
    runs = decoded_runs(rle['counts'])
    assert sum(runs) == total
    intervals, offset = [], 0
    for index, length in enumerate(runs):
        end = offset+length
        if index % 2 and length:
            if view == 3:
                intervals.append((total-end,total-offset))
            else:
                start = offset
                while start < end:
                    column, y = divmod(start,height)
                    stop = min(end,(column+1)*height)
                    span = stop-start
                    if view == 1:
                        mapped = (width-1-column)*height+y
                        intervals.append((mapped,mapped+span))
                    else:
                        intervals.append((column*height+height-y-span,column*height+height-y))
                    start = stop
        offset = end
    intervals.sort()
    merged = []
    for start, end in intervals:
        assert 0 <= start < end <= total
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0],max(end,merged[-1][1]))
        else:
            merged.append((start,end))
    counts, offset = [], 0
    for start, end in merged:
        counts.extend([start-offset,end-start]); offset=end
    if offset < total:
        counts.append(total-offset)
    if not counts:
        counts=[total]
    return canonical(mu.frPyObjects({'size':[height,width],'counts':counts},height,width))


def majority(members, anchor):
    """The frozen majority/tie rule expressed as Boolean RLE intersections."""
    assert 1 <= len(members) <= 4 and members[0] == anchor
    if len(members) <= 2:
        # With two members all disagreements are ties, so the anchor wins.
        return canonical(anchor)
    if len(members) == 3:
        pairs = [mu.merge([members[a],members[b]],intersect=True) for a,b in [(0,1),(0,2),(1,2)]]
        return canonical(mu.merge(pairs,intersect=False))
    others=members[1:]
    anchor_and_any=mu.merge([anchor,mu.merge(others,intersect=False)],intersect=True)
    all_others=mu.merge(others,intersect=True)
    return canonical(mu.merge([anchor_and_any,all_others],intersect=False))


def exclusive_bank(rows, shape=(2048,2048), minimum_area=5):
    """Same descending-score ownership and tiny-mask rule, without dense masks."""
    height,width=shape
    occupied=None
    result=[]
    last_score=float('inf')
    for row in rows:
        score=float(row['score'])
        assert 0 <= score <= 1 and score <= last_score
        last_score=score
        assert tuple(row['rle']['size']) == shape
        if occupied is None:
            owned=canonical(row['rle'])
        else:
            # Swap foreground/background while retaining a canonical initial
            #run. Two leading zeros would describe the same pixels but could
            #propagate noncanonical counts through COCO merge.
            runs=decoded_runs(canonical(occupied)['counts'])
            complement_runs=runs[1:] if runs[0] == 0 else [0]+runs
            complement=mu.frPyObjects({'size':[height,width],
                'counts':complement_runs},height,width)
            owned=canonical(mu.merge([row['rle'],complement],intersect=True))
        if mu.area(owned) < minimum_area:
            continue
        occupied=owned if occupied is None else mu.merge([occupied,owned],intersect=False)
        result.append({'score':score,'rle':owned})
    return result


def fuse(views, shape=(2048,2048), maximum=100):
    assert len(views) == 4 and 0 < maximum <= 100
    ordered=[]
    for view,rows in enumerate(views):
        assert len(rows) <= 100
        for rank,row in enumerate(rows):
            score=float(row['score'])
            assert np.isfinite(score) and 0 <= score <= 1
            assert tuple(row['rle']['size']) == shape and mu.area(row['rle']) > 0
            ordered.append((score,view,rank,row['rle']))
    ordered.sort(key=lambda row:(-row[0],row[1],row[2]))
    clusters=[]
    for score,view,rank,rle in ordered:
        eligible=[i for i,cluster in enumerate(clusters) if view not in cluster['views']]
        match=None
        if eligible:
            overlaps=mu.iou([rle],[clusters[i]['anchor'] for i in eligible],[0]*len(eligible))[0]
            best=int(np.argmax(overlaps))
            if overlaps[best] >= .5:match=eligible[best]
        if match is None:
            clusters.append({'anchor':rle,'views':set(),'members':[]})
            match=len(clusters)-1
        clusters[match]['views'].add(view)
        clusters[match]['members'].append((score,view,rank,rle))
    result=[]
    for cluster in clusters:
        rle=majority([row[3] for row in cluster['members']],cluster['anchor'])
        if mu.area(rle) == 0:continue
        result.append({'score':sum(row[0] for row in cluster['members'])/4.,'rle':rle,
            'support_views':sorted(cluster['views']),
            'member_ranks':[[view,rank] for _,view,rank,_ in cluster['members']]})
    result.sort(key=lambda row:-row['score'])
    return result[:maximum]
