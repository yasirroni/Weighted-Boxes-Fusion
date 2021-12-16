from numba import jit
import numpy as np
import copy

@jit(nopython=True)
def bb_intersection_over_union(A, B) -> float:
    xA = max(A[0], B[0])
    yA = max(A[1], B[1])
    xB = min(A[2], B[2])
    yB = min(A[3], B[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    if interArea == 0:
        return 0.0

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (A[2] - A[0]) * (A[3] - A[1])
    boxBArea = (B[2] - B[0]) * (B[3] - B[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# TODO:
# support named_box
def union(boxes, iou_thr=None):
    """
    Implementation of greedy union

    boxes:
        list of box, box in (xmin, ymin, xmax, ymax)
    """
    if iou_thr is None:
        f = intersects()
    else:
        f = intersects_over_threshold(iou_thr)
    boxes = copy.deepcopy(boxes)
    i = 0
    while i < len(boxes):
        j = i+1
        while j < len(boxes):
            if f(boxes[i], boxes[j]) and i != j:
                boxes[i] = union_merge(boxes[i], boxes[j])
                boxes.pop(j)
                if i > j:
                    i -= 1
                j = -1 # -1 + 1 = 0
            j += 1
        i += 1
    return boxes

def union_merge(box1, box2):
    return (min(box1[0],box2[0]), min(box1[1],box2[1]), max(box1[2],box2[2]), max(box1[3],box2[3]))

class intersects():
    """
    box1 and box2:
        (xmin, ymin, xmax, ymax)
        
    Separating Axis Theorem:
        https://stackoverflow.com/a/40795835/11671779
    """
    def __call__(self, box1, box2):
        return not (box1[2] < box2[0] or box1[0] > box2[2] or box1[1] > box2[3] or box1[3] < box2[1])

class intersects_over_threshold():
    def __init__(self, iou_thr):
        self.iou_thr = iou_thr
    
    def __call__(self, box1, box2):
        iou = bb_intersection_over_union(box1, box2)
        return iou > self.iou_thr