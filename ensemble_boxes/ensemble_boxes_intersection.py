from numba import jit
import numpy as np
import copy

import warnings

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

def prefilter_boxes(boxes, scores, labels, weights, thr):
    # Create dict with boxes stored by its label
    new_boxes = dict()

    for t in range(len(boxes)):

        if len(boxes[t]) != len(scores[t]):
            print('Error. Length of boxes arrays not equal to length of scores array: {} != {}'.format(len(boxes[t]), len(scores[t])))
            exit()

        if len(boxes[t]) != len(labels[t]):
            print('Error. Length of boxes arrays not equal to length of labels array: {} != {}'.format(len(boxes[t]), len(labels[t])))
            exit()

        for j in range(len(boxes[t])):
            score = scores[t][j]
            if score < thr:
                continue
            label = int(labels[t][j])
            box_part = boxes[t][j]
            x1 = float(box_part[0])
            y1 = float(box_part[1])
            x2 = float(box_part[2])
            y2 = float(box_part[3])

            # Box data checks
            if x2 < x1:
                warnings.warn('X2 < X1 value in box. Swap them.')
                x1, x2 = x2, x1
            if y2 < y1:
                warnings.warn('Y2 < Y1 value in box. Swap them.')
                y1, y2 = y2, y1
            if x1 < 0:
                warnings.warn('X1 < 0 in box. Set it to 0.')
                x1 = 0
            if x1 > 1:
                warnings.warn('X1 > 1 in box. Set it to 1. Check that you normalize boxes in [0, 1] range.')
                x1 = 1
            if x2 < 0:
                warnings.warn('X2 < 0 in box. Set it to 0.')
                x2 = 0
            if x2 > 1:
                warnings.warn('X2 > 1 in box. Set it to 1. Check that you normalize boxes in [0, 1] range.')
                x2 = 1
            if y1 < 0:
                warnings.warn('Y1 < 0 in box. Set it to 0.')
                y1 = 0
            if y1 > 1:
                warnings.warn('Y1 > 1 in box. Set it to 1. Check that you normalize boxes in [0, 1] range.')
                y1 = 1
            if y2 < 0:
                warnings.warn('Y2 < 0 in box. Set it to 0.')
                y2 = 0
            if y2 > 1:
                warnings.warn('Y2 > 1 in box. Set it to 1. Check that you normalize boxes in [0, 1] range.')
                y2 = 1
            if (x2 - x1) * (y2 - y1) == 0.0:
                warnings.warn("Zero area box skipped: {}.".format(box_part))
                continue

            # [label, score, weight, model index, x1, y1, x2, y2]
            b = [int(label), float(score) * weights[t], weights[t], t, x1, y1, x2, y2]
            if label not in new_boxes:
                new_boxes[label] = []
            new_boxes[label].append(b)

    # Sort each list in dict by score and transform it to numpy array
    for k in new_boxes:
        current_boxes = np.array(new_boxes[k])
        new_boxes[k] = current_boxes[current_boxes[:, 1].argsort()[::-1]]

    return new_boxes

# TODO:
# support named_box
def intersection(boxes_list, scores_list, labels_list, weights=None, iou_thr=0.55, skip_box_thr=0.0):
    if weights is None:
        weights = [1] * len(boxes_list)
    filtered_boxes = prefilter_boxes(boxes_list, scores_list, labels_list, weights, skip_box_thr)
    final_box = []
    final_scores = []
    final_labels = []
    for label in filtered_boxes:
        boxes = filtered_boxes[label][:,4:].tolist()
        scores = filtered_boxes[label][:,1].tolist()
        boxes, scores = greedy_intersection(boxes, scores=scores, iou_thr=iou_thr, inplace=False)
        final_box.extend(boxes)
        final_scores.extend(scores)
        final_labels.extend([label]*len(final_box))
    return final_box, final_scores, np.array(final_labels)

def greedy_intersection(boxes, scores=None, iou_thr=None, inplace=False,):
    """
    Implementation of greedy union

    boxes:
        list of box, box in (xmin, ymin, xmax, ymax)
    """
    if iou_thr is None:
        f = intersects()
    else:
        f = intersects_over_threshold(iou_thr)
    if not inplace:
        boxes = copy.deepcopy(boxes)
    i = 0
    while i < len(boxes):
        j = i+1
        while j < len(boxes):
            if f(boxes[i], boxes[j]) and i != j:
                if scores is not None:
                    scores[i] = max(scores[i], scores[j])
                    scores.pop(j)
                boxes[i] = intersection_merge(boxes[i], boxes[j])
                boxes.pop(j)
                if i > j:
                    i -= 1
                j = -1 # -1 + 1 = 0
            j += 1
        i += 1
    return boxes, scores

def intersection_merge(box1, box2):
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