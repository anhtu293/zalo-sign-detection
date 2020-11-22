import numpy as np


def iou(bbox1, bbox2):
    x_min = max(bbox1[0], bbox2[0])
    y_min = max(bbox1[1], bbox2[1])
    x_max = min(bbox1[2], bbox2[2])
    y_max = min(bbox1[3], bbox2[3])
    width = x_max - x_min
    height = y_max - y_min
    if width < 0 or height < 0:
        intersection = 0
    else:
        intersection = width*height
    union = (bbox1[2]-bbox1[0])*(bbox1[3] - bbox1[1]) + (bbox2[2]-bbox2[0])*(bbox2[3] - bbox2[1]) - intersection
    return intersection/union


def nms_interclass(result):
    bboxes = []
    threshold = 0.7
    for cls, res in enumerate(result):
        if res.shape[0] == 0:
            continue
        for i in range(res.shape[0]):
            bbox = res[i, :].tolist()
            bbox.append(cls)
            bboxes.append(bbox)
    keep = [True for _ in range(len(bboxes))]
    for i, pred in enumerate(bboxes):
        if not keep[i]:
            continue
        ious = [iou(pred, box2) if keep[j] else 0 for j, box2 in enumerate(bboxes)]
        overlap = []
        for j in range(len(ious)):
            if ious[j] > threshold:
                overlap.append(j)
        if len(overlap) > 0:
            best = overlap[np.argmax(np.array([bboxes[k][5] for k in overlap]))]
            for j in overlap:
                if j != best:
                    keep[j] = False
    new_bboxes = [bboxes[i] for i in range(len(bboxes)) if keep[i]]
    new_res = [np.array([], dtype=np.float32).reshape(0,5) for _ in range(7)]
    for bbox in new_bboxes:
        new_res[bbox[5]] = np.concatenate((new_res[bbox[5]], np.array(bbox[:5]).reshape(1,5)))
    return new_res
