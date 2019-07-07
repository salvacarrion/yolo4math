import random
import json
from collections import defaultdict
import cv2
import os
import numpy as np
from nms import nms
from PIL import Image, ImageDraw, ImageFont

import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from albumentations.augmentations.bbox_utils import convert_bboxes_to_albumentations, convert_bboxes_from_albumentations


def load_dataset(filename):
    with open(filename, 'r') as f:
        return json.load(f)


def save_dataset(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)


def get_classes(filename):
    with open(filename) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(filename):
    with open(filename) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def draw_bbox(image, bbox, color=None, thickness=2):
    # Get dimentions
    x, y, w, h = (int(x) for x in bbox)

    draw = ImageDraw.Draw(image)
    draw.rectangle([(x, y), (x+w, y+h)], outline=color, width=thickness)


def draw_labels(image, bbox, text, color=None, font_size=24):
    # Get dimentions
    x, y, w, h = (int(x) for x in bbox)

    # Set font
    font = ImageFont.truetype(font='font/FiraMono-Medium.otf', size=int(font_size))
    font_w, font_h = font.getsize(text)
    h_factor = 1.0

    # Draw text
    draw = ImageDraw.Draw(image)
    draw.rectangle([(x, round(y-font_h*h_factor)), (x+font_w, round(y-font_h*h_factor+font_h))], fill=color)
    draw.text((x, round(y-font_h*h_factor)), text=text, fill=(255, 255, 255, 255), font=font)


def save_image(image, filename, use_cv=False):
    if isinstance(image, np.ndarray):
        image = image.astype(np.uint8)

    if not use_cv:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        image = image.convert('RGB')
        image.save(filename, "JPEG", quality=100, optimize=True, progressive=True)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filename, image)


def crop_images(filename, bboxes, base_path='', image_id='', categories=None):
    # TODO: DELETE
    print("Croping image #{} => {}...".format(image_id, filename))
    image = cv2.imread(filename)

    # Crop bounding boxes
    total_cats = {k: 0 for k in categories}
    for i, bbox in enumerate(bboxes, 1):
        print("\t-Cropping region #{}...".format(i))

        x, y, w, h = bbox['bbox']
        label = bbox['category_id']
        total_cats[label] += 1

        # Crop image
        crop = image[y:y + h, x:x + w]

        # Save image
        filename2 = base_path + '/' + str(label) + '/' + 'image_' + str(image_id) + '__' + str(total_cats[label]) + '.jpg'
        cv2.imwrite(filename2, crop)


def letterbox_image(img, inp_dim, padding=True):
    # TODO: DELETE
    """
    resize image with unchanged aspect ratio using padding
    """
    ori_w, ori_h = img.shape[1], img.shape[0]
    max_w, max_h = inp_dim

    # Resize image
    min_ar = min(max_w / ori_w, max_h / ori_h)
    new_w = min(int(ori_w * min_ar) + 1, max_w)
    new_h = min(int(ori_h * min_ar) + 1, max_h)
    resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    if padding:
        # Fill image
        canvas = np.full((inp_dim[1], inp_dim[0], 3), 128, dtype=np.uint8)

        # Paste image
        y1 = (max_h - new_h) // 2
        x1 = (max_w - new_w) // 2
        canvas[y1:y1 + new_h, x1:x1 + new_w, :] = resized_image
    else:
        x1, y1 = 0, 0
        canvas = resized_image

    return canvas, (x1, y1, new_w, new_h, ori_w, ori_h)


def resize_bboxes(bboxes, coords):
    # TODO: DELETE
    new_bboxes = list(bboxes)  # Copy
    x1, y1, new_w, new_h, ori_w, ori_h = coords

    # Compute aspect ratio
    ratio_w = new_w/ori_w
    ratio_h = new_h/ori_h

    for i in range(len(new_bboxes)):
        x_b, y_b, w_b, h_b = bboxes[i]['bbox']

        # Interpolate
        x_b = x1 + x_b*ratio_w
        y_b = y1 + y_b*ratio_h
        w_b *= ratio_w
        h_b *= ratio_h

        # Update bonding box
        new_bboxes[i]['bbox'] = [round(x_b), round(y_b), round(w_b), round(h_b)]

    return new_bboxes


def coco2cxcywh(x):
    # From ABS(x_min, y_min, width, height) => ABS(center_x, center_y, w, h)
    y = x.new(x.shape)  # copy array shape
    y[..., 0] = (x[..., 0] + x[..., 2] / 2)  # center_x = x1 + width/2
    y[..., 1] = (x[..., 1] + x[..., 3] / 2)  # center_y = y1 + height/2
    y[..., 2] = x[..., 2]  # Abs width
    y[..., 3] = x[..., 3]  # Abs height
    return y


def abs2rel(x, height, width):
    # From ABS(x, y, w, h) => REL(x, y, w, h) [0.0-1.0]
    y = x.new(x.shape)  # copy array shape
    y[..., 0] = x[..., 0]/width
    y[..., 1] = x[..., 1]/height
    y[..., 2] = x[..., 2]/width
    y[..., 3] = x[..., 3]/height
    return y


#############################################
#############################################
#############################################
#############################################
#############################################
#############################################


def to_cpu(tensor):
    return tensor.detach().cpu()


def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def rescale_boxes(boxes, current_shape, original_shape):
    """ Rescales bounding boxes to the original shape """
    orig_h, orig_w = original_shape
    this_h, this_w = current_shape

    # The amount of padding that was added
    pad_x = max(orig_h - orig_w, 0) * (this_w / max(original_shape))
    pad_y = max(orig_w - orig_h, 0) * (this_h / max(original_shape))
    # Image height and width after padding is removed
    unpad_h = this_h - pad_y
    unpad_w = this_w - pad_x
    # Rescale bounding boxes to dimension of original image
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h
    return boxes



def xywh2xyxy(x):
    # From (center_x_abs, center_y_abs, w_abs, w_abs) => (x1_abs, y1_abs, x2_abs, y2_abs)
    y = x.new(x.shape)  # copy array shape
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # x1 = center_x - width/2
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # y1 = center_y - height/2
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # x2 = center_x + width/2
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # y2 = center_y + height/2
    return y


def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Object confidence times class confidence
        score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]
        # Sort by it
        image_pred = image_pred[(-score).argsort()]
        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)
        detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)
        # Perform non-maximum suppression
        keep_boxes = []
        while detections.size(0):
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
            label_match = detections[0, -1] == detections[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            weights = detections[invalid, 4:5]
            # Merge overlapping bboxes by order of confidence
            detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
            keep_boxes += [detections[0]]
            detections = detections[~invalid]
        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)

    return output



def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in tqdm.tqdm(unique_classes, desc="Computing AP"):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def get_batch_statistics(outputs, targets, iou_threshold):
    """ Compute true positives, predicted scores and predicted labels per sample """
    batch_metrics = []
    for sample_i in range(len(outputs)):

        if outputs[sample_i] is None:
            continue

        output = outputs[sample_i]
        pred_boxes = output[:, :4]
        pred_scores = output[:, 4]
        pred_labels = output[:, -1]

        true_positives = np.zeros(pred_boxes.shape[0])

        annotations = targets[targets[:, 0] == sample_i][:, 1:]
        target_labels = annotations[:, 0] if len(annotations) else []
        if len(annotations):
            detected_boxes = []
            target_boxes = annotations[:, 1:]

            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):

                # If targets are found break
                if len(detected_boxes) == len(annotations):
                    break

                # Ignore if label is not one of the target labels
                if pred_label not in target_labels:
                    continue

                iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0)
                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]
        batch_metrics.append([true_positives, pred_scores, pred_labels])
    return batch_metrics


def bbox_wh_iou(wh1, wh2):
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def remove_low_conf(prediction, conf_thres=0.5):
    # Filter out confidence scores below threshold
    output = []
    for image_i, image_pred in enumerate(prediction):
        image_preds = image_pred[image_pred[:, 4] >= conf_thres]
        output.append(image_preds)
    return output


def set_voc_format(prediction):
    # prediction = (image_i, hypothesis_i, (x,y,w,h,obj_conf, classes)
    # => take all all dimentions 'till last one, and then, take from the 0th to 4th (x,y,w,h)
    xywh = prediction[..., :4]
    prediction[..., :4] = xywh2xyxy(xywh)


def remove_useless_data(detections):
    for i, image_pred in enumerate(detections):
        # Object confidence times class confidence
        objects_conf = image_pred[:, 4]
        # Get probabilities and max class
        class_probs, class_idxs, = image_pred[:, 5:].max(1)

        # Compute score
        score = objects_conf * class_probs  # Real prob

        # Sort prediction by score (descending)
        image_pred = image_pred[(-score).argsort()]

        # Get max probability and index
        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)

        # Join: (x1,y1,x2,y2, obj_conf) + class_conf + class_idx
        detections[i] = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)

def build_targets(pred_boxes, pred_cls, target, anchors, ignore_thres):

    ByteTensor = torch.cuda.ByteTensor if pred_boxes.is_cuda else torch.ByteTensor
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor

    nB = pred_boxes.size(0)  # images
    nA = pred_boxes.size(1)  # anchors
    nC = pred_cls.size(-1)  # classes
    nG = pred_boxes.size(2)  # grid_size

    # Output tensors
    obj_mask = ByteTensor(nB, nA, nG, nG).fill_(0)
    noobj_mask = ByteTensor(nB, nA, nG, nG).fill_(1)
    class_mask = FloatTensor(nB, nA, nG, nG).fill_(0)
    iou_scores = FloatTensor(nB, nA, nG, nG).fill_(0)
    tx = FloatTensor(nB, nA, nG, nG).fill_(0)
    ty = FloatTensor(nB, nA, nG, nG).fill_(0)
    tw = FloatTensor(nB, nA, nG, nG).fill_(0)
    th = FloatTensor(nB, nA, nG, nG).fill_(0)
    tcls = FloatTensor(nB, nA, nG, nG, nC).fill_(0)

    # Convert to position relative to box
    target_boxes = target[:, 2:6] * nG
    gxy = target_boxes[:, :2]
    gwh = target_boxes[:, 2:]
    # Get anchors with best iou
    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])
    best_ious, best_n = ious.max(0)
    # Separate target values
    b, target_labels = target[:, :2].long().t()
    gx, gy = gxy.t()
    gw, gh = gwh.t()
    gi, gj = gxy.long().t()  # Integer. Which cell is responsible for X object?
    # Set masks
    obj_mask[b, best_n, gj, gi] = 1
    noobj_mask[b, best_n, gj, gi] = 0

    # Set noobj mask to zero where iou exceeds ignore threshold
    for i, anchor_ious in enumerate(ious.t()):
        noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0

    # Coordinates
    tx[b, best_n, gj, gi] = gx - gx.floor()
    ty[b, best_n, gj, gi] = gy - gy.floor()
    # Width and height
    tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
    th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)
    # One-hot encoding of label
    tcls[b, best_n, gj, gi, target_labels] = 1
    # Compute label correctness and iou at best anchor
    class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
    iou_scores[b, best_n, gj, gi] = bbox_iou(pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False)

    tconf = obj_mask.float()
    return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf


import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim


def evaluate(model, dataloader, iou_thres, conf_thres, nms_thres, img_size, batch_size):
    model.eval()

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):

        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)

        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class