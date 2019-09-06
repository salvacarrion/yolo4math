import os
import sys
import torch
from models.ssd.utils import find_jaccard_overlap
from utils.utils import img2img, rescale_boxes, plot_bboxes

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, BASE_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def confusion_matrix(det_boxes, det_labels, det_scores, true_boxes, true_labels, n_classes, ignore_bg=False):
    """
    Calculate the Mean Average Precision (mAP) of detected objects.

    See https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173 for an explanation

    :param det_boxes: list of tensors, one tensor for each image containing detected objects' bounding boxes
    :param det_labels: list of tensors, one tensor for each image containing detected objects' labels
    :param det_scores: list of tensors, one tensor for each image containing detected objects' labels' scores
    :param true_boxes: list of tensors, one tensor for each image containing actual objects' bounding boxes
    :param true_labels: list of tensors, one tensor for each image containing actual objects' labels
    :return: list of average precisions for all classes, mean average precision (mAP)
    """
    assert len(det_boxes) == len(det_labels) == len(det_scores) == len(true_boxes) == len(true_labels)

    # Store all (true) objects in a single continuous tensor while keeping track of the image it is from
    true_images = list()
    for i in range(len(true_labels)):
        true_images.extend([i] * true_labels[i].size(0))
    true_images = torch.LongTensor(true_images).to(device)  # (n_objects), n_objects is the total no. of objects across all images
    true_boxes = torch.cat(true_boxes, dim=0)  # (n_objects, 4)
    true_labels = torch.cat(true_labels, dim=0)  # (n_objects)

    assert true_images.size(0) == true_boxes.size(0) == true_labels.size(0)

    # Store all detections in a single continuous tensor while keeping track of the image it is from
    det_images = list()
    for i in range(len(det_labels)):
        det_images.extend([i] * det_labels[i].size(0))
    det_images = torch.LongTensor(det_images).to(device)  # (n_detections)
    det_boxes = torch.cat(det_boxes, dim=0)  # (n_detections, 4)
    det_labels = torch.cat(det_labels, dim=0)  # (n_detections)
    det_scores = torch.cat(det_scores, dim=0)  # (n_detections)
    det_true_labels = []

    assert det_images.size(0) == det_boxes.size(0) == det_labels.size(0) == det_scores.size(0)

    stats = {}
    ini = 0 if not ignore_bg else 1
    for c in range(ini, n_classes):
        # Extract only objects with this class
        true_class_images = true_images[true_labels == c]  # (n_class_objects)
        true_class_boxes = true_boxes[true_labels == c]  # (n_class_objects, 4)
        n_class_objects = true_class_boxes.size(0)

        # Keep track of which true objects with this class have already been 'detected'
        # So far, none
        true_class_boxes_detected = torch.zeros((true_class_boxes.size(0)), dtype=torch.uint8).to(device)  # (n_class_objects)

        # Extract only detections with this class
        det_class_images = det_images[det_labels == c]  # (n_class_detections)
        det_class_boxes = det_boxes[det_labels == c]  # (n_class_detections, 4)
        det_class_scores = det_scores[det_labels == c]  # (n_class_detections)
        n_class_detections = det_class_boxes.size(0)

        # Initialize arrays
        true_positives = torch.zeros(n_class_detections, dtype=torch.float).to(device)  # (n_class_detections)
        false_positives = torch.zeros(n_class_detections, dtype=torch.float).to(device)  # (n_class_detections)

        if n_class_detections == 0:
            stats[c] = {'true_positives': true_positives,
                        'false_positives': false_positives,
                        'n_detections': int(n_class_detections),
                        'n_ground_truths': int(n_class_objects),
                        }
            continue

        # Sort detections in decreasing order of confidence/scores
        det_class_scores, sort_ind = torch.sort(det_class_scores, dim=0, descending=True)  # (n_class_detections)
        det_class_images = det_class_images[sort_ind]  # (n_class_detections)
        det_class_boxes = det_class_boxes[sort_ind]  # (n_class_detections, 4)

        # In the order of decreasing scores, check if true or false positive
        for d in range(n_class_detections):
            this_detection_box = det_class_boxes[d].unsqueeze(0)  # (1, 4)
            this_image = det_class_images[d]  # (), scalar

            # Find objects in the same image with this class and whether they have been detected before
            object_boxes = true_class_boxes[true_class_images == this_image]  # (n_class_objects_in_img)
            # If no such object in this image, then the detection is a false positive
            if object_boxes.size(0) == 0:
                false_positives[d] = 1
                continue

            # Find maximum overlap of this detection with objects in this image of this class
            overlaps = find_jaccard_overlap(this_detection_box, object_boxes)  # (1, n_class_objects_in_img)
            max_overlap, ind = torch.max(overlaps.squeeze(0), dim=0)  # (), () - scalars

            # 'ind' is the index of the object in these image-level tensors 'object_boxes', 'object_difficulties'
            # In the original class-level tensors 'true_class_boxes', etc., 'ind' corresponds to object with index...
            original_ind = torch.LongTensor(range(true_class_boxes.size(0)))[true_class_images == this_image][ind]
            # We need 'original_ind' to update 'true_class_boxes_detected'

            # If the maximum overlap is greater than the threshold of 0.5, it's a match
            if max_overlap.item() > 0.5:
                # If this object has already not been detected, it's a true positive
                if true_class_boxes_detected[original_ind] == 0:
                    true_positives[d] = 1
                    true_class_boxes_detected[original_ind] = 1  # this object has now been detected/accounted for
                # Otherwise, it's a false positive (since this object is already accounted for)
                else:
                    false_positives[d] = 1
            # Otherwise, the detection occurs in a different location than the actual object, and is a false positive
            else:
                false_positives[d] = 1

        stats[c] = {'true_positives': true_positives,
                    'false_positives': false_positives,
                    'n_detections': int(n_class_detections),
                    'n_ground_truths': int(n_class_objects),
        }
    return stats


def get_stats(confusion_matrix):
    n_classes = len(confusion_matrix.keys())
    average_precisions = torch.zeros(n_classes, dtype=torch.float)  # (n_classes - 1)
    total_tp = 0
    total_fp = 0
    total_gt = 0
    total_det = 0

    metrics = {'classes': {}}
    for c, stats in confusion_matrix.items():
        true_positives = stats['true_positives']
        false_positives = stats['false_positives']

        # Compute cumulative precision and recall at each detection in the order of decreasing scores
        cumul_true_positives = torch.cumsum(true_positives, dim=0)  # (n_class_detections)
        cumul_false_positives = torch.cumsum(false_positives, dim=0)  # (n_class_detections)
        cumul_precision = cumul_true_positives / (
                cumul_true_positives + cumul_false_positives + 1e-10)  # (n_class_detections)
        cumul_recall = cumul_true_positives / len(true_positives)  # (n_class_detections)

        # Find the mean of the maximum of the precisions corresponding to recalls above the threshold 't'
        recall_thresholds = torch.arange(start=0, end=1.1, step=.1).tolist()  # (11)
        precisions = torch.zeros((len(recall_thresholds)), dtype=torch.float).to(device)  # (11)
        for i, t in enumerate(recall_thresholds):
            recalls_above_t = cumul_recall >= t
            if recalls_above_t.any():
                precisions[i] = cumul_precision[recalls_above_t].max()
            else:
                precisions[i] = 0.

        # Compute more stats
        s_tp = int(true_positives.sum())
        s_fp = int(false_positives.sum())
        total_tp += s_tp
        total_fp += s_fp
        total_gt += int(stats['n_ground_truths'])
        total_det += int(stats['n_detections'])

        recall = float(s_tp / stats['n_ground_truths']) if stats['n_ground_truths'] else 0 # TP/(TP+FN)
        precision = float(s_fp / stats['n_detections']) if stats['n_detections'] else 0 # TP/(TP+FP)
        f1 = 2.0 * (precision * recall) / (precision + recall) if (precision + recall) else 0

        avg_precision = float(precisions.mean())
        average_precisions[c - 1] = avg_precision  # c is in [1, n_classes - 1]

        metrics['classes'][c] = {
            'recall11': recall_thresholds,
            'precision11': precisions.cpu().data.numpy().tolist(),
            'AP': float(avg_precision),
            'recall': float(recall),
            'precision': float(precision),
            'f1': float(f1),
        }

    # Calculate Mean Average Precision (mAP)
    metrics['mAP'] = float(average_precisions.mean().item())
    metrics['recall'] = float(total_tp / total_gt)
    metrics['precision'] = float(total_tp / total_det)
    metrics['f1'] = float(2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall']))

    return metrics


def ignore_bg(labels, remove_bg=False):
    new_labels = []
    for i in range(len(labels)):
        labels[i] -= 1
        if not remove_bg or labels[i][0] >= 0:
            new_labels.append(labels[i])
    return new_labels


def match_classes(det_boxes, det_labels, det_scores, true_boxes, true_labels, n_classes):
    """
    list of detections. Background encoded as label zero
    """
    pred_score = []
    pred_iou = []
    true_class = []
    for i, (boxes, labels, scores, t_boxes, true_labels) in \
            enumerate(zip(det_boxes, det_labels, det_scores, true_boxes, true_labels), 1):
        print("{}/{}".format(i, len(det_boxes)))

        assert boxes.size(0) == labels.size(0) == scores.size(0)
        assert t_boxes.size(0) == true_labels.size(0)

        overlaps = find_jaccard_overlap(boxes, t_boxes)  # (1, n_class_objects_in_img)
        max_overlap, ind = torch.max(overlaps, dim=1)  # (), () - scalars
        t_labels = true_labels[ind]

        # Keep IOUs
        pred_iou.append(max_overlap)
        true_class.append(t_labels)

    # Concatenate results
    pred_class = torch.cat(det_labels)
    pred_score = torch.cat(det_scores)
    pred_iou = torch.cat(pred_iou)
    true_class = torch.cat(true_class)

    assert pred_class.size(0) == pred_score.size(0) == pred_iou.size(0) == true_class.size(0)

    return pred_class, pred_score, pred_iou, true_class


def compute_hoeim(pred_class, pred_iou, true_class):
    correct_class = pred_class == true_class
    incorrect_class = pred_class != true_class

    # Hoiem
    correct = correct_class * (pred_iou > 0.5)
    localization = correct_class * (pred_iou > 0.1) * (pred_iou <= 0.5)
    other = incorrect_class * (pred_iou > 0.1)
    background = pred_iou <= 0.1

    analysis = {
        'correct': correct.sum().item(),
        'localization': localization.sum().item(),
        'other': other.sum().item(),
        'background': background.sum().item(),
    }

    assert (analysis['correct'] + analysis['localization'] + analysis['other'] +
            analysis['background']) == pred_class.size(0)
    return analysis


def plot_predictions(image_paths, images, det_boxes, det_labels, det_scores, true_boxes, true_labels, class_names):
    assert len(det_boxes) == len(det_labels) == len(det_scores) == len(true_boxes) == len(true_labels)

    for i in range(len(image_paths)):
        use_original = False
        output_path = BASE_PATH + "/outputs"
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        save_path = "{}/{}".format(output_path, image_paths[i].split('/')[-1])

        # Output
        p_bboxes = det_boxes[i].cpu().data.numpy()
        p_labels = det_labels[i].cpu().data.numpy()
        t_bboxes = true_boxes[i].cpu().data.numpy()
        t_labels = true_labels[i].cpu().data.numpy()

        if use_original:
            input_img = img2img(image_paths[i])

            # Rescale boxes
            p_bboxes = rescale_boxes(p_bboxes, current_shape=input_img.shape[:2], original_shape=input_img.shape[:2],
                                     relxyxy=True)
            t_bboxes = rescale_boxes(t_bboxes, current_shape=input_img.shape[:2], original_shape=input_img.shape[:2],
                                     relxyxy=True)

            plot_bboxes(use_original, p_bboxes, class_ids=p_labels, class_names=class_names, show_results=False,
                        coords_rel=False,
                        t_bboxes=t_bboxes, title="Detection + ground truth ({})".format(image_paths[i]),
                        save_path=save_path)
        else:
            input_img = img2img(images[i])
            plot_bboxes(input_img, p_bboxes, class_ids=p_labels, class_names=class_names, show_results=False,
                        coords_rel=True,
                        t_bboxes=t_bboxes, title="Detection + ground truth ({})".format(image_paths[i]),
                        save_path=save_path)
