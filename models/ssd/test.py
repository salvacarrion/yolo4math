import os
import sys
import time
import datetime
import argparse
import tqdm
import os
import sys

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, BASE_PATH)

import torch

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler

from models.ssd.utils import find_jaccard_overlap

from terminaltables import AsciiTable

from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from utils.evaluate import *


def evaluate_raw(model, images_path, labels_path, iou_thres, conf_thres, nms_thres, input_size, batch_size, top_k, class_names=None,  plot_detections=None):
    # Get dataloader
    dataset = ListDatasetSSD(images_path=images_path, labels_path=labels_path, input_size=opt.input_size, class_names=class_names)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    return evaluate(model, dataloader, iou_thres, conf_thres, nms_thres, input_size,  top_k, class_names, plot_detections)


def evaluate(model, dataloader, iou_thres, conf_thres, nms_thres, input_size, top_k, class_names=None, plot_detections=None):
    model.eval()
    running_loss = 0

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for batch_i, (img_paths, images, boxes, labels) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects"), 1):
    # for batch_i, (images_path, input_imgs, targets) in enumerate(dataloader, 1):

        images = Variable(images.type(Tensor), requires_grad=False)

        # Forward prop.
        predicted_locs, predicted_scores = model(images)

        # Detect objects in SSD output
        det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=conf_thres,
                                                                 max_overlap=iou_thres, top_k=top_k, cpu=True)

        # Clip predictions
        for i in range(len(det_boxes)):
            det_boxes[i] = torch.clamp(det_boxes[i], min=0.0, max=1.0)


        # Show detections
        if plot_detections and batch_i <= plot_detections:
            if det_boxes is not None:
                use_original = True
                save_path = BASE_PATH+'/outputs/{}'.format(img_paths[0].split('/')[-1])
                # Scale target bboxes
                input_img = img2img(images[0])
                ori_img = img2img(img_paths[0])

                # Output
                p_bboxes = det_boxes[0].cpu().data.numpy()
                p_labels = det_labels[0].cpu().data.numpy()
                t_bboxes = boxes[0].cpu().data.numpy()

                if use_original:
                    p_bboxes = rescale_boxes(p_bboxes, current_shape=input_img.shape[:2], original_shape=ori_img.shape[:2], relxyxy=True)
                    t_bboxes = rescale_boxes(t_bboxes, current_shape=input_img.shape[:2], original_shape=ori_img.shape[:2], relxyxy=True)
                    plot_bboxes(ori_img, p_bboxes,  class_ids=p_labels, class_names=class_names, show_results=False, coords_rel=False,
                                t_bboxes=t_bboxes, title="Detection + ground truth ({})".format(img_paths[0]), save_path=save_path)
                else:
                    plot_bboxes(input_img, p_bboxes, class_ids=p_labels, class_names=class_names, show_results=False, coords_rel=True,
                                t_bboxes=t_bboxes, title="Detection + ground truth ({})".format(img_paths[0]), save_path=save_path)
            else:
                print("NO DETECTIONS")


def make_predictions(dataloader, model, min_score=0.01, max_overlap=0.45, top_k=200, plot_detections=None):
    # Make sure it's in eval mode
    model.eval()

    # Lists to store detected and true boxes, labels, scores
    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()

    with torch.no_grad():

        # Get predictions
        for batch_i, (img_paths, images, boxes, labels) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects"), 1):
            images = images.to(device)
            batch_size = len(img_paths)

            # Forward prop.
            predicted_locs, predicted_scores = model(images)

            # Detect objects in SSD output
            det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(
                predicted_locs, predicted_scores, min_score, max_overlap, top_k, cpu=False)

            # Store this batch's results for mAP calculation
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]
            # Clip predictions
            for i in range(len(det_boxes)):
                det_boxes[i] = torch.clamp(det_boxes[i], min=0.0, max=1.0)

            # Plot predictions
            total_plots = (batch_i - 1) * batch_size
            if plot_detections and total_plots < plot_detections:
                plot_predictions(img_paths, images, det_boxes_batch, det_labels_batch, det_scores_batch, boxes,
                                 labels, class_names)

            det_boxes.extend(det_boxes_batch)
            det_labels.extend(det_labels_batch)
            det_scores.extend(det_scores_batch)
            true_boxes.extend(boxes)
            true_labels.extend(labels)

    return det_boxes, det_labels, det_scores, true_boxes, true_labels



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1, help="size of each image batch")
    parser.add_argument("--data_config", type=str, default=BASE_PATH+"/config/custom.data", help="path to data config file")
    parser.add_argument("--weights_path", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--input_size", default=(1024, 1024), help="size of each image dimension")
    parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")
    parser.add_argument("--shuffle_dataset", type=int, default=False, help="shuffle dataset")
    parser.add_argument("--validation_split", type=float, default=0.0, help="validation split [0..1]")
    parser.add_argument("--checkpoint_dir", type=str, default=BASE_PATH+"/checkpoints", help="path to checkpoint folder")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.3, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--top_k", type=int, default=200, help="Keep top K best hypothesis")
    parser.add_argument("--plot_detections", type=int, default=None, help="Number of detections to plot and save")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_config = parse_data_config(opt.data_config)
    dataset_path = "/home/salvacarrion/" + "Documents/"
    test_path = dataset_path + data_config["test"].format(opt.input_size[0])
    labels_path = dataset_path + data_config["labels"]
    class_names = load_classes(dataset_path + data_config["classes"])
    class_names.insert(0, 'background')

    # Load model
    model = torch.load(opt.weights_path).to(device)

    print("\nEvaluating model:\n")

    # Dataloader
    dataset = ListDatasetSSD(images_path=test_path, labels_path=labels_path, input_size=opt.input_size,
                             class_names=class_names)
    valid_sampler = SubsetRandomSampler(range(10))
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=False, num_workers=0, pin_memory=False,
        collate_fn=dataset.collate_fn
    )

    # Make predictions
    print("Making predictions...")
    det_boxes, det_labels, det_scores, true_boxes, true_labels = \
        make_predictions(dataloader, model, min_score=opt.conf_thres, max_overlap=opt.nms_thres, top_k=opt.top_k,
                         plot_detections=opt.plot_detections)
    data = {
        'det_boxes': det_boxes,
        'det_labels': det_labels,
        'det_scores': det_scores,
        'true_boxes': true_boxes,
        'true_labels': true_labels,
    }
    save_obj(data, "predictions.pkl")

    # For debbuging
    # data = load_obj('predictions.pkl')
    # det_boxes = data['det_boxes']
    # det_labels = data['det_labels']
    # det_scores = data['det_scores']
    # true_boxes = data['true_boxes']
    # true_labels = data['true_labels']

    # Confusion matrix
    print("Computing confusion matrix...")
    confusion_matrix = confusion_matrix(det_boxes, det_labels, det_scores, true_boxes, true_labels, len(class_names), ignore_bg=True)
    save_obj({'confusion_matrix': confusion_matrix}, "confusion_matrix.pkl")
    # confusion_matrix = load_obj('confusion_matrix.pkl')

    # Compute stats
    print("Computing stats...")
    stats = get_stats(confusion_matrix)
    save_dataset(stats, "stats.json")

    # Show stats
    for k, v in stats.items():
        print("{}: {}".format(k, v))

    print("Done!")

    sdasdasd = 33
    #
    # precision_list = []
    # recall_list = []
    # f1_list = []
    # mAP_list = []
    # loss_list = []
    #
    # iou_thres_grid  = [0.5] #[0.1, 0.3, 0.5, 0.7, 0.9]
    # conf_thres_grid = [0.5] #[0.1, 0.3, 0.5, 0.7, 0.9]
    # nms_thres_grid  = [0.3] #[0.1, 0.3, 0.5, 0.7, 0.9]
    #
    # print("Grids:")
    # print("\t- IOU thresholds: " + str(iou_thres_grid))  # What we consider as a positive result (checked against GT)
    # print("\t- Conf. thresholds: " + str(conf_thres_grid))  # Minimum object confidence
    # print("\t- NMS thresholds: " + str(nms_thres_grid))  # When we remove overlapping bboxes?
    # print("\nRuns:")
    #
    # for iou_thres in iou_thres_grid:
    #     for conf_thres in conf_thres_grid:
    #         for nms_thres in nms_thres_grid:
    #             precision, recall, AP, f1, ap_class, loss = evaluate_raw(
    #                 model,
    #                 images_path=test_path,
    #                 labels_path=labels_path,
    #                 iou_thres=iou_thres,
    #                 conf_thres=conf_thres,
    #                 nms_thres=nms_thres,
    #                 input_size=opt.input_size,
    #                 batch_size=opt.batch_size,
    #                 top_k=opt.top_k,
    #                 class_names=class_names,
    #                 plot_detections=opt.plot_detections
    #             )
    #
    #             print("Results train+test: [iou_thres={}; conf_thres={}; nms_thres={}]".format(iou_thres, conf_thres, nms_thres))
    #             # Print class APs and mAP
    #             ap_table = [["Index", "Class name", "AP"]]
    #             for i, c in enumerate(ap_class):
    #                 ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
    #             print(AsciiTable(ap_table).table)
    #             print("test_precision: {:.5f}".format(precision.mean()))
    #             print("test_recall: {:.5f}".format(recall.mean()))
    #             print("test_f1: {:.5f}".format(f1.mean()))
    #             print("test_mAP: {:.5f}".format(AP.mean()))
    #             print("test_loss: {:.5f}".format(loss))
    #             print("\n")
    #
    #             # Append values
    #             precision_list.append(precision.mean())
    #             recall_list.append(recall.mean())
    #             f1_list.append(f1.mean())
    #             mAP_list.append(AP.mean())
    #             loss_list.append(loss)
    #
    # print("Summary:")
    # print("- Precision: {}".format(precision_list))
    # print("- Recall: {}".format(recall_list))
    # print("- F1: {}".format(f1_list))
    # print("- mAP: {}".format(mAP_list))
    # print("- Loss: {}".format(loss_list))
