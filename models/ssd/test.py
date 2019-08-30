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

from terminaltables import AsciiTable

from models.ssd.model import SSD300
from models.ssd.utils import calculate_mAP

from utils.utils import *
from utils.datasets import *
from utils.parse_config import *


def evaluate_raw(model, images_path, labels_path, iou_thres, conf_thres, nms_thres, input_size, top_k, batch_size, class_names=None,  plot_detections=None):
    # Get dataloader
    dataset = ListDataset(images_path=images_path, labels_path=labels_path, input_size=input_size,
                          class_names=class_names, single_channel=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    return evaluate(model, dataloader, iou_thres, conf_thres, nms_thres, input_size, top_k, class_names, plot_detections)


def evaluate(model, dataloader, iou_thres, conf_thres, nms_thres, input_size, top_k, class_names=None, plot_detections=None):
    model.eval()

    # Lists to store detected and true boxes, labels, scores
    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()
    true_difficulties = list()

    with torch.no_grad():
        for batch_i, (img_paths, images, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects"), 1):

            # Forward prop.
            images = images.to(device)
            predicted_locs, predicted_scores = model(images)
            #
            # indxs_priors = torch.randint(0, len(predicted_scores[0]), (15000,))
            # predicted_locs = predicted_locs[0][indxs_priors].unsqueeze(0)
            # predicted_scores = predicted_scores[0][indxs_priors].unsqueeze(0)

             # Detect objects in SSD output
            det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(
                predicted_locs, predicted_scores, min_score=0.2, max_overlap=0.5, top_k=50, idxs=None)

            # Get boxes
            class_ids = det_labels_batch[0].cpu()-1
            p_bboxes = det_boxes_batch[0].cpu()
            t_bboxes = xywh2xyxy(targets[targets[:, 0] == 0][:, 2:].cpu())


            # Compute stats
            det_boxes.extend(det_boxes_batch)
            det_labels.extend(det_labels_batch)
            det_scores.extend(det_scores_batch)
            true_boxes.extend([t_bboxes])
            true_labels.extend([class_ids])
            true_difficulties.extend([torch.zeros(1) for i in range(len(t_bboxes))])

            # Show detections
            if plot_detections and batch_i <= plot_detections:
                if det_boxes:
                    use_original = False
                    save_path = BASE_PATH + '/outputs/{}'.format(img_paths[0].split('/')[-1])
                    save_path=None

                    # Scale target bboxes
                    input_img = img2img(images[0])
                    ori_img = img2img(img_paths[0])

                    p_bboxes = rel2abs(p_bboxes, input_size, input_size)
                    t_bboxes = rel2abs(t_bboxes, input_size, input_size)

                    if use_original:
                        p_bboxes = rescale_boxes(p_bboxes, current_shape=input_img.shape[:2],
                                                 original_shape=ori_img.shape[:2])
                        t_bboxes = rescale_boxes(t_bboxes, current_shape=input_img.shape[:2],
                                                 original_shape=ori_img.shape[:2])
                        plot_bboxes(ori_img, p_bboxes, class_ids=class_ids, class_names=class_names, show_results=True,
                                    t_bboxes=t_bboxes, title="Detection + ground truth ({})".format(img_paths[0]),
                                    save_path=save_path)
                    else:
                        plot_bboxes(input_img, p_bboxes, class_ids=class_ids, class_names=class_names, show_results=True,
                                    t_bboxes=t_bboxes, title="Detection + ground truth ({})".format(img_paths[0]),
                                    save_path=save_path)
                else:
                    print("NO DETECTIONS")


    # Calculate mAP
    APs, mAP = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties)

    return precision, recall, AP, f1, ap_class, 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1, help="size of each image batch")
    parser.add_argument("--data_config", type=str, default=BASE_PATH+"/config/custom.data", help="path to data config file")
    parser.add_argument("--weights_path", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--input_size", type=int, default=300, help="size of each image dimension")
    parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")
    parser.add_argument("--shuffle_dataset", type=int, default=False, help="shuffle dataset")
    parser.add_argument("--validation_split", type=float, default=0.0, help="validation split [0..1]")
    parser.add_argument("--checkpoint_dir", type=str, default=BASE_PATH+"/checkpoints", help="path to checkpoint folder")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--top_k", type=int, default=200, help="Keep top K best hypothesis")
    parser.add_argument("--plot_detections", type=int, default=50, help="Number of detections to plot and save")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_config = parse_data_config(opt.data_config)
    test_path = data_config["test"].format(1024)
    labels_path = data_config["labels"]
    class_names = load_classes(data_config["classes"])

    # Initiate model
    model = torch.load(opt.weights_path).to(device)

    print("\nEvaluating model:\n")
    precision_list = []
    recall_list = []
    f1_list = []
    mAP_list = []
    loss_list = []

    iou_thres_grid = [0.5]
    conf_thres_grid = [0.5]
    nms_thres_grid = [0.3]

    print("Grids:")
    print("\t- IOU thresholds: " + str(iou_thres_grid))  # What we consider as a positive result (checked against GT)
    print("\t- Conf. thresholds: " + str(conf_thres_grid))  # Minimum object confidence
    print("\t- NMS thresholds: " + str(nms_thres_grid))  # When we remove overlapping bboxes?
    print("\nRuns:")

    for iou_thres in iou_thres_grid:
        for conf_thres in conf_thres_grid:
            for nms_thres in nms_thres_grid:
                precision, recall, AP, f1, ap_class, loss = evaluate_raw(
                    model,
                    images_path=test_path,
                    labels_path=labels_path,
                    iou_thres=opt.iou_thres,
                    conf_thres=opt.conf_thres,
                    nms_thres=nms_thres,
                    input_size=opt.input_size,
                    top_k=opt.top_k,
                    batch_size=opt.batch_size,
                    class_names=class_names,
                    plot_detections=opt.plot_detections
                )

                print("Results train+test: [iou_thres={}; conf_thres={}; nms_thres={}]".format(iou_thres, conf_thres,
                                                                                               nms_thres))
                # Print class APs and mAP
                ap_table = [["Index", "Class name", "AP"]]
                for i, c in enumerate(ap_class):
                    ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
                print(AsciiTable(ap_table).table)
                print("test_precision: {:.5f}".format(precision.mean()))
                print("test_recall: {:.5f}".format(recall.mean()))
                print("test_f1: {:.5f}".format(f1.mean()))
                print("test_mAP: {:.5f}".format(AP.mean()))
                print("test_loss: {:.5f}".format(loss))
                print("\n")

                # Append values
                precision_list.append(precision.mean())
                recall_list.append(recall.mean())
                f1_list.append(f1.mean())
                mAP_list.append(AP.mean())
                loss_list.append(loss)

    print("Summary:")
    print("- Precision: {}".format(precision_list))
    print("- Recall: {}".format(recall_list))
    print("- F1: {}".format(f1_list))
    print("- mAP: {}".format(mAP_list))
    print("- Loss: {}".format(loss_list))