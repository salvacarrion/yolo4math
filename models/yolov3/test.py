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

from models.yolov3.darknet import Darknet

from utils.utils import *
from utils.datasets import *
from utils.parse_config import *


def evaluate_raw(model, images_path, labels_path, iou_thres, conf_thres, nms_thres, input_size, batch_size, class_names=None,  plot_detections=None):
    # Get dataloader
    dataset = ListDataset(images_path=images_path, labels_path=labels_path, input_size=input_size, class_names=class_names)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    return evaluate(model, dataloader, iou_thres, conf_thres, nms_thres, input_size, class_names, plot_detections)


def evaluate(model, dataloader, iou_thres, conf_thres, nms_thres, input_size, class_names=None, plot_detections=None):
    model.eval()
    running_loss = 0

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for batch_i, (images_path, input_imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects"), 1):
    # for batch_i, (images_path, input_imgs, targets) in enumerate(dataloader, 1):
    #
    #     # Format boxes to YOLO format REL(cxcywh)
    #     targets = format2yolo(targets)

        # Extract labels
        labels += targets[:, 1].tolist()

        input_imgs = Variable(input_imgs.type(Tensor), requires_grad=False)
        dev_targets = Variable(targets.type(Tensor), requires_grad=False)

        with torch.no_grad():
            loss, detections = model(input_imgs, dev_targets)  # TODO: Validation loss working properly
            running_loss += loss.item()

            detections[..., :4] = cxcywh2xyxy(detections[..., :4])
            detections = remove_low_conf(detections, conf_thres=conf_thres)
            detections = keep_max_class(detections)
            detections = non_max_suppression(detections, nms_thres=nms_thres)
        # detections = [detections[0][23].unsqueeze(0)]

        # Targets (here, already formated due to the dataloader)
        targets[:, 2:] = xywh2xyxy(rel2abs(targets[:, 2:], input_size, input_size))

        # Show detections
        if plot_detections and batch_i <= plot_detections:
            if detections:
                use_original = True
                save_path = BASE_PATH+'/outputs/{}'.format(images_path[0].split('/')[-1])
                # Scale target bboxes
                input_img = img2img(input_imgs[0])
                ori_img = img2img(images_path[0])

                # Output
                class_ids = detections[0][:, -1]
                p_bboxes = detections[0][:, :4]
                t_bboxes = targets[targets[:, 0] == 0][:, 2:]

                if use_original:
                    p_bboxes = rescale_boxes(p_bboxes, current_shape=input_img.shape[:2], original_shape=ori_img.shape[:2])
                    t_bboxes = rescale_boxes(t_bboxes, current_shape=input_img.shape[:2], original_shape=ori_img.shape[:2])
                    plot_bboxes(ori_img, p_bboxes,  class_ids=class_ids, class_names=class_names, show_results=False,
                                t_bboxes=t_bboxes, title="Detection + ground truth ({})".format(images_path[0]), save_path=save_path)
                else:
                    plot_bboxes(input_img, p_bboxes, class_ids=class_ids, class_names=class_names, show_results=False,
                                t_bboxes=t_bboxes, title="Detection + ground truth ({})".format(images_path[0]), save_path=save_path)
            else:
                print("NO DETECTIONS")

        # Concatenate sample statistics
        sample_metrics += get_true_positives(detections, targets, iou_threshold=iou_thres)


    # Compute metrics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]

    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    # Compute loss
    val_loss = running_loss/len(dataloader)

    return precision, recall, AP, f1, ap_class, val_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1, help="size of each image batch")
    parser.add_argument("--data_config", type=str, default=BASE_PATH+"/config/custom.data", help="path to data config file")
    parser.add_argument("--model_def", type=str, help="path to model definition file")
    parser.add_argument("--weights_path", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--input_size", type=int, default=1024, help="size of each image dimension")
    parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")
    parser.add_argument("--shuffle_dataset", type=int, default=False, help="shuffle dataset")
    parser.add_argument("--validation_split", type=float, default=0.0, help="validation split [0..1]")
    parser.add_argument("--checkpoint_dir", type=str, default=BASE_PATH+"/checkpoints", help="path to checkpoint folder")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    #parser.add_argument("--top_k", type=int, default=200, help="Keep top K best hypothesis")
    parser.add_argument("--plot_detections", type=int, default=50, help="Number of detections to plot and save")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_config = parse_data_config(opt.data_config)
    test_path = data_config["test"].format(opt.input_size)
    labels_path = data_config["labels"]
    class_names = load_classes(data_config["classes"])

    # Initiate model
    model = Darknet(config_path=opt.model_def, input_size=opt.input_size).to(device)
    model.apply(weights_init_normal)

    # Load weights
    if opt.weights_path:
        if opt.weights_path.endswith(".pth"):
            model.load_state_dict(torch.load(opt.weights_path))
        else:
            model.load_darknet_weights(opt.weights_path, cutoff=None, freeze_layers=None)

    print("\nEvaluating model:\n")

    precision_list = []
    recall_list = []
    f1_list = []
    mAP_list = []
    loss_list = []

    iou_thres_grid  = [0.5] #[0.1, 0.3, 0.5, 0.7, 0.9]
    conf_thres_grid = [0.5] #[0.1, 0.3, 0.5, 0.7, 0.9]
    nms_thres_grid  = [0.3] #[0.1, 0.3, 0.5, 0.7, 0.9]

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
                    iou_thres=iou_thres,
                    conf_thres=conf_thres,
                    nms_thres=nms_thres,
                    input_size=opt.input_size,
                    batch_size=opt.batch_size,
                    class_names=class_names,
                    plot_detections=opt.plot_detections
                )

                print("Results train+test: [iou_thres={}; conf_thres={}; nms_thres={}]".format(iou_thres, conf_thres, nms_thres))
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
