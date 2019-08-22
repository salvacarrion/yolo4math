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

from models.yolov3.darknet import Darknet

from utils.utils import *
from utils.datasets import *
from utils.parse_config import *


def evaluate_raw(model, images_path, labels_path, iou_thres, conf_thres, nms_thres, input_size, batch_size, class_names=None):
    # Get dataloader
    dataset = ListDataset(images_path=images_path, labels_path=labels_path, input_size=input_size, class_names=class_names)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    return evaluate(model, dataloader, iou_thres, conf_thres, nms_thres, input_size, class_names)


def evaluate(model, dataloader, iou_thres, conf_thres, nms_thres, input_size, class_names=None, plot_detections=False):
    model.eval()
    running_loss = 0

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    # for batch_i, (images_path, input_imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects"), 1):
    for batch_i, (images_path, input_imgs, targets) in enumerate(dataloader):

        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        #targets[:, 2:] = cxcywh2xyxy(targets[:, 2:])
        #targets[:, 2:] *= input_size

        input_imgs = Variable(input_imgs.type(Tensor), requires_grad=False)
        dev_targets = Variable(targets.type(Tensor), requires_grad=False)

        with torch.no_grad():
            loss, detections = model(input_imgs, dev_targets)
            running_loss += loss.item()

            detections[..., :4] = cxcywh2xyxy(detections[..., :4])
            detections = remove_low_conf(detections, conf_thres=conf_thres)
            detections = keep_max_class(detections)
            detections = non_max_suppression(detections, nms_thres=nms_thres)
        # detections = [detections[0][23].unsqueeze(0)]

        # Targets (here, already formated due to the dataloader)
        targets[:, 2:] = xywh2xyxy(rel2abs(targets[:, 2:], input_size, input_size))

        # Show detections
        if plot_detections:
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

                # if batch_i == 50:
                #     break
            else:
                print("NO DETECTIONS")
        sample_metrics += get_true_positives(detections, targets, iou_threshold=iou_thres)

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class, running_loss/len(dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--data_config", type=str, default=BASE_PATH+"/config/custom.data", help="path to data config file")
    parser.add_argument("--model_def", type=str, default=BASE_PATH+"/config/yolov3-math.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--input_size", type=int, default=1024, help="size of each image dimension")
    parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")
    parser.add_argument("--shuffle_dataset", type=int, default=False, help="shuffle dataset")
    parser.add_argument("--validation_split", type=float, default=0.0, help="validation split [0..1]")
    parser.add_argument("--logdir", type=str, default=BASE_PATH+"/logs", help="path to logs folder")
    parser.add_argument("--checkpoint_dir", type=str, default=BASE_PATH+"/checkpoints", help="path to checkpoint folder")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_config = parse_data_config(opt.data_config)
    test_path = data_config["test"]
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
            model.load_darknet_weights(opt.weights_path, cutoff=None, free_layers=None)

    print("Compute mAP...")

    precision, recall, AP, f1, ap_class = evaluate_raw(
        model,
        images_path=test_path,
        labels_path=labels_path,
        iou_thres=opt.iou_thres,
        conf_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        input_size=opt.input_size,
        batch_size=opt.batch_size,
        class_names=class_names,
    )

    print("Average Precisions:")
    for i, c in enumerate(ap_class):
        print("+ Class '{}' ({}) - AP: {}".format(c, class_names[c], AP[i]))

    print("mAP: {}".format(AP.mean()))
