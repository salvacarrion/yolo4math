import os
import sys
import time
import datetime
import argparse
import tqdm

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



def evaluate(model, images_path, labels_path, iou_thres, conf_thres, nms_thres, input_size, batch_size):
    model.eval()

    # Get dataloader
    dataset = ListDataset(images_path=images_path, labels_path=labels_path, input_size=input_size)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    # for batch_i, (_, input_imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
    for batch_i, (_, input_imgs, targets) in enumerate(dataloader):

        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        #targets[:, 2:] = cxcywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= input_size

        input_imgs = Variable(input_imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            detections = model(input_imgs)
            detections[..., :4] = cxcywh2xyxy(detections[..., :4])
            detections = remove_low_conf(detections, conf_thres=opt.conf_thres)
            detections = keep_max_class(detections)
            detections = non_max_suppression(detections, nms_thres=opt.nms_thres)

        sample_metrics += get_batch_statistics(detections, targets, iou_threshold=iou_thres)

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--data_config", type=str, default=BASE_PATH+"/config/custom.data", help="path to data config file")
    parser.add_argument("--model_def", type=str, default=BASE_PATH+"/config/yolov3-math.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default=BASE_PATH+"/checkpoints/yolov3_best__10.pth", help="if specified starts from checkpoint model")
    parser.add_argument("--input_size", type=int, default=1024, help="size of each image dimension")
    parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")
    parser.add_argument("--shuffle_dataset", type=int, default=False, help="shuffle dataset")
    parser.add_argument("--validation_split", type=float, default=0.0, help="validation split [0..1]")
    parser.add_argument("--logdir", type=str, default=BASE_PATH+"/logs", help="path to logs folder")
    parser.add_argument("--checkpoint_dir", type=str, default=BASE_PATH+"/checkpoints", help="path to checkpoint folder")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.001, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_config = parse_data_config(opt.data_config)
    test_path = data_config["test"]
    labels_path = data_config["labels"]
    class_names = load_classes(data_config["classes"])

    # Initiate model
    model = Darknet(config_path=opt.model_def).to(device)
    model.apply(weights_init_normal)

    # Load weights
    if opt.weights_path:
        if opt.weights_path.endswith(".pth"):
            model.load_state_dict(torch.load(opt.weights_path))
        else:
            model.load_darknet_weights(opt.weights_path, cutoff=None, free_layers=None)

    print("Compute mAP...")

    precision, recall, AP, f1, ap_class = evaluate(
        model,
        images_path=test_path,
        labels_path=labels_path,
        iou_thres=opt.iou_thres,
        conf_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        input_size=opt.input_size,
        batch_size=opt.batch_size,
    )

    print("Average Precisions:")
    for i, c in enumerate(ap_class):
        print("+ Class '{}' ({}) - AP: {}".format(c, class_names[c], AP[i]))

    print("mAP: {}".format(AP.mean()))
