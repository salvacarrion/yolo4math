from models.darknet import Darknet

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Equations
    parser.add_argument("--image_folder", type=str, default="datasets/equations/resized/1024x1024", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="models/pretrained/YOLOv3-tiny/yolov4math-tiny.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="checkpoints/yolov3_ckpt_30.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="datasets/equations/equations.names", help="path to class label file")

    # COCO
    # parser.add_argument("--image_folder", type=str, default="datasets/coco/train2014/images", help="path to dataset")
    # parser.add_argument("--model_def", type=str, default="models/pretrained/YOLOv3-608/model.cfg", help="path to model definition file")
    # parser.add_argument("--weights_path", type=str, default="models/pretrained/YOLOv3-608/yolov3.weights", help="path to weights file")
    # parser.add_argument("--class_path", type=str, default="datasets/coco/coco.names", help="path to class label file")

    parser.add_argument("--conf_thres", type=float, default=0.51, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=1024, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")

    os.makedirs("output", exist_ok=True)

    # Get data configuration
    class_names = load_classes(opt.class_path)

    # Initiate model
    model = Darknet(config_path=opt.model_def, img_size=opt.img_size).to(device)

    # Load weights
    if opt.weights_path:
        if opt.weights_path.endswith(".pth"):
            model.load_state_dict(torch.load(opt.weights_path))
        else:
            model.load_darknet_weights(opt.weights_path)

    # Set in evaluation mode
    model.eval()

    # Get dataloader
    dataset = ImageFolder(opt.image_folder, img_size=opt.img_size)

    # Build data loader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, num_workers=opt.n_cpu)

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    img_paths = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    print("\nPerforming object detection:")
    prev_time = time.time()
    for batch_i, (paths, input_imgs) in enumerate(dataloader):
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))

        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            detections[..., :4] = cxcywh2xyxy(detections[..., :4])
            detections = remove_low_conf(detections, conf_thres=opt.conf_thres)
            detections = keep_max_class(detections)
            detections = non_max_suppression(detections, nms_thres=opt.nms_thres)

        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

        # Save image and detections
        img_paths.extend(paths)
        img_detections.extend(detections)

        if batch_i == 0:
            break

    if img_detections:
        # Show detections
        process_detections(img_paths, img_detections, opt.img_size, class_names, rescale_bboxes=True, title="Detection result", colors=None)
    else:
        print("NO DETECTIONS")

    asdsd = 3