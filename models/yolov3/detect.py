import os
import time
import datetime
import argparse
import os
import sys

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, BASE_PATH)


import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

from models.yolov3.darknet import Darknet

from utils.datasets import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="/home/salvacarrion/Documents/datasets/equations/train", help="path to dataset")
    parser.add_argument("--model_def", type=str, default=BASE_PATH+"/config/yolov3-math.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default=BASE_PATH+"/checkpoints/yolov3_best.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="/home/salvacarrion/Documents/datasets/equations/class.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.6, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")
    parser.add_argument("--input_size", type=int, default=1024, help="size of each image dimension")
    parser.add_argument("--output_dir", type=str, default=BASE_PATH+'/output', help="path to checkpoint folder")
    opt = parser.parse_args()
    print(opt)

    # Make default dirs
    os.makedirs(opt.output_dir, exist_ok=True)

    # Get data configuration
    class_names = load_classes(opt.class_path)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    # Initiate model
    model = Darknet(config_path=opt.model_def).to(device)
    model.apply(weights_init_normal)

    # Load weights
    if opt.weights_path:
        if opt.weights_path.endswith(".pth"):
            model.load_state_dict(torch.load(opt.weights_path))
            print("Model loaded! (*.pth)")
        else:
            model.load_darknet_weights(opt.weights_path)
            print("Model loaded!")

    # Set in evaluation mode
    model.eval()

    # Get dataloader
    dataset = ImageFolder(opt.image_folder, input_size=opt.input_size)

    # Build data loader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, num_workers=opt.n_cpu)

    print("\nPerforming object detection:")
    prev_time = time.time()
    for batch_i, (img_paths, imgs) in enumerate(dataloader):
        # Configure input
        input_imgs = Variable(imgs.type(Tensor))

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

        # Show detections
        if detections:
            process_detections(img_paths, detections, opt.input_size, class_names, rescale_bboxes=True,
                               show_results=False, save_path=opt.output_dir, title="Detection result", colors=None)
        else:
            print("\t\t=> NO DETECTIONS: (#{})".format(img_paths[0]))

        if batch_i == 5:
            break
