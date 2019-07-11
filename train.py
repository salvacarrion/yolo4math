from models.darknet import Darknet

from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=1, help="number of gradientaccums before step")

    # Equations
    parser.add_argument("--dataset_path", type=str, default="datasets/equations/resized/1024x1024", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="models/pretrained/YOLOv3/yolov3-tiny4math.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default=None, help="path to weights file")
    parser.add_argument("--class_path", type=str, default="datasets/equations/equations.names", help="path to class label file")

    # COCO
    # parser.add_argument("--dataset_path", type=str, default="datasets/coco/train2014/images/", help="path to dataset")
    # parser.add_argument("--class_path", type=str, default="datasets/coco/coco.names", help="path to class label file")
    # parser.add_argument("--model_def", type=str, default="models/pretrained/YOLOv3-608/model.cfg", help="path to model definition file")
    # parser.add_argument("--weights_path", type=str, default="models/pretrained/YOLOv3-608/yolov3.weights", help="path to weights file")
    #parser.add_argument("--model_def", type=str, default="models/pretrained/YOLOv3/yolov3-tiny.cfg", help="path to model definition file")
    #parser.add_argument("--weights_path", type=str, default="models/pretrained/YOLOv3/yolov3-tiny.weights", help="path to weights file")

    # COCO BIG
    # Labels: /home/salvacarrion/Documents/datasets/coco2014/coco/labels/val2014
    # Images: /home/salvacarrion/Documents/datasets/coco2014/coco/images/val2014
    #         /home/salvacarrion/Documents/datasets/coco2014/coco/labels/val2014/COCO_val2014_000000079362.txt n
    #parser.add_argument("--dataset_path", type=str, default="/home/salvacarrion/Documents/datasets/coco2014/coco/images/val2014", help="path to dataset")
    #parser.add_argument("--class_path", type=str, default="datasets/coco/coco.names", help="path to class label file")
    #parser.add_argument("--model_def", type=str, default="models/pretrained/YOLOv3/yolov3-tiny.cfg", help="path to model definition file")
    #parser.add_argument("--weights_path", type=str, default=None, help="path to weights file")

    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")  # Anchors (abs-wh) are dependent of the input size
    parser.add_argument("--checkpoint_interval", type=int, default=10, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=5, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=False, help="allow for multi-scale training")
    parser.add_argument("--shuffle_dataset", default=True, help="shuffle dataset")
    parser.add_argument("--validation_split", default=0.0, help="validation split")
    parser.add_argument("--random_seed", default=42, help="random seed")
    opt = parser.parse_args()
    print(opt)

    logger = Logger("logs")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Get data configuration
    class_names = load_classes(opt.class_path)
    colors = np.array([[200, 0, 0, 255], [0, 0, 200, 255]], dtype=np.float)/255.0

    # Initiate model
    print("INPUT SIZE: {}x{}".format(opt.img_size, opt.img_size))
    print("CONF THRES.: {}".format(opt.conf_thres))
    print("NMS THRES.: {}".format(opt.nms_thres))
    print("BATCH SIZE.: {}".format(opt.batch_size))
    model = Darknet(config_path=opt.model_def, img_size=opt.img_size, num_classes=len(class_names), in_channels=3).to(device)
    model.apply(weights_init_normal)

    # Load weights
    if opt.weights_path:
        if opt.weights_path.endswith(".pth"):
            model.load_state_dict(torch.load(opt.weights_path))
        else:
            model.load_darknet_weights(opt.weights_path, cutoff=None, free_layers=None)

    # Data augmentation
    data_aug = A.Compose([
        # Extra
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=(0.0, 0.0625), rotate_limit=2,
                           interpolation=cv2.INTER_AREA, border_mode=cv2.BORDER_REPLICATE, p=1.0),
    ], p=1.0)

    # Get dataloader
    dataset = EQDataset(opt.dataset_path, img_size=opt.img_size, transform=data_aug, multiscale=opt.multiscale_training)
    #dataset = COCODataset(opt.dataset_path, img_size=opt.img_size, transform=data_aug, multiscale=opt.multiscale_training)

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(opt.validation_split * dataset_size))
    if opt.shuffle_dataset:
        np.random.seed(opt.random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    # Build data loader
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, sampler=train_sampler, num_workers=opt.n_cpu, pin_memory=True, collate_fn=dataset.collate_fn)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, sampler=valid_sampler, num_workers=opt.n_cpu, pin_memory=True, collate_fn=dataset.collate_fn)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]

    best_loss = 999999999
    # Start training
    for epoch in range(opt.epochs):
        model.train()
        start_time = time.time()
        print("Epoch {}/{} ----------------->".format(epoch, opt.epochs))

        # Train model
        for batch_i, (img_paths, imgs, targets) in enumerate(train_loader):

            batches_done = len(train_loader) * epoch + batch_i
            print("\t- Batch {}/{}".format(batch_i+1, len(train_loader)))

            # Input target => image_i + class_id + REL(cxcywh)
            # Output target => ABS(cxcywh) + obj_conf + class_prob + class_id

            # Sanity check I (img_path => only default transformations can be reverted)
            fake_targets = in_target2out_target(targets, out_h=opt.img_size, out_w=opt.img_size)
            process_detections([imgs[0]], [fake_targets], opt.img_size, class_names, rescale_bboxes=False, title="Augmented final", colors=None)

            # Inputs/Targets to device
            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)

            # Fit model
            loss, outputs = model(imgs, targets)
            loss.backward()

            # Sanity check II
            outputs[..., :4] = cxcywh2xyxy(outputs[..., :4])
            detections = remove_low_conf(outputs, conf_thres=opt.conf_thres)
            detections = keep_max_class(detections)
            detections = non_max_suppression(detections, nms_thres=opt.nms_thres)
            if detections:
                process_detections([imgs[0]], [detections[0]], opt.img_size, class_names, rescale_bboxes=False, title="Detection result", colors=None)
            else:
                print("NO DETECTIONS")

            if batches_done % opt.gradient_accumulations:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            # ----------------
            #   Log progress
            # ----------------

            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt.epochs, batch_i, len(train_loader))
            metric_table = [["Metrics", *["YOLO Layer {}".format(i) for i in range(len(model.yolo_layers))]]]

            # Log metrics at each YOLO layer
            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                metric_table += [[metric, *row_metrics]]

                # Tensorboard logging
                tensorboard_log = []
                for j, yolo in enumerate(model.yolo_layers):
                    for name, metric in yolo.metrics.items():
                        if name != "grid_size":
                            tensorboard_log += [("{}_{}".format(name, j+1), metric)]
                tensorboard_log += [("loss", loss.item())]
                logger.list_of_scalars_summary(tensorboard_log, batches_done)

            log_str += AsciiTable(metric_table).table
            log_str += "\nTotal loss {}".format(loss.item())

            # Determine approximate time left for epoch
            epoch_batches_left = len(train_loader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += "\n---- ETA {}".format(time_left)

            print(log_str)

            model.seen += imgs.size(0)

        #
        # if epoch % opt.evaluation_interval == 0:
        #     print("\n---- Evaluating Model ----")
        #     # Evaluate the model on the validation set
        #     precision, recall, AP, f1, ap_class = evaluate(
        #         model,
        #         dataloader=validation_loader,
        #         iou_thres=0.5,
        #         conf_thres=0.5,
        #         nms_thres=0.5,
        #         img_size=opt.img_size,
        #         batch_size=opt.batch_size,
        #     )
        #     evaluation_metrics = [
        #         ("val_precision", precision.mean()),
        #         ("val_recall", recall.mean()),
        #         ("val_mAP", AP.mean()),
        #         ("val_f1", f1.mean()),
        #     ]
        #     logger.list_of_scalars_summary(evaluation_metrics, epoch)
        #
        #     # Print class APs and mAP
        #     ap_table = [["Index", "Class name", "AP"]]
        #     for i, c in enumerate(ap_class):
        #         ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
        #     print(AsciiTable(ap_table).table)
        #     print("---- mAP".format(AP.mean()))

        # if epoch % opt.checkpoint_interval == 0:
        if loss < best_loss:
            best_loss = loss
            filename = "checkpoints/yolov3_best.pth"
            print("Saving model (best_loss={}): {}".format(best_loss, filename))
            torch.save(model.state_dict(), filename)

    # Save last mode
    filename = "checkpoints/yolov3_ckpt_{}.pth".format("last")
    print("Saving model: {}".format(filename))
    torch.save(model.state_dict(), filename)
