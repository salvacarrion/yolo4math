import time
import datetime
import os
import sys

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, BASE_PATH)


from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from terminaltables import AsciiTable

from models.yolov3.darknet import Darknet

from utils.logger import *
from utils.datasets import *
from utils.parse_config import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Equations
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

    # # COCO
    # parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    # parser.add_argument("--batch_size", type=int, default=1, help="size of each image batch")
    # parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    # parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    # parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    # parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    # parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")
    # parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    # parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    # parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    # parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    # parser.add_argument("--multiscale_training", default=False, help="allow for multi-scale training")
    opt = parser.parse_args()
    print(opt)

    # Make default dirs
    os.makedirs(opt.logdir, exist_ok=True)
    os.makedirs(opt.checkpoint_dir, exist_ok=True)

    logger = Logger(opt.logdir)

    # Get data configuration
    data_config = parse_data_config(opt.data_config)
    images_path = data_config["train"]
    labels_path = data_config["labels"]
    class_names = load_classes(data_config["classes"])
    colors = np.array([[200, 0, 0, 255], [0, 0, 200, 255]], dtype=np.float)/255.0

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initiate model
    model = Darknet(config_path=opt.model_def).to(device)
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
    dataset = ListDataset(images_path=images_path, labels_path=labels_path, input_size=opt.input_size, transform=data_aug)

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(opt.validation_split * dataset_size))
    if opt.shuffle_dataset:
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    # Build data loader
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=opt.shuffle_dataset, sampler=train_sampler, num_workers=opt.n_cpu, pin_memory=True, collate_fn=dataset.collate_fn)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=opt.shuffle_dataset, sampler=valid_sampler, num_workers=opt.n_cpu, pin_memory=True, collate_fn=dataset.collate_fn)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

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


    def format2yolo(targets):
        new_target = targets.clone().detach()
        # Boxes to YOLO format: From REL(xywh) to REL(cxcywh)
        bboxes_xywh_rel = new_target[:, 2:]
        new_target[:, 2:] = xywh2cxcywh(bboxes_xywh_rel)
        return new_target

    best_loss = 999999999
    # Start training
    for epoch in range(opt.epochs):
        start_time = time.time()
        model.train()
        loss_sum = 0

        # Train model
        for batch_i, (img_paths, imgs, targets) in enumerate(train_loader):
            batches_done = len(train_loader) * epoch + batch_i
            # Input target => image_i + class_id + REL(cxcywh)
            # Output target => ABS(cxcywh) + obj_conf + class_prob + class_id

            # Format boxes to YOLO format REL(cxcywh)
            targets = format2yolo(targets)

            # Sanity check I (img_path => only default transformations can be reverted)
            # f_img = img2img(imgs[0])
            # fake_targets = in_target2out_target(targets, out_h=f_img.shape[0], out_w=f_img.shape[1])
            # process_detections([f_img], [fake_targets], opt.input_size, class_names, rescale_bboxes=False, title="Augmented final ({})".format(img_paths[0]), colors=None)

            # Inputs/Targets to device
            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)

            # Fit model
            loss, outputs = model(imgs, targets)
            loss.backward()
            loss_sum += loss.item()

            # Sanity check II
            # outputs[..., :4] = cxcywh2xyxy(outputs[..., :4])
            # detections = remove_low_conf(outputs, conf_thres=opt.conf_thres)
            # detections = keep_max_class(detections)
            # detections = non_max_suppression(detections, nms_thres=opt.nms_thres)
            # if detections:
            #     process_detections([f_img], [detections[0]], opt.img_size, class_names, rescale_bboxes=False, title="Detection result", colors=None)
            # else:
            #     print("NO DETECTIONS")

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

        # Save best model
        avg_loss = loss_sum / len(train_loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            print("Saving best model.... (loss={})".format(best_loss))
            torch.save(model.state_dict(), opt.checkpoint_dir + "/yolov3_best.pht")
