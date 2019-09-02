import time
import datetime
import os
import sys
import math

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, BASE_PATH)


from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

from terminaltables import AsciiTable

from models.ssd.model import SSD300, MultiBoxLoss
# from models.ssd.datasets import PascalVOCDataset
from models.ssd.utils import *
from models.ssd.detect import *
from models.ssd.test import evaluate

from utils.datasets import *
from utils.parse_config import *


def eval(model, running_loss, epoch_batches_done, epoch , hist=False):
    # ********* AUX VARS *********
    train_loss = running_loss / epoch_batches_done if epoch_batches_done else 0
    train_conf_loss = running_conf_loss / epoch_batches_done if epoch_batches_done else 0
    train_loc_loss = running_loc_loss / epoch_batches_done if epoch_batches_done else 0

    # ********* LOG PROCESS *********
    # [TB] Scalars
    writer.add_scalar(tag="loss", scalar_value=train_loss, global_step=epoch)
    writer.add_scalar(tag="conf_loss", scalar_value=train_conf_loss, global_step=epoch)
    writer.add_scalar(tag="loc_loss", scalar_value=train_loc_loss, global_step=epoch)

    if hist:
        # [TB] Histogram / one per epoch (takes more time (0.x seconds))
        for name, param in model.named_parameters():
            writer.add_histogram(name, param.clone().cpu().data.numpy(), global_step=epoch)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Equations
    parser.add_argument("--epochs", type=int, default=300, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="size of each image batch")
    parser.add_argument("--data_config", type=str, default=BASE_PATH+"/config/custom.data", help="path to data config file")
    parser.add_argument("--weights_path", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--input_size", type=int, default=(1024, 1024), help="size of each image dimension")#(512, 400)
    parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")
    parser.add_argument("--shuffle_dataset", type=int, default=True, help="shuffle dataset")
    parser.add_argument("--validation_split", type=float, default=0.1, help="validation split [0..1]")
    parser.add_argument("--checkpoint_dir", type=str, default=BASE_PATH+"/checkpoints", help="path to checkpoint folder")
    parser.add_argument("--logdir", type=str, default=BASE_PATH+"/logs", help="path to logs folder")
    parser.add_argument("--log_name", type=str, default="SSD-1024-1024-adam", help="name of the experiment (tensorboard)")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    opt = parser.parse_args()
    print(opt)

    # Make default dirs
    os.makedirs(opt.logdir, exist_ok=True)
    os.makedirs(opt.checkpoint_dir, exist_ok=True)

    # Get data configuration
    data_config = parse_data_config(opt.data_config)
    images_path = data_config["train"].format(1024)
    labels_path = data_config["labels"]
    class_names = load_classes(data_config["classes"])
    class_names.insert(0, 'background')
    colors = np.array([[200, 0, 0, 255], [0, 0, 200, 255]], dtype=np.float)/255.0
    #
    # class_names = {"background": 0, "aeroplane": 1, "bicycle": 2, "bird": 3, "boat": 4, "bottle": 5, "bus": 6, "car": 7,
    #                "cat": 8, "chair": 9, "cow": 10, "diningtable": 11, "dog": 12, "horse": 13, "motorbike": 14,
    #                "person": 15, "pottedplant": 16, "sheep": 17, "sofa": 18, "train": 19, "tvmonitor": 20}
    # class_names = list(class_names.keys())

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model or load checkpoint
    if not opt.weights_path:
        model = SSD300(n_classes=len(class_names), input_size=opt.input_size)
    else:
        model = torch.load(opt.weights_path).to(device)
        model.priors_cxcy = model.create_prior_boxes(model.input_size)
        model.priors_cxcy = model.priors_cxcy.to(device)

    print("Number of priors: {}".format(len(model.priors_cxcy)))

    # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
    biases = list()
    not_biases = list()
    for param_name, param in model.named_parameters():
        if param.requires_grad:
            if param_name.endswith('.bias'):
                biases.append(param)
            else:
                not_biases.append(param)
    lr = 1e-4
    optimizer = torch.optim.Adam(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
                                 lr=lr, weight_decay=5e-4)

    grad_clip = None  # clip if gradients are exploding, which may happen at larger batch sizes (sometimes at 32) - you will recognize it by a sorting error in the MuliBox loss calculation

    # Move to default device
    model = model.to(device)
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy, alpha=2.0).to(device)

    # Data augmentation
    data_aug = A.Compose([
        # Extra
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=(0.0, 0.0625), rotate_limit=2,
                           interpolation=cv2.INTER_AREA, border_mode=cv2.BORDER_REPLICATE, p=1.0),
    ], p=1.0)

    # Get dataloader
    # images_path = "/home/salvacarrion/Documents/Programming/Python/Projects/a-PyTorch-Tutorial-to-Object-Detection"
    # dataset = PascalVOCDataset(dataset_path=images_path, input_size=opt.input_size,
    #                          transform=data_aug, balance_classes=False, class_names=class_names, single_channel=False)
    dataset = ListDatasetSSD(images_path=images_path, labels_path=labels_path, input_size=opt.input_size,
                             transform=data_aug, balance_classes=False, class_names=class_names, single_channel=False)

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
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, sampler=train_sampler, num_workers=opt.n_cpu, pin_memory=True, collate_fn=dataset.collate_fn)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, sampler=valid_sampler, num_workers=opt.n_cpu, pin_memory=True, collate_fn=dataset.collate_fn)

    # Optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    metrics = [
        "conf_loss",
        "loc_loss",
    ]

    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter(opt.logdir + "/{}".format(opt.log_name))
    # Create graph
    # dummy_input = Variable(torch.zeros(1, 3, opt.input_size, opt.input_size).to(device))
    # writer.add_graph(model, dummy_input, True)

    best_loss = 999999999
    start_epoch = 0
    batches_done = start_epoch * len(train_loader)
    # Start training
    for epoch in range(start_epoch, opt.epochs):
        start_time = time.time()
        model.train()
        running_loss = 0
        running_conf_loss = 0
        running_loc_loss = 0

        # Train model
        epoch_batches_done = 0

        for batch_i, (img_paths, images, boxes, labels) in enumerate(train_loader, 1):

            if boxes is None or len(boxes) == 0:
                # print("Skipping image #{}...".format(batch_i))
                continue
            # Ignore empty targets (problems with the data)
            # for batch_i, (images, boxes, labels, _) in enumerate(train_loader, 1):
            batches_done += 1
            epoch_batches_done += 1

            # Sanity check I (img_path => only default transformations can be reverted)
            # pl_bboxes = np.array(boxes[0].data.numpy())
            # pl_labels = np.array(labels[0].data.numpy())
            # plot_bboxes(images[0], pl_bboxes, pl_labels, [1.0] * len(pl_bboxes), class_names,
            #             title="Augmented final ({})".format(0), colors=colors, coords_rel=True)

            # Inputs/Targets to device
            # Move to default device
            images = images.to(device)  # (batch_size (N), 3, 300, 300)
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]

            optimizer.zero_grad()

            # Forward prop.
            predicted_locs, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, n_classes)
            loss, l_metrics = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar
            loss.backward()
            optimizer.step()

            # # Clip gradients, if necessary
            # if grad_clip is not None:
            #     clip_gradient(optimizer, grad_clip)

            # Track losses
            running_loss += loss.item()
            running_conf_loss += l_metrics['conf_loss'].item()
            running_loc_loss += l_metrics['loc_loss'].item()

            # if batches_done % opt.gradient_accumulations == 0:  # Starts at 1: when mod==0 => reset
            #     # Accumulates gradient before each step
            #

            # ********* PRINT PROCESS *********
            # Build log and add data to tensorboard
            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch+1, opt.epochs, batch_i, len(train_loader))
            metric_table = [["Metric", 'Value']]
            # Log metrics
            for i, metric in enumerate(metrics):
                # Add relevant data
                metric_table += [[metric, "%.6f" % l_metrics[metric]]]
            log_str += AsciiTable(metric_table).table
            log_str += "\nTotal loss: {:.5f}".format(loss.item())

            # Determine approximate time left for epoch
            epoch_batches_left = len(train_loader) - batch_i
            avg_time_minibatch = (time.time() - start_time) / batch_i
            time_left = datetime.timedelta(seconds=epoch_batches_left * avg_time_minibatch)
            log_str += "\nETA: {}".format(time_left)
            print(log_str)

            # if (batch_i-1) % int(360/opt.batch_size) == 0:
            #     model.eval()
            #     print("Saving model... (backup)")
            #     torch.save(model, opt.checkpoint_dir + "/ssd_last.pth")
            #     print("Saved!")
            #
            #     # img_path = '/home/salvacarrion/Documents/datasets/VOC2007/JPEGImages/000001.jpg'
            #     # original_image = Image.open(img_path, mode='r')
            #     # original_image = original_image.convert('RGB')
            #     img_path = '/home/salvacarrion/Documents/datasets/equations/1024/10.1.1.1.2018_5.jpg'
            #     #img_path = '/home/salvacarrion/Documents/datasets/VOC2007/JPEGImages/000001.jpg'
            #     myimg = detect(model, img_path, min_score=0.2, max_overlap=0.5, top_k=200, class_names=class_names,
            #                    save_path="output/mine_e{}_{}.jpg".format((epoch+1), batch_i-1),
            #                    input_size=opt.input_size)
            #     #myimg.save("output/mine_e{}_{}.jpg".format(epoch-1, batch_i-1), "JPEG")
            #     # myimg.show()
            #     model.train()

            # # Evaluate model
            # eval(model, running_loss, epoch_batches_done, batches_done)
        # Evaluate model
        eval(model, running_loss, epoch_batches_done, epoch+1, True)

        # Loss
        train_loss = running_loss / epoch_batches_done

        # Save best model
        if train_loss < best_loss:
            best_loss = train_loss
            print("Saving best model.... (loss={})".format(best_loss))
            torch.save(model, opt.checkpoint_dir + "/ssd_best.pth")

    # Close writer
    writer.close()
