import glob
import random
import os
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import albumentations as A

# from utils.augmentations import horisontal_flip
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from utils.utils import *

class ListDatasetSSD(Dataset):
    def __init__(self, images_path, labels_path, input_size, transform=None, multiscale=False, normalized_bboxes=True,
                 balance_classes=False, class_names=None):
        self.img_files = []
        self.label_files = []
        self.input_size = input_size
        self.transform = transform
        self.normalized_bboxes = normalized_bboxes
        self.multiscale = multiscale
        self.min_input_size = self.input_size - 3 * 32  # Network stride
        self.max_input_size = self.input_size + 3 * 32
        self.balance_classes = balance_classes
        self.class_counter = np.zeros(len(class_names))
        self.ignored, self.total = 0, 0

        # Data format
        self.data_format = A.Compose([
            A.ToGray(p=1.0),
            A.LongestMaxSize(max_size=self.input_size, interpolation=cv2.INTER_AREA),
            A.PadIfNeeded(min_height=self.input_size, min_width=self.input_size, border_mode=cv2.BORDER_CONSTANT,
                          value=(128, 128, 128)),
        ], p=1)

        # Get files
        self.img_files = []
        self.label_files = []
        for filename in os.listdir(images_path):
            img_path = os.path.join(images_path, filename)
            label_path = os.path.join(labels_path, filename.replace(".jpg", ".txt").replace(".png", ".txt"))

            # Check if label file exists
            if os.path.exists(label_path):
                self.img_files.append(img_path)
                self.label_files.append(label_path)

    def __getitem__(self, index):
        # For debugging
        # print("Index: {}".format(index))
        # index = 174
        index = index
        # Get paths
        img_path = self.img_files[index % len(self.img_files)].rstrip()
        label_path = self.label_files[index % len(self.img_files)].rstrip()

        # Load bboxes
        bboxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))

        # Remove elements to balance training
        if self.balance_classes:
            kept_indices = balance_batch(bboxes[:, 0].numpy(), self.class_counter)
            bboxes = bboxes[kept_indices]

        # Ignore image if empty boxes
        if bboxes.size(0) == 0:
            self.ignored += 1
            print("Ignored {}/{}".format(self.ignored, self.total))
            return img_path, None, None

        # Load image
        img = np.asarray(Image.open(img_path).convert('RGB'))

        # Get input dimensions
        h, w, c = img.shape
        h_factor, w_factor = (h, w) if self.normalized_bboxes else (1, 1)

        # Convert bboxes
        bboxes_labels = bboxes[:, 0]
        bboxes_xywh_rel = bboxes[:, 1:]
        bboxes_xyxy_rel = xywh2xyxy(bboxes_xywh_rel)
        bboxes_xyxy_abs = rel2abs(bboxes_xyxy_rel, h_factor, w_factor)  # image size

        # Sanity check I
        # plot_bboxes(img, bboxes_xyxy_abs, title="Original ({})".format(img_path))

        # Convert bboxes to albumentations [BBOXES=NUMPY]
        bboxes_albu = convert_bboxes_to_albumentations(bboxes_xyxy_abs.numpy(), source_format='pascal_voc', rows=img.shape[0], cols=img.shape[1])

        # Default image format
        img_format = self.data_format(image=img, bboxes=bboxes_albu)
        img = img_format['image']#[..., 0]  # Remove redundant channels
        # img = img[..., np.newaxis]  # Add channel dimension
        bboxes_albu = img_format['bboxes']

        # Custom transformations
        if self.transform:
            # Perform augmentation
            img_format = self.transform(image=img, bboxes=bboxes_albu)
            img = img_format['image']
            bboxes_albu = img_format['bboxes']

        # Convert bboxes from albumentations [BBOXES=NUMPY]
        bboxes_xyxy_abs = convert_bboxes_from_albumentations(bboxes_albu, target_format='pascal_voc', rows=img.shape[0], cols=img.shape[1])

        # Sanity check II
        # print("Regions: {}".format(len(bboxes_xyxy_abs)))
        # plot_bboxes(img, bboxes_xyxy_abs, title="Augmented ({})".format(img_path))

        # Convert (PIL/Numpy) to PyTorch Tensor
        img = transforms.ToTensor()(Image.fromarray(img))
        img_c, img_h, img_w = img.shape

        # Fix bboxes (keep into the region boundaries)
        bboxes_xyxy_abs = torch.tensor(bboxes_xyxy_abs)
        bboxes_xyxy_abs, kept_indices = fix_bboxes(bboxes_xyxy_abs, img_h, img_w)  # Use new size (padding)
        bboxes_labels += 1  # Background starts at 0
        bboxes_labels = bboxes_labels[kept_indices]  # Math dimensions

        # Keep embedded/isolated (debugging)
        kept_indices = torch.ByteTensor(bboxes_labels == 2)
        bboxes_labels = bboxes_labels[kept_indices]
        bboxes_xyxy_abs = bboxes_xyxy_abs[kept_indices]

        # Sanity check III
        # plot_bboxes(img, bboxes_xyxy_abs, title="Augmented Fix ({})".format(img_path))

        # Format boxes to REL(xywh)
        boxes_xyxy_rel = abs2rel(bboxes_xyxy_abs, img_h, img_w)  # new size (padding)

        # print(self.class_counter.tolist())
        self.total += 1
        return img_path, img, boxes_xyxy_rel, bboxes_labels.type(torch.int64)

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).

        This describes how to combine these tensors of different sizes. We use lists.

        Note: this need not be defined in this Class, can be standalone.

        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        img_paths = list()
        images = list()
        boxes = list()
        labels = list()

        for b in batch:
            img_paths.append(b[0])
            images.append(b[1])
            boxes.append(b[2])
            labels.append(b[3])

        images = torch.stack(images, dim=0)

        return img_paths, images, boxes, labels  # tensor (N, 3, 300, 300), 3 lists of N tensors each

    def __len__(self):
        return len(self.img_files)


class ListDataset(Dataset):
    def __init__(self, images_path, labels_path, input_size, transform=None, multiscale=False, normalized_bboxes=True,
                 balance_classes=False, class_names=None):
        self.img_files = []
        self.label_files = []
        self.input_size = input_size
        self.transform = transform
        self.normalized_bboxes = normalized_bboxes
        self.multiscale = multiscale
        self.min_input_size = self.input_size - 3 * 32  # Network stride
        self.max_input_size = self.input_size + 3 * 32
        self.balance_classes = balance_classes
        self.class_counter = np.zeros(len(class_names))
        self.ignored, self.total = 0, 0

        # Data format
        self.data_format = A.Compose([
            A.ToGray(p=1.0),
            A.LongestMaxSize(max_size=self.input_size, interpolation=cv2.INTER_AREA),
            A.PadIfNeeded(min_height=self.input_size, min_width=self.input_size, border_mode=cv2.BORDER_CONSTANT,
                          value=(128, 128, 128)),
        ], p=1)

        # Get files
        self.img_files = []
        self.label_files = []
        for filename in os.listdir(images_path):
            img_path = os.path.join(images_path, filename)
            label_path = os.path.join(labels_path, filename.replace(".jpg", ".txt").replace(".png", ".txt"))

            # Check if label file exists
            if os.path.exists(label_path):
                self.img_files.append(img_path)
                self.label_files.append(label_path)

    def __getitem__(self, index):
        # For debugging
        # print("Index: {}".format(index))
        # index = 174
        index = index
        # Get paths
        img_path = self.img_files[index % len(self.img_files)].rstrip()
        label_path = self.label_files[index % len(self.img_files)].rstrip()

        # Load bboxes
        bboxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))

        # Remove elements to balance training
        if self.balance_classes:
            kept_indices = balance_batch(bboxes[:, 0].numpy(), self.class_counter)
            bboxes = bboxes[kept_indices]

        # Ignore image if empty boxes
        if bboxes.size(0) == 0:
            self.ignored += 1
            print("Ignored {}/{}".format(self.ignored, self.total))
            return img_path, None, None

        # Load image
        img = np.asarray(Image.open(img_path).convert('RGB'))

        # Get input dimensions
        h, w, c = img.shape
        h_factor, w_factor = (h, w) if self.normalized_bboxes else (1, 1)

        # Convert bboxes
        bboxes_labels = bboxes[:, 0]
        bboxes_xywh_rel = bboxes[:, 1:]
        bboxes_xyxy_rel = xywh2xyxy(bboxes_xywh_rel)
        bboxes_xyxy_abs = rel2abs(bboxes_xyxy_rel, h_factor, w_factor)  # image size

        # Sanity check I
        # plot_bboxes(img, bboxes_xyxy_abs, title="Original ({})".format(img_path))

        # Convert bboxes to albumentations [BBOXES=NUMPY]
        bboxes_albu = convert_bboxes_to_albumentations(bboxes_xyxy_abs.numpy(), source_format='pascal_voc', rows=img.shape[0], cols=img.shape[1])

        # Default image format
        img_format = self.data_format(image=img, bboxes=bboxes_albu)
        img = img_format['image'][..., 0]  # Remove redundant channels
        # img = img[..., np.newaxis]  # Add channel dimension
        bboxes_albu = img_format['bboxes']

        # Custom transformations
        if self.transform:
            # Perform augmentation
            img_format = self.transform(image=img, bboxes=bboxes_albu)
            img = img_format['image']
            bboxes_albu = img_format['bboxes']

        # Convert bboxes from albumentations [BBOXES=NUMPY]
        bboxes_xyxy_abs = convert_bboxes_from_albumentations(bboxes_albu, target_format='pascal_voc', rows=img.shape[0], cols=img.shape[1])

        # Sanity check II
        # print("Regions: {}".format(len(bboxes_xyxy_abs)))
        # plot_bboxes(img, bboxes_xyxy_abs, title="Augmented ({})".format(img_path))

        # Convert (PIL/Numpy) to PyTorch Tensor
        img = transforms.ToTensor()(Image.fromarray(img))
        img_c, img_h, img_w = img.shape

        # Fix bboxes (keep into the region boundaries)
        bboxes_xyxy_abs = torch.tensor(bboxes_xyxy_abs)
        bboxes_xyxy_abs, kept_indices = fix_bboxes(bboxes_xyxy_abs, img_h, img_w)  # Use new size (padding)
        bboxes_labels = bboxes_labels[kept_indices]  # Math dimensions

        # Sanity check III
        # plot_bboxes(img, bboxes_xyxy_abs, title="Augmented Fix ({})".format(img_path))

        # Format boxes to REL(xywh)
        boxes_xywh_abs = xyxy2xywh(bboxes_xyxy_abs)
        boxes_xywh_rel = abs2rel(boxes_xywh_abs, img_h, img_w)  # new size (padding)

        # For debugging
        # print("xyxy_abs: {}".format(bboxes_xyxy[0]))
        # print("cxcywh_abs: {}".format(boxes_cxcywh[0]))
        # print("cxcywh_rel: {}".format(boxes_cxcywh[0]))

        # Transform targets (bboxes)
        targets = torch.zeros((len(boxes_xywh_rel), 6))  # 0(batch), class_id + cxcywh_rel (REL)
        targets[:, 1] = bboxes_labels
        targets[:, 2:] = boxes_xywh_rel

        # # [Debug]: Keep class X
        # targets = targets[targets[:, 1] == 0]

        # Update seen classes
        for c in targets[:, 1]:
            self.class_counter[int(c)] += 1

        # print(self.class_counter.tolist())
        self.total += 1
        return img_path, img, targets

    def collate_fn(self, batch):
        img_paths, imgs, targets = list(zip(*batch))

        # If empty, leave
        if targets[0] is None:
            return None, None, None

        # Get targets as a list of tensors
        targets = [boxes for boxes in targets if boxes is not None]

        # Add index to track this batch
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i

        # Stack all boxes (fixed sized)
        targets = torch.cat(targets, dim=0)

        # Images to Tensor
        imgs = torch.stack([img for img in imgs])
        return img_paths, imgs, targets

    def __len__(self):
        return len(self.img_files)


class ImageFolder(Dataset):
    def __init__(self, images_path, input_size, transform=None):
        self.images = []
        self.input_size = input_size
        self.transform = transform

        # Data format
        self.data_format = A.Compose([
            A.LongestMaxSize(max_size=self.input_size, interpolation=cv2.INTER_AREA),
            A.PadIfNeeded(min_height=self.input_size, min_width=self.input_size, border_mode=cv2.BORDER_CONSTANT,
                          value=(128, 128, 128))
        ], p=1)

        # Get images
        for file in os.listdir(images_path):
            self.images.append(os.path.join(images_path, file))

    def __getitem__(self, index):
        # For debugging
        # print("Index: {}".format(index))
        # index = 174
        image_path = self.images[index % len(self.images)]

        # Load image as RGB
        img = np.asarray(Image.open(image_path).convert('RGB'))  # L

        # Default image format
        img = self.data_format(image=img)
        img = img['image']

        if self.transform:
            # Perform augmentation
            augmented_data = self.transform(image=img)
            img = augmented_data['image']

        # Convert image (PIL/Numpy) to PyTorch Tensor
        img = transforms.ToTensor()(img)
        return image_path, img

    def __len__(self):
        return len(self.images)

