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


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


class YOLODataset(Dataset):
    def __init__(self, folder_path, transform=None, multiscale=False, img_size=1024):
        self.transform = transform
        self.wd = os.path.abspath(folder_path)
        self.multiscale = multiscale  # Requires relative coordinates

        # Other
        self.img_size = img_size
        self.force_resize = False
        self.abs_coords = False
        self.network_stride = 32
        self.min_size = self.img_size - 3 * self.network_stride
        self.max_size = self.img_size + 3 * self.network_stride
        self.batch_count = 0

        # Load dataset
        json_dataset = load_dataset(folder_path + '/train.json')
        self.images = json_dataset['images']
        self.annotations = json_dataset['annotations']
        self.class_names = json_dataset['categories']

    def __getitem__(self, index):
        image_data = self.images[index]
        image_id = str(image_data['id'])
        image_path = os.path.join(self.wd, image_data['filename'])
        bboxes = [x['bbox'] for x in self.annotations[image_id]]
        classes_id = [int(x['category_id']) for x in self.annotations[image_id]]

        # Load image as grayscale
        img = np.asarray(Image.open(image_path).convert('RGB'))  #L

        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        if self.transform:
            # Convert bboxes to albumentations
            bboxes = convert_bboxes_to_albumentations(bboxes, source_format='coco',
                                                      rows=img.shape[0], cols=img.shape[1])

            # Perform augmentation
            augmented_data = self.transform(image=img, bboxes=bboxes)
            img = augmented_data['image']

            # Convert bboxes from albumentations to coco
            bboxes = convert_bboxes_from_albumentations(augmented_data['bboxes'],
                                                        target_format='coco',  # COCO: [x_min, y_min, width, height]
                                                        rows=img.shape[0],
                                                        cols=img.shape[1])

        # Convert image (PIL/Numpy) to PyTorch Tensor
        img = transforms.ToTensor()(img)
        _, h, w = img.shape

        # Convert bboxes to Tensor (YOLO format)
        bboxes = torch.from_numpy(np.array(bboxes))
        bboxes = coco2cxcywh(bboxes)  # ABS(x, y, w, h) => ABS(center_x, center_y, w, h)

        # Relative bboxes
        if not self.abs_coords:
            bboxes = abs2rel(bboxes, h, w)  # => REL(center_x, center_y, w, h)

        # Convert classes_id to Tensor
        classes_id = torch.from_numpy(np.array(classes_id))

        # Transform targets (bboxes)
        targets = torch.zeros((len(bboxes), 6))
        targets[:, 1] = classes_id
        targets[:, 2:] = bboxes

        #print("\timage shape: {};\ttargets: {}".format(img.shape, targets.shape))
        return image_path, img, targets

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))

        # Get targets as a list of tensors
        targets = [boxes for boxes in targets if boxes is not None]

        # Add index to each bbox (box_index, [x,y,w,h])
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i

        # Stack all boxes (fixed sized)
        targets = torch.cat(targets, dim=0)

        # Relative coordinates are a must for resizing the input
        if self.force_resize:
            if not self.abs_coords:
                # Selects new image size every X batches
                if self.multiscale and self.batch_count % 10 == 0:
                    # Random size, but multiple of the network stride (32)
                    self.img_size = random.choice(range(self.min_size, self.max_size + 1, self.network_stride))

                # Resize images to input shape
                imgs = [resize(img, self.img_size) for img in imgs]
            else:
                raise ValueError('We cannot resize images with absolute bounding boxes')

        # Images to Tensor
        imgs = torch.stack([img for img in imgs])

        self.batch_count += 1
        return paths, imgs, targets

    def __len__(self):
        return len(self.images)

