import time
import torch
import numpy as np
import json
import cv2
from utils.utils import *
import random
from collections import defaultdict


def main():
    WIDTH, HEIGHT = (1024, 1024)
    COLORS = np.array([[200, 0, 0, 255], [0, 0, 200, 255]])

    load_path = 'datasets/equations/resized/{}x{}-padded'.format(WIDTH, HEIGHT)
    json_dataset = load_dataset(load_path)
    images = json_dataset['images']
    annotations = json_dataset['annotations']
    class_names = json_dataset['categories']

    print('Loading images from: {}'.format(load_path))
    print('-----------------------------\n')

    # Get data
    for i, image_data in enumerate(images, 1):
        image_id = str(image_data['id'])
        bboxes = annotations[image_id]  # not boxes

        # Build paths
        filename = load_path + '/' + image_data['filename']

        print("Loading image  #{} ({}/{})...".format(image_id, i, len(images)))
        image = cv2.imread(filename)

        # print("\t- Performing bbox augmentation...")
        # bboxes = augment_bboxes(bboxes)
        #
        # print("\t- Performing non-maximum supression...")
        # bboxes, total_boxes1, total_boxes2 = non_maximum_supression(bboxes)
        # print("\t\t- Boxes supressed: {} ({:.2f}%)".format(total_boxes1 - total_boxes2, (total_boxes1 - total_boxes2) / total_boxes1 * 100))

        # Draw bounding boxes
        print("\t- Drawing bboxes...")
        image = Image.fromarray(image)
        for annotation in bboxes:
            cat_id = str(annotation['category_id'])
            text = class_names[cat_id]
            draw_bbox(image, annotation['bbox'], cat_id, COLORS, thickness=2)
            draw_labels(image, annotation['bbox'], text, cat_id, COLORS, font_size=12)

        # Show image
        f, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.imshow(image)
        f.tight_layout()
        #f.savefig('output/image_{}.eps'.format(i))  # Doesn't work properly
        plt.show()

        # Finish loop
        if i == 10:
            break


if __name__ == "__main__":
    pass
    main()
