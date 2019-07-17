import time
import torch
import numpy as np
import json
import cv2
from utils.utils import *
import random
from collections import defaultdict


# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.array([[200, 0, 0, 255], [0, 0, 200, 255]])#np.random.randint(0, 255, size=(len(CATEGORIES), 3), dtype="uint8")

#CATEGORIES = {c['id']: c['name'] for c in JSON_DATASET['categories']}
# ANNOTATIONS = defaultdict(list)
#
# # Get annotations per image
# for annotation in JSON_DATASET['annotations']:
#     image_id = annotation['image_id']
#     data = {}
#     for k, v in annotation.items():
#         if k != 'image_id':
#             data[k] = v
#     ANNOTATIONS[image_id].append(data)
#
#
# NEW_JSON = {'categories': CATEGORIES,
#             'annotations': dict(ANNOTATIONS),
#             'images': JSON_DATASET['images']}
# save_dataset(NEW_JSON, 'new_train.json')
#
# asw = 3

def main():
    start_time = time.time()
    WIDTH, HEIGHT = (1440, 1440)

    load_path_raw = '/home/salvacarrion/Documents/datasets/equations/raw'

    load_path = load_path_raw  # load_path_resized
    save_path = "/home/salvacarrion/Documents/datasets/equations/{}".format(HEIGHT)
    JSON_DATASET = load_dataset(load_path + '/train.json')

    # Make dir if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print('Loading images from: {}'.format(load_path))
    print('Save images to: {}'.format(save_path))
    print('-----------------------------\n')

    total_w = 0
    total_h = 0
    min_w = 10000000000
    min_h = 10000000000
    max_w = 0
    max_h = 0

    # Get data
    for i, image_data in enumerate(JSON_DATASET['images'], 1):
        image_id = str(image_data['id'])
        bboxes = JSON_DATASET['annotations'][image_id]

        # Build paths
        filename = load_path + '/' + image_data['filename']
        save_filename = save_path + '/' + image_data['filename']

        print("Loading image  #{} ({}/{})...".format(image_id, i, len(JSON_DATASET['images'])))
        image = cv2.imread(filename)

        print("\t- Resizing image...")
        image, new_coords = letterbox_image(image, (WIDTH, HEIGHT), padding=False)

        # Stats
        h, w, _ = image.shape
        total_w += w
        total_h += h
        min_w = min(min_w, w)
        min_h = min(min_h, h)
        max_w = max(max_w, w)
        max_h = max(max_h, h)

        print("\t- Resizing bounding boxes...")
        bboxes = resize_bboxes(bboxes, new_coords)
        JSON_DATASET['annotations'][image_id] = bboxes

        # # print("\t- Performing bbox augmentation...")
        # # bboxes = augment_bboxes(bboxes)
        # #
        # # print("\t- Performing non-maximum supression...")
        # # bboxes, total_boxes1, total_boxes2 = non_maximum_supression(bboxes)
        # # print("\t\t- Boxes supressed: {} ({:.2f}%)".format(total_boxes1 - total_boxes2, (total_boxes1 - total_boxes2) / total_boxes1 * 100))
        #
        # # Draw bounding boxes
        # print("\t- Drawing bboxes...")
        # image = Image.fromarray(image)
        # for annotation in bboxes:
        #     cat_id = str(annotation['category_id'])
        #     text = JSON_DATASET['categories'][cat_id]
        #     draw_bbox(image, annotation['bbox'], cat_id, COLORS, thickness=2)
        #     draw_labels(image, annotation['bbox'], text, cat_id, COLORS, font_size=24)

        # # Crop boxes
        # #crop_images(filename, bboxes, 'datasets/equations/crops', image_id, CATEGORIES)
        #
        # Show image
        #image.show()

        print('\t- Saving image...')
        save_image(image, save_filename)

        # Finish loop
        # if i == 10:
        #     break

    # # Fix json
    # categories = {int(k): v for k, v in JSON_DATASET['categories'].items()}
    # images = []
    # for x in JSON_DATASET['images']:
    #     images.append({'id': int(x['id']), 'filename': x['file_name']})
    # annotations = {int(k): v for k, v in JSON_DATASET['annotations'].items()}
    # NEW_JSON_DATASET = {'categories': categories,
    #                     'annotations': annotations,
    #                     'images': images}
    # Save new json
    print('Saving JSON dataset...')
    save_dataset(JSON_DATASET, save_path + '/train.json')

    print('\n\nSUMMARY:')
    print('-----------------------------')

    avg_w = total_w/len(JSON_DATASET['images'])
    avg_h = total_h/len(JSON_DATASET['images'])

    print('avg_w: ', avg_w)
    print('avg_h: ', avg_h)
    print('min_w: ', min_w)
    print('min_h: ', min_h)
    print('max_w: ', max_w)
    print('max_h: ', max_h)

    # Compute elapsed time
    hours, rem = divmod(time.time() - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("\nTotal time: {:0>2}:{:0>2}:{:05.3f}".format(int(hours), int(minutes), seconds))
    print('Done!')


if __name__ == "__main__":
    pass
    main()
