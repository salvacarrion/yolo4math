# import time
# import torch
# import numpy as np
# import json
# import cv2
# from utils.utils import *
# import random
# from collections import defaultdict
#
#
# if __name__ == "__main__":
#     start_time = time.time()
#     WIDTH, HEIGHT = (1024, 1024)
#     SAVE_AS_TEXT = True
#     NORMALIZE_BBOXES = True
#
#     load_path = "/home/salvacarrion/Documents/datasets/equations/raw"
#     labels_path = "/home/salvacarrion/Documents/datasets/equations/labels"
#
#     print('Loading images from: {}'.format(load_path))
#     print('Saving labels to: {}'.format(labels_path))
#     print('-----------------------------\n')
#
#     # Get data
#     JSON_DATASET = load_dataset(load_path + '/train.json')
#     images = JSON_DATASET['images']
#
#     for i, image_data in enumerate(images, 1):
#         print("Processing label {}/{}".format(i, len(images)))
#         image_id = str(image_data['id'])
#         bboxes = JSON_DATASET['annotations'][image_id]
#         img_filename = image_data['filename']
#         label_filename = img_filename.replace('jpg', 'txt')
#
#         # Open image and get shape
#         img = Image.open(load_path + '/' + img_filename)
#         img_w, img_h = img.size
#
#         target_i = []
#         for bbox in bboxes:
#             class_id = bbox['category_id']
#             x1, y1, w, h = bbox['bbox']  # Abs
#             x2, y2 = x1+w, y1+h
#
#             # # Compute center [cxcywh]
#             # cx_abs = x_abs + w_abs/2
#             # cy_abs = y_abs + h_abs/2
#             # x_abs, y_abs = cx_abs, cy_abs  # Alias
#
#             # Convert to relative
#             if NORMALIZE_BBOXES:
#                 x1, y1, x2, y2 = x1/img_w, y1/img_h,  x2/img_w, y2/img_h
#                 w, h = w/img_w, h/img_h
#
#             # Append result
#             target_i.append([class_id, x1, y1, w, h])
#
#         # Save bboxes
#         target_i = np.array(target_i, dtype=np.float32)
#         if SAVE_AS_TEXT:
#             np.savetxt(labels_path + '/' + label_filename, target_i, fmt='%.6f', header="class_id x y w h")
#         else:
#             np.save(labels_path + '/' + label_filename, target_i)
#
#     print("Done!")
