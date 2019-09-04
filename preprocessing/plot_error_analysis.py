import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set()
import torch
from utils.utils import load_dataset, load_obj
from sklearn.metrics import roc_curve, auc
from models.ssd.utils import find_jaccard_overlap
from utils.evaluate import *

base_path = "/home/salvacarrion/Documents/Programming/Python/Projects/yolo4math/models/yolov3"
predictions = load_obj(base_path + "/predictions.pkl")
confusion_matrix = load_obj(base_path + "/confusion_matrix.pkl")
stats = load_dataset(base_path + "/stats.json")


det_boxes = predictions['det_boxes']
det_labels = predictions['det_labels']
det_scores = predictions['det_scores']
true_boxes = predictions['true_boxes']
true_labels = predictions['true_labels']


# det_labels = ignore_bg(det_labels)
# true_labels = ignore_bg(true_labels)

pred_class, pred_score, pred_iou, true_class = match_classes(det_boxes, det_labels, det_scores, true_boxes, true_labels, n_classes=2)

# Set background at index 0
pred_class += 1
true_class += 1
analysis = compute_hoeim(pred_class, pred_iou, true_class)


labels = ['Correct', 'Background', 'Localization', 'Other']
sizes = [analysis[l.lower()] for l in labels]
explode = (0, 0.05, 0.10, 0.15)

fig, ax = plt.subplots()
w,l,p = ax.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', pctdistance=1.1, labeldistance=None,
               startangle=15, rotatelabels=True, textprops={'fontsize': 8}, counterclock=False)
ax.axis('equal')


plt.title('YOLOv3')
plt.legend(loc="lower right")
plt.savefig('error-analysis.eps')
plt.show()
asdsad = 3