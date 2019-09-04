import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set()
import torch
from utils.utils import load_dataset, load_obj
from sklearn.metrics import roc_curve, auc
from models.ssd.utils import find_jaccard_overlap


base_path = "/Users/salvacarrion/Documents/Programming/Python/yolo4math/models/ssd"
predictions = load_obj(base_path + "/predictions.pkl")
#confusion_matrix = load_obj(base_path + "/confusion_matrix.pkl")
stats = load_dataset(base_path + "/stats.json")


det_boxes = predictions['det_boxes']
det_labels = predictions['det_labels']
det_scores = predictions['det_scores']
true_boxes = predictions['true_boxes']
true_labels = predictions['true_labels']

d = {
    0: {'y': [], 'y_scores': []},
    1: {'y': [], 'y_scores': []},
}
n_classes = 2
for i, (boxes, labels, scores, t_boxes, true_labels) in enumerate(zip(det_boxes, det_labels, det_scores, true_boxes, true_labels), 1):
    print("{}/{}".format(i, len(det_boxes)))

    assert boxes.size(0) == labels.size(0) == scores.size(0)
    assert t_boxes.size(0) == true_labels.size(0)
    for c in range(n_classes):
        try:
            class_boxes = boxes[labels == c]
            class_scores = scores[labels == c]
            class_true_labels = true_labels
            class_true_boxes = t_boxes

            # Compute overlap (>0.5 -> detected)
            overlaps = find_jaccard_overlap(class_boxes, class_true_boxes)  # (1, n_class_objects_in_img)
            if overlaps.size(0) == 0:
                continue
            max_overlap, ind = torch.max(overlaps, dim=1)  # (), () - scalars

            # # Keep only matches
            # idxs_match = max_overlap > 0.5  # det indices
            # max_overlap = max_overlap[idxs_match]
            # ind = ind[idxs_match]  # true indices

            t_labels = class_true_labels[ind]
            d_scores = class_scores#[idxs_match]

            assert class_true_labels.size(0) == class_scores.size(0)
            d[c]['y'].append(class_true_labels)
            d[c]['y_scores'].append(class_scores)
            asasd = 33
        except Exception as e:
            print(e)

#y = [torch.cat(d['y']), torch.cat(d['y'])]
#y_scores = [torch.cat(d['y_scores']), torch.cat(d['y_scores'])]
#y = torch.cat(y).data.numpy()
#y_scores = torch.cat(y_scores).data.numpy()
#
#fpr, tpr, _ = roc_curve(y, y_scores)
#roc_auc = auc(fpr, tpr)


# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    y = torch.cat(d[i]['y']).data.numpy()
    y_scores = torch.cat(d[i]['y_scores']).data.numpy()
    fpr[i], tpr[i], _ = roc_curve(y, y_scores)
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curve
plt.figure()
for i, cls in enumerate(['Embedded', 'Isolated']):
    plt.plot(fpr[i], tpr[i], label="ROC curve for '{0}' (area = {1:0.2f})"
                                   ''.format(cls, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend(loc="lower right")
plt.show()

asdasd =3
