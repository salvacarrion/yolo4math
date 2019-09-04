import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set()
import torch
from utils.utils import load_dataset, load_obj

base_path = "/home/salvacarrion/Documents/Programming/Python/Projects/yolo4math/models/yolov3"
predictions = load_obj(base_path + "/predictions.pkl")
confusion_matrix = load_obj(base_path + "/confusion_matrix.pkl")
stats = load_dataset(base_path + "/stats.json")


data_embedded = {
    'recall': stats['classes']['0']['recall11'],
    'precision': stats['classes']['0']['precision11'],
}
data_isolated = {
    'recall': stats['classes']['1']['recall11'],
    'precision': stats['classes']['1']['precision11'],
}

# Plots
# ax2 = sns.lineplot(x='conf', y='precision', data=pd.DataFrame(data_embedded), markers=True,  label='Precision')

# Settings
plt.figure()
ax1 = sns.lineplot(x='recall', y='precision', data=pd.DataFrame(data_embedded), markers=True, label='Embedded', drawstyle='steps-pre')
ax1 = sns.lineplot(x='recall', y='precision', data=pd.DataFrame(data_isolated), markers=True, label='Isolated', drawstyle='steps-pre')
# ax1 = sns.lineplot(data=pd.DataFrame(data, index='recall'), sort=False, legend="full", markers=True, )
# plt.title('ROC curve')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Recall-Precision curve')
# plt.xlabel('Recall')
# plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])

plt.title('YOLO Recall-Precision curve')
plt.legend(loc="lower right")
plt.savefig('pr-curve-yolo.eps')
plt.show()
asdas = 3
