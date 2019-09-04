import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set()
import torch
from utils.utils import load_dataset, load_obj


json_path = "/Users/salvacarrion/Documents/Experiments/tfm/YOLOv3-1024-noweights-6clusters/YOLO_stats.json"
# json_path = "/Users/salvacarrion/Documents/Experiments/tfm/SSD-1024-1024-adam-alpha2.0/SSD_stats.json"
raw_data = load_dataset(json_path)


data_embedded = {
    'recall': raw_data['0']['recall11'],
    'precision': raw_data['0']['precision11'],
}
data_isolated = {
    'recall': raw_data['1']['recall11'],
    'precision': raw_data['1']['precision11'],
}

# Plots
ax2 = sns.lineplot(x='conf', y='precision', data=pd.DataFrame(data), markers=True,  label='Precision')

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
plt.legend()

# Show and save
plt.savefig('1024-rp.eps')
plt.show()
asdas = 3
