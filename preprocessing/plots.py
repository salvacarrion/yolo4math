from utils.utils import *

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()



def plot_clusters():
    # Load cluster IOUs
    WIDTH, HEIGHT = (1024, 1440)
    base_path = '/home/salvacarrion/Documents/Programming/Python/Projects/yolo4math/anchors/{}'.format(HEIGHT)
    data = load_dataset(base_path + '/cluster_ious.json')

    x = []
    y = []
    y2 = []
    last_val = 0.0
    for i, t in enumerate(data.items()):
        k, v = t
        x.append(k)
        y.append(v)
        y2.append(v - last_val)
        last_val = v

    df = pd.DataFrame({'Avg. IOU': y, 'Gain': y2}, index=x)

    plt.figure()
    sns.lineplot(data=df, sort=False, legend="full", markers=True)
    plt.xlabel('Clusters')
    plt.ylabel('Avg. IOU')
    plt.title('Clustering box dimensions')
    plt.savefig(base_path + '/cluters.eps')
    plt.show()



if __name__ == '__main__':
    plot_clusters()
