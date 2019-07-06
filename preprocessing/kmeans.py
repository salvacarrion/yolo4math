import numpy as np
from utils.utils import *


class YOLO_Kmeans:

    def __init__(self, cluster_number):
        self.cluster_number = cluster_number

    def iou(self, boxes, clusters):  # 1 box -> k clusters
        n = boxes.shape[0]
        k = self.cluster_number

        box_area = boxes[:, 0] * boxes[:, 1]
        box_area = box_area.repeat(k)
        box_area = np.reshape(box_area, (n, k))

        cluster_area = clusters[:, 0] * clusters[:, 1]
        cluster_area = np.tile(cluster_area, [1, n])
        cluster_area = np.reshape(cluster_area, (n, k))

        box_w_matrix = np.reshape(boxes[:, 0].repeat(k), (n, k))
        cluster_w_matrix = np.reshape(np.tile(clusters[:, 0], (1, n)), (n, k))
        min_w_matrix = np.minimum(cluster_w_matrix, box_w_matrix)

        box_h_matrix = np.reshape(boxes[:, 1].repeat(k), (n, k))
        cluster_h_matrix = np.reshape(np.tile(clusters[:, 1], (1, n)), (n, k))
        min_h_matrix = np.minimum(cluster_h_matrix, box_h_matrix)
        inter_area = np.multiply(min_w_matrix, min_h_matrix)

        result = inter_area / (box_area + cluster_area - inter_area)
        return result

    def avg_iou(self, boxes, clusters):
        iou = self.iou(boxes, clusters)
        accuracy = np.mean([np.max(iou, axis=1)])
        return accuracy

    def kmeans(self, boxes, k, dist=np.median):
        box_number = boxes.shape[0]
        distances = np.empty((box_number, k))
        last_nearest = np.zeros((box_number,))
        np.random.seed()
        clusters = boxes[np.random.choice(
            box_number, k, replace=False)]  # init k clusters
        while True:

            distances = 1 - self.iou(boxes, clusters)

            current_nearest = np.argmin(distances, axis=1)
            if (last_nearest == current_nearest).all():
                break  # clusters won't change
            for cluster in range(k):
                clusters[cluster] = dist(  # update clusters
                    boxes[current_nearest == cluster], axis=0)

            last_nearest = current_nearest

        return clusters

    def result2txt(self, data, avg_iou, filename):
        f = open(filename, 'w')
        row = np.shape(data)[0]
        for i in range(row):
            if i == 0:
                x_y = "%d,%d" % (data[i][0], data[i][1])
            else:
                x_y = "\n%d,%d" % (data[i][0], data[i][1])
            f.write(x_y)

        f.write('\n\n--------------------')
        f.write('\nk={}'.format(self.cluster_number))
        f.write('\navg_iou={}'.format(avg_iou))
        f.close()

    def txt2boxes(self, filename):
        f = open(filename, 'r')
        dataSet = []
        for line in f:
            x1, y1, x2, y2 = [int(x) for x in line.strip().split(',')]
            dataSet.append([int(x2-x1), int(y2-y1)])
        result = np.array(dataSet)
        f.close()
        return result

    def txt2clusters(self):
        all_boxes = self.txt2boxes()
        result = self.kmeans(all_boxes, k=self.cluster_number)
        result = result[np.lexsort(result.T[0, None])]
        avg_iou = self.avg_iou(all_boxes, result)
        print("K anchors:\n {}".format(result))
        print("Accuracy: {:.2f}%".format(avg_iou * 100))
        self.result2txt(result, avg_iou)
        return avg_iou

    def get_clusters(self, box_sizes):
        all_boxes = np.array(box_sizes)
        result = self.kmeans(all_boxes, k=self.cluster_number)
        result = result[np.lexsort(result.T[0, None])]
        avg_iou = self.avg_iou(all_boxes, result)
        print("K anchors:\n {}".format(result))
        print("Accuracy: {:.2f}%".format(avg_iou * 100))
        return result, avg_iou


def get_boxes(json):
    # Get sizes
    box_sizes = []
    for k, v in json['annotations'].items():
        for annotations in v:
            x, y, w, h = annotations['bbox']
            box_sizes.append([w, h])
    return box_sizes


if __name__ == "__main__":
    # Settings
    WIDTH, HEIGHT = (1024, 1024)
    K = 10
    subfolder = "{}x{}".format(WIDTH, HEIGHT)
    save_path = 'anchors/' + subfolder
    load_path = 'datasets/equations/resized/' + subfolder

    # Get box sizes
    JSON_DATASET = load_dataset(load_path + '/train.json')
    box_sizes = get_boxes(JSON_DATASET)

    # Make dir if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    cluster_ious = {}
    for i in range(1, K+1):
        print('Clustering with k={}...'.format(i))
        cluster_number = i
        kmeans = YOLO_Kmeans(cluster_number)

        # Compute clusters
        result, avg_iou = kmeans.get_clusters(box_sizes)
        cluster_ious[i] = avg_iou

        # Save file
        kmeans.result2txt(result, avg_iou, save_path + "/anchors_c{}.txt".format(kmeans.cluster_number))

    # Save json
    with open(save_path + "/cluster_ious.json".format(subfolder), 'w') as f:
        json.dump(cluster_ious, f)


