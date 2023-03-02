import glob
import matplotlib
import numpy as np
import os
from matplotlib import pyplot as plt
from tqdm import tqdm
from xml.etree import ElementTree as ET

if os.name == "nt":
    matplotlib.use("Agg")
else:
    matplotlib.use("TkAgg")


def cas_iou(box, cluster):
    x = np.minimum(cluster[:, 0], box[0])
    y = np.minimum(cluster[:, 1], box[1])
    intersection = x * y
    area1 = box[0] * box[1]
    area2 = cluster[:, 0] * cluster[:, 1]
    return intersection / (area1 + area2 - intersection)


def avg_iou(box, cluster):
    return np.mean([np.max(cas_iou(box[i], cluster)) for i in range(box.shape[0])])


def kmeans(box, k):
    row = box.shape[0]
    distance = np.empty((row, k))
    last_clu = np.zeros((row,))
    np.random.seed()
    cluster = box[np.random.choice(row, k, replace=False)]
    iternum = 0
    while True:
        for i in range(row):
            distance[i] = 1 - cas_iou(box[i], cluster)
        near = np.argmin(distance, axis=1)
        if (last_clu == near).all():
            break
        for j in range(k):
            cluster[j] = np.median(box[near == j], axis=0)
        last_clu = near
        if iternum % 5 == 0:
            print("iter: {:d}; avg_iou: {:.2f}".format(iternum, avg_iou(box, cluster)))
        iternum += 1
    return cluster, near


def load_data(path):
    data = []
    for xml_file in tqdm(glob.glob(f"{path}/*xml")):
        tree = ET.parse(xml_file)
        height = int(tree.findtext("./size/height"))
        width = int(tree.findtext("./size/width"))
        if height <= 0 or width <= 0:
            continue
        for obj in tree.iter("object"):
            xmin = np.float64(int(float(obj.findtext("bndbox/xmin"))) / width)
            ymin = np.float64(int(float(obj.findtext("bndbox/ymin"))) / height)
            xmax = np.float64(int(float(obj.findtext("bndbox/xmax"))) / width)
            ymax = np.float64(int(float(obj.findtext("bndbox/ymax"))) / height)
            data.append([xmax - xmin, ymax - ymin])
    return np.array(data)


if __name__ == "__main__":
    np.random.seed(0)
    input_shape = [416, 416]
    anchors_num = 9
    data = load_data("../VOC/Annotations")
    cluster, near = kmeans(data, anchors_num)
    data = data * np.array([input_shape[1], input_shape[0]])
    cluster = cluster * np.array([input_shape[1], input_shape[0]])
    for j in range(anchors_num):
        plt.scatter(data[near == j][:, 0], data[near == j][:, 1])
        plt.scatter(cluster[j][0], cluster[j][1], marker="x", c="black")
    os.makedirs("../tmp", exist_ok=True)
    plt.savefig("../tmp/kmeans.jpg")
    print("../tmp/kmeans.jpg saved.")
    cluster = cluster[np.argsort(cluster[:, 0] * cluster[:, 1])]
    print("avg_ratio: {:.2f}".format(avg_iou(data, cluster)))
    print(f"cluster:\n{cluster}")
    f = open("../tmp/anchors.txt", "w")
    row = np.shape(cluster)[0]
    for i in range(row):
        if i == 0:
            xy = "%d,%d" % (cluster[i][0], cluster[i][1])
        else:
            xy = ", %d,%d" % (cluster[i][0], cluster[i][1])
        f.write(xy)
    f.close()
