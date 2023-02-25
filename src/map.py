import os
from PIL import Image
import sys
from tqdm import tqdm
from xml.etree import ElementTree as ET

sys.path.append(os.path.dirname(sys.path[0]))
from utils.util import get_classes, get_map
from utils.yolo import YOLO

if __name__ == "__main__":
    classes_path = "../data/classes.txt"
    min_overlap = 0.5
    confidence = 0.001
    nms_iou = 0.5
    score_threshold = 0.5
    maps_out_path = "../tmp/maps_out"
    image_ids = open("../VOC/ImageSets/Main/test.txt").read().strip().split()
    if not os.path.exists(maps_out_path):
        os.makedirs(maps_out_path)
    if not os.path.exists(os.path.join(maps_out_path, "ground-truth")):
        os.makedirs(os.path.join(maps_out_path, "ground-truth"))
    if not os.path.exists(os.path.join(maps_out_path, "detection")):
        os.makedirs(os.path.join(maps_out_path, "detection"))
    class_names, _ = get_classes(classes_path)
    yolo = YOLO(confidence=confidence, nms_iou=nms_iou)
    for image_id in tqdm(image_ids):
        image_path = "../VOC/JPEGImages/" + image_id + ".jpg"
        image = Image.open(image_path)
        yolo.get_map_txt(image_id, image, class_names, maps_out_path)
        with open(os.path.join(maps_out_path, "ground-truth/" + image_id + ".txt"), "w") as new_f:
            root = ET.parse("../VOC/Annotations/" + image_id + ".xml").getroot()
            for obj in root.findall("object"):
                difficult_flag = False
                if obj.find("difficult") is not None:
                    difficult = obj.find("difficult").text
                    if int(difficult) == 1:
                        difficult_flag = True
                obj_name = obj.find("name").text
                if obj_name not in class_names:
                    continue
                bndbox = obj.find("bndbox")
                left = bndbox.find("xmin").text
                top = bndbox.find("ymin").text
                right = bndbox.find("xmax").text
                bottom = bndbox.find("ymax").text
                if difficult_flag:
                    new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
                else:
                    new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
    get_map(min_overlap, True, score_threshold=score_threshold, path=maps_out_path)
