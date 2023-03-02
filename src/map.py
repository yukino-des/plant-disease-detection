import os

from PIL import Image
import sys
from tqdm import tqdm
from xml.etree import ElementTree as ET

sys.path.append(os.path.dirname(sys.path[0]))
from utils.util import get_classes, get_map
from utils.yolo import YOLO

if __name__ == "__main__":
    min_overlap = 0.5
    confidence = 0.001
    nms_iou = 0.5
    score_threshold = 0.5
    image_ids = open("../VOC/ImageSets/Main/test.txt").read().strip().split()
    os.makedirs("../tmp/maps_out/.gt", exist_ok=True)
    os.makedirs("../tmp/maps_out/.dr", exist_ok=True)
    class_names, _ = get_classes("../data/classes.txt")
    yolo = YOLO(confidence=confidence, nms_iou=nms_iou)
    for image_id in tqdm(image_ids):
        image_path = f"../VOC/JPEGImages/{image_id}.jpg"
        image = Image.open(image_path)
        yolo.get_map_txt(image_id, image, class_names, "../tmp/maps_out")
        with open(f"../tmp/maps_out/.gt/{image_id}.txt", "w") as new_f:
            root = ET.parse(f"../VOC/Annotations/{image_id}.xml").getroot()
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
                    new_f.write(f"{obj_name} {left} {top} {right} {bottom} difficult\n")
                else:
                    new_f.write(f"{obj_name} {left} {top} {right} {bottom}\n")
    get_map(min_overlap, True, score_threshold=score_threshold, path="../tmp/maps_out")
