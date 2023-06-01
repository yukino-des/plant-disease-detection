import glob
import os
import shutil
from datetime import datetime
from xml.etree import ElementTree

import numpy as np
import torch
import uvicorn
from PIL import Image
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from matplotlib import pyplot as plt
from starlette.responses import FileResponse, JSONResponse
from thop import clever_format, profile
from torchsummary import summary
from tqdm import tqdm

from model import Yolo, YoloBody, MobileNetV2
from utils import avg_iou, get_classes, get_map, get_txt, k_means

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"],
                   allow_headers=["*"])


@app.get("/data/{file_path:path}")
async def data(file_path):
    filename = file_path.rsplit("/", 1)[-1]
    return FileResponse(f"data/{file_path}", filename=filename)


@app.post("/file")
async def image(file: UploadFile):
    if file is None:
        return JSONResponse({}, 404)
    file_name, extend_name = file.filename.rsplit(".", 1)
    if extend_name.lower() in ["jpeg", "jpg", "png"]:
        image_path = f"data/cache/image/{file.filename}"
        image_out_path = f"data/cache/image/out/{file_name}.png"
        with open(image_path, "wb+") as wb:
            shutil.copyfileobj(file.file, wb)
        target_info = yolo.detect_image(image_path)
        return JSONResponse({"imageUrl": f"http://0.0.0.0:2023/{image_path}",
                             "imageOutUrl": f"http://0.0.0.0:2023/{image_out_path}",
                             "targetInfo": target_info}, 200)
    elif extend_name.lower() in ["mp4", "mov", "avi"]:
        video_path = f"data/cache/video/{file.filename}"
        with open(video_path, "wb+") as wb:
            shutil.copyfileobj(file.file, wb)
        _time_str = datetime.strftime(datetime.now(), '%Y%m%d%H%M%S')
        yolo.detect_video(video_path, _time_str)
        os.rename(f"data/cache/video/out/{_time_str}.avi", f"data/cache/video/out/{file_name}.avi")
        return JSONResponse({"videoPath": f"data/cache/video/out/{file_name}.avi"}, 200)
    else:
        return JSONResponse({}, 404)


if __name__ == "__main__":
    mode = input("Input mode(app, directory, fps, k-means, map, onnx, backbone, summary, camera): ")

    if mode == "app":
        yolo = Yolo()
        shutil.rmtree("data/cache/loss", ignore_errors=True)
        shutil.rmtree("data/cache/image", ignore_errors=True)
        os.makedirs("data/cache/image/out", exist_ok=True)
        shutil.rmtree("data/cache/video", ignore_errors=True)
        os.makedirs("data/cache/video/out", exist_ok=True)
        uvicorn.run(app, host="0.0.0.0", port=2023, workers=0)

    elif mode == "camera":
        yolo = Yolo()
        time_str = datetime.strftime(datetime.now(), '%Y%m%d%H%M%S')
        yolo.detect_video(0, time_str)

    elif mode == "directory":
        yolo = Yolo()
        image_dir = input("Input directory path: ")
        image_names = os.listdir(image_dir)
        for image_name in tqdm(image_names):
            if not image_name.lower().endswith(["jpeg", "jpg", "png"]):
                continue
            image = yolo.detect_image(os.path.join(image_dir, image_name))
            os.makedirs("data/cache/image/out", exist_ok=True)
            image.save(f"data/cache/image/out/{image_name.rsplit('.', 1)[0]}.png", quality=95, subsampling=0)

    elif mode == "fps":
        yolo = Yolo()
        fps_image_path = input("Input image path: ")
        image = Image.open(fps_image_path)
        tact_time = yolo.get_fps(image, test_interval=100)
        print(str(tact_time) + " seconds; " + str(1 / tact_time) + " fps; @batch_size 1")

    elif mode == "k-means":
        np.random.seed(0)
        data = []
        for xml_file in tqdm(glob.glob("data/VOCdevkit/VOC2007/Annotations/*xml")):
            tree = ElementTree.parse(xml_file)
            height = int(tree.findtext("./size/height"))
            width = int(tree.findtext("./size/width"))
            if height <= 0 or width <= 0:
                continue
            for obj in tree.iter("object"):
                x_min = np.float64(int(float(obj.findtext("bndbox/xmin"))) / width)
                y_min = np.float64(int(float(obj.findtext("bndbox/ymin"))) / height)
                x_max = np.float64(int(float(obj.findtext("bndbox/xmax"))) / width)
                y_max = np.float64(int(float(obj.findtext("bndbox/ymax"))) / height)
                data.append([x_max - x_min, y_max - y_min])
        data = np.array(data)
        cluster, near = k_means(data, 9)
        data = data * np.array([416, 416])
        cluster = cluster * np.array([416, 416])
        for j in range(9):
            plt.scatter(data[near == j][:, 0], data[near == j][:, 1])
            plt.scatter(cluster[j][0], cluster[j][1], marker="x", c="black")
        os.makedirs("data/cache", exist_ok=True)
        plt.savefig("data/cache/k-means.jpg")
        print("data/cache/k-means.jpg saved.")
        cluster = cluster[np.argsort(cluster[:, 0] * cluster[:, 1])]
        print("avg_ratio: %.2f" % (avg_iou(data, cluster)))
        print(f"cluster:\n{cluster}")
        f = open("data/cache/anchors.txt", "w")
        row = np.shape(cluster)[0]
        for i in range(row):
            if i == 0:
                xy = "%d,%d" % (cluster[i][0], cluster[i][1])
            else:
                xy = ", %d,%d" % (cluster[i][0], cluster[i][1])
            f.write(xy)
        f.close()

    elif mode == "map":
        get_txt(0, 0.9, 0.9)
        image_ids = open("data/VOCdevkit/VOC2007/ImageSets/Main/test.txt").read().strip().split()
        os.makedirs("data/cache/map/ground-truth", exist_ok=True)
        os.makedirs("data/cache/map/result", exist_ok=True)
        class_names, _ = get_classes("data/classes.txt")
        yolo = Yolo(confidence=0.001, nms_iou=0.5)
        for image_id in tqdm(image_ids):
            image = Image.open(f"data/VOCdevkit/VOC2007/JPEGImages/{image_id}.jpg")
            yolo.get_map_txt(image_id, image, class_names)
            with open(f"data/cache/map/ground-truth/{image_id}.txt", "w") as new_f:
                root = ElementTree.parse(f"data/VOCdevkit/VOC2007/Annotations/{image_id}.xml").getroot()
                for obj in root.findall("object"):
                    difficult_flag = False
                    if obj.find("difficult") is not None:
                        difficult = obj.find("difficult").text
                        if int(difficult) == 1:
                            difficult_flag = True
                    obj_name = obj.find("name").text
                    if obj_name not in class_names:
                        continue
                    bnd_box = obj.find("bndbox")
                    left = bnd_box.find("xmin").text
                    top = bnd_box.find("ymin").text
                    right = bnd_box.find("xmax").text
                    bottom = bnd_box.find("ymax").text
                    if difficult_flag:
                        new_f.write(f"{obj_name} {left} {top} {right} {bottom} difficult\n")
                    else:
                        new_f.write(f"{obj_name} {left} {top} {right} {bottom}\n")
        get_map(0.5, 0.5)

    elif mode == "onnx":
        yolo = Yolo()
        yolo.convert_to_onnx(simplify=False)

    elif mode == "backbone":
        model = MobileNetV2()
        summary(model, (3, 416, 416))

    elif mode == "summary":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        m = YoloBody(27).to(device)
        summary(m, (3, 416, 416))
        dummy_input = torch.randn(1, 3, 416, 416).to(device)
        flops, params = profile(m.to(device), (dummy_input,), verbose=False)
        flops = flops * 2
        flops, params = clever_format([flops, params], "%.3f")
        print(f"Total flops: {flops}\nTotal params: {params}\n{'-' * 95}")

    else:
        print("mode error")
