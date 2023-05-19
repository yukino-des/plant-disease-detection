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
from tqdm import tqdm

from model import Yolo, YoloBody
from utils import avg_iou, get_classes, get_map, k_means, load_data, summary

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"],
                   allow_headers=["*"])


# 响应检测图像、model.onnx、model.pth、summary.txt
@app.get("/data/{file_path:path}")
async def data(file_path):
    filename = file_path.rsplit("/", 1)[-1]
    return FileResponse(f"data/{file_path}", filename=filename)


# 请求图像、视频
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
        return JSONResponse({"imageUrl": f"http://0.0.0.0:2475/{image_path}",
                             "imageOutUrl": f"http://0.0.0.0:2475/{image_out_path}",
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
    mode = input("Input d as directory, f as fps, k as k-means, m as map, o as onnx, s as summary, c as camera: ")
    # 目录检测
    if str.__contains__(str.lower(mode), "d"):
        yolo = Yolo()
        image_dir = input("Input directory path: ")
        image_names = os.listdir(image_dir)
        for image_name in tqdm(image_names):
            if not image_name.lower().endswith(
                    [".bmp", ".dib", ".png", ".jpg", ".jpeg", ".pbm", ".pgm", ".ppm", ".tif", ".tiff"]):
                continue
            image = yolo.detect_image(os.path.join(image_dir, image_name))
            os.makedirs("data/cache/image/out", exist_ok=True)
            image.save(f"data/cache/image/out/{image_name.rsplit('.', 1)[0]}.png", quality=95, subsampling=0)

    # fps检测
    if str.__contains__(str.lower(mode), "f"):
        yolo = Yolo()
        fps_image_path = input("Input image path: ")
        image = Image.open(fps_image_path)
        tact_time = yolo.get_fps(image, test_interval=100)
        print(str(tact_time) + " seconds; " + str(1 / tact_time) + " fps; @batch_size 1")

    # 得到data/cache/anchors.txt、data/cache/k-means.jpg
    if str.__contains__(str.lower(mode), "k"):
        np.random.seed(0)
        data = load_data()
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

    # 得到data/cache/map
    if str.__contains__(str.lower(mode), "m"):
        image_ids = open("data/VOC/ImageSets/Main/test.txt").read().strip().split()
        os.makedirs("data/cache/map/ground-truth", exist_ok=True)
        os.makedirs("data/cache/map/result", exist_ok=True)
        class_names, _ = get_classes()
        yolo = Yolo(confidence=0.001, nms_iou=0.5)
        for image_id in tqdm(image_ids):
            image = Image.open(f"data/VOC/JPEGImages/{image_id}.jpg")
            yolo.get_map_txt(image_id, image, class_names)
            with open(f"data/cache/map/ground-truth/{image_id}.txt", "w") as new_f:
                root = ElementTree.parse(f"data/VOC/Annotations/{image_id}.xml").getroot()
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

    # 得到data/cache/model.onnx
    if str.__contains__(str.lower(mode), "o"):
        yolo = Yolo()
        yolo.convert_to_onnx(simplify=False)

    # 得到data/cache/summary.txt
    if str.__contains__(str.lower(mode), "s"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        m = YoloBody(80).to(device)
        model_summary = summary(m, (3, 416, 416))
        dummy_input = torch.randn(1, 3, 416, 416).to(device)
        flops, params = profile(m.to(device), (dummy_input,), verbose=False)
        flops = flops * 2
        flops, params = clever_format([flops, params], "%.3f")
        model_summary += f"Total flops: {flops}\nTotal params: {params}\n{'-' * 95}"
        sum_txt = open("data/cache/summary.txt", "w")
        sum_txt.write(model_summary)
        sum_txt.close()
        print("data/cache/summary.txt saved.")

    # 调用摄像头
    if str.__contains__(str.lower(mode), "c"):
        yolo = Yolo()
        time_str = datetime.strftime(datetime.now(), '%Y%m%d%H%M%S')
        yolo.detect_video(0, time_str)

    # 启动后端服务器，监听2475号端口
    if mode == "":
        yolo = Yolo()
        # 清空训练缓存
        shutil.rmtree("data/cache/loss", ignore_errors=True)
        # 清空图像缓存
        shutil.rmtree("data/cache/image", ignore_errors=True)
        os.makedirs("data/cache/image/out", exist_ok=True)
        # 清空视频缓存
        shutil.rmtree("data/cache/video", ignore_errors=True)
        os.makedirs("data/cache/video/out", exist_ok=True)
        #局域网访问
        uvicorn.run(app, host="0.0.0.0", port=2475, workers=0)
