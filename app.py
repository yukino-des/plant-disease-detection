import cv2
import numpy as np
import os
import shutil
import time
import torch
import uvicorn
from datetime import datetime
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from matplotlib import pyplot as plt
from model import Yolo, YoloBody
from PIL import Image
from starlette.responses import FileResponse
from thop import clever_format, profile
from tqdm import tqdm
from utils import avg_iou, get_classes, get_map, k_means, load_data, summary
from xml.etree import ElementTree

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["GET", "POST"],
                   allow_headers=["Content-Type", "X-Requested-With"])


# 响应检测图像、model.onnx、model.pth、summary.txt
@app.get("/data/{file_path:path}", response_class=FileResponse)
def data(file_path):
    try:
        filename = file_path.rsplit("/", 1)[1]
    except IndexError:
        filename = file_path
    return FileResponse(f"data/{file_path}", filename=filename, headers={"Content-Type": "multipart/form-data"})


# 请求图像
@app.post("/image", response_model=dict)
def image(file: UploadFile):
    if file is None:
        return {"status": 404}
    file_name, extend_name = file.filename.rsplit(".", 1)
    _image_path = f"data/cache/image/{file.filename}"
    image_out_path = f"data/cache/image/out/{file_name}.png"
    image_heatmap_path = f"data/cache/image/heatmap/{file_name}.png"
    with open(_image_path, "wb+") as wb:
        shutil.copyfileobj(file.file, wb)
    if extend_name.lower() not in ["bmp", "dib", "jpeg", "jpg", "pbm", "pgm", "png", "ppm", "tif", "tiff"]:
        return {"status": 404}
    target_info = yolo.detect_image(_image_path)
    yolo.detect_heatmap(_image_path)
    return {"status": "ok",
            "imageUrl": f"http://0.0.0.0:8080/{_image_path}",
            "imageOutUrl": f"http://0.0.0.0:8080/{image_out_path}",
            "imageHeatmapUrl": f"http://0.0.0.0:8080/{image_heatmap_path}",
            "targetInfo": target_info}


if __name__ == "__main__":
    mode = input("Input mode in [app, directory, fps, heatmap, image, k-means, map, onnx, summary, video]: ")

    # mode == "app"
    if mode == "app":
        yolo = Yolo()
        # 生成model.onnx
        yolo.convert_to_onnx(simplify=False)
        # 生成summary.txt
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        m = YoloBody(80).to(device)
        _sum = summary(m, (3, 416, 416))
        dummy_input = torch.randn(1, 3, 416, 416).to(device)
        flops, params = profile(m.to(device), (dummy_input,), verbose=False)
        flops = flops * 2
        flops, params = clever_format([flops, params], "%.3f")
        _sum += f"Total flops: {flops}\nTotal params: {params}\n{'-' * 95}"
        with open("data/summary.txt", "w") as sum_txt:
            sum_txt.write(_sum)
        sum_txt.close()
        # 清空缓存
        shutil.rmtree("data/cache", ignore_errors=True)
        os.makedirs("data/cache/image/out", exist_ok=True)
        os.makedirs("data/cache/image/heatmap", exist_ok=True)
        uvicorn.run(app, host="0.0.0.0", port=8080)

    # mode == "directory"
    elif mode == "directory":
        yolo = Yolo()
        image_dir = input("Input directory path: ")
        image_names = os.listdir(image_dir)
        for image_name in tqdm(image_names):
            if not image_name.lower().endswith(
                    [".bmp", ".dib", ".png", ".jpg", ".jpeg", ".pbm", ".pgm", ".ppm", ".tif", ".tiff"]):
                continue
            image_path = os.path.join(image_dir, image_name)

            image = yolo.detect_image(image_path)
            os.makedirs("data/cache/image/out", exist_ok=True)
            image.save(f"data/cache/image/out/{image_name.rsplit('.', 1)[0]}.png", quality=95, subsampling=0)

    # mode == "fps"
    elif mode == "fps":
        yolo = Yolo()
        fps_image_path = input("Input image path: ")
        image = Image.open(fps_image_path)
        tact_time = yolo.get_fps(image, test_interval=100)
        print(str(tact_time) + " seconds; " + str(1 / tact_time) + " fps; @batch_size 1")

    # mode == "k-means"
    elif mode == "k-means":
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

    # mode == "map"
    elif mode == "map":
        image_ids = open("data/VOC/ImageSets/Main/test.txt").read().strip().split()
        os.makedirs("data/cache/map/.gt", exist_ok=True)
        os.makedirs("data/cache/map/.dr", exist_ok=True)
        class_names, _ = get_classes()
        yolo = Yolo(confidence=0.001, nms_iou=0.5)
        for image_id in tqdm(image_ids):
            image_path = f"data/VOC/JPEGImages/{image_id}.jpg"
            image = Image.open(image_path)
            yolo.get_map_txt(image_id, image, class_names)
            with open(f"data/cache/map/.gt/{image_id}.txt", "w") as new_f:
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
                    bndbox = obj.find("bndbox")
                    left = bndbox.find("xmin").text
                    top = bndbox.find("ymin").text
                    right = bndbox.find("xmax").text
                    bottom = bndbox.find("ymax").text
                    if difficult_flag:
                        new_f.write(f"{obj_name} {left} {top} {right} {bottom} difficult\n")
                    else:
                        new_f.write(f"{obj_name} {left} {top} {right} {bottom}\n")
        get_map(0.5, 0.5)

    # mode == "onnx"
    elif mode == "onnx":
        yolo = Yolo()
        yolo.convert_to_onnx(simplify=False)

    # mode == "summary"
    elif mode == "summary":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        m = YoloBody(80).to(device)
        _sum = summary(m, (3, 416, 416))
        dummy_input = torch.randn(1, 3, 416, 416).to(device)
        flops, params = profile(m.to(device), (dummy_input,), verbose=False)
        flops = flops * 2
        flops, params = clever_format([flops, params], "%.3f")
        _sum += f"Total flops: {flops}\nTotal params: {params}\n{'-' * 95}"
        sum_txt = open("data/summary.txt", "w")
        sum_txt.write(_sum)
        sum_txt.close()
        print(_sum)

    # mode == "video"
    elif mode == "video":
        yolo = Yolo()
        video_path = input("Input video path, default camera.: ")
        os.makedirs("data/cache/video", exist_ok=True)
        video_out_path = f"data/cache/video/{datetime.strftime(datetime.now(), '%Y%m%d%H%M%S')}.avi"
        capture = cv2.VideoCapture(0 if video_path == "" else video_path)
        out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*"XVID"), 25.0,
                              (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        ref = False
        while not ref:
            ref, frame = capture.read()
        fps = 0.0
        try:
            while True:
                t1 = time.time()
                ref, frame = capture.read()
                if not ref:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(np.uint8(frame))
                image = yolo.detect_video(frame)
                frame = np.array(image)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                fps = (fps + (1. / (time.time() - t1))) / 2
                frame = cv2.putText(frame, "fps=%.2f" % fps, (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if video_path == "":
                    cv2.imshow("video", frame)
                c = cv2.waitKey(1) & 0xff
                out.write(frame)
                if c == 27:
                    break
        except KeyboardInterrupt:
            pass
        out.release()
        capture.release()
        cv2.destroyAllWindows()
        print(video_out_path + " saved")

    # raise ValueError
    else:
        raise ValueError("Input mode in [app, directory, fps, heatmap, image, k-means, map, onnx, summary, video].")
