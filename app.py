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
from torchsummary import summary
from tqdm import tqdm
from utils import avg_iou, get_classes, get_map, k_means, load_data
from xml.etree import ElementTree

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["GET", "POST"],
                   allow_headers=["Content-Type", "X-Requested-With"])


@app.get("/data/{file_path:path}", response_class=FileResponse)
def data(file_path):
    return FileResponse(f"data/{file_path}", headers={"Content-Type": "image/png"})


@app.post("/upload", response_model=dict)
def upload(file: UploadFile):
    if file is None:
        return {"status": 0}
    file_name, extend_name = file.filename.rsplit(".", 1)
    img_path = f"data/cache/img/{file.filename}"
    img_out_path = f"data/cache/img/out/{file_name}.png"
    with open(img_path, "wb+") as buffer:
        shutil.copyfileobj(file.file, buffer)
    if extend_name.lower() in ("bmp", "dib", "jpeg", "jpg", "pbm", "pgm", "png", "ppm", "tif", "tiff"):
        _image, image_info = yolo.detect_image(Image.open(img_path))
        _image.save(img_out_path, quality=95, subsampling=0)
        return {"status": 1,
                "image_url": "http://0.0.0.0:8081/" + img_path,
                "draw_url": "http://0.0.0.0:8081/" + img_out_path,
                "image_info": image_info}
    else:
        return {"status": 0}


if __name__ == "__main__":
    mode = input("Input mode in [app, dir, fps, heatmap, img, k-means, map, onnx, sum, video]: ")
    if mode == "app":
        yolo = Yolo()
        shutil.rmtree("data/cache", ignore_errors=True)
        os.makedirs("data/cache/img/out", exist_ok=True)
        uvicorn.run(app, host="0.0.0.0", port=8081)
    elif mode == "dir":
        yolo = Yolo()
        img_dir = input("Input directory path: ")
        img_names = os.listdir(img_dir)
        for img_name in tqdm(img_names):
            if not img_name.lower().endswith(
                    (".bmp", ".dib", ".png", ".jpg", ".jpeg", ".pbm", ".pgm", ".ppm", ".tif", ".tiff")):
                continue
            image_path = os.path.join(img_dir, img_name)
            image = Image.open(image_path)
            image, _ = yolo.detect_image(image)
            os.makedirs("data/cache/img/out", exist_ok=True)
            image.save(f"data/cache/img/out/{img_name.rsplit('.', 1)[0]}.png", quality=95, subsampling=0)
    elif mode == "fps":
        yolo = Yolo()
        fps_image_path = input("Input image path: ")
        img = Image.open(fps_image_path)
        tact_time = yolo.get_fps(img, test_interval=100)
        print(str(tact_time) + " seconds; " + str(1 / tact_time) + " fps; @batch_size 1")
    elif mode == "heatmap":
        yolo = Yolo()
        img = input("Input image path: ")
        image = Image.open(img)
        yolo.detect_heatmap(image)
    elif mode == "img":
        yolo = Yolo()
        img = input("Input image path: ")
        img_name = img.rsplit("/", 1)[1].split(".")[0]
        image = Image.open(img)
        image, _ = yolo.detect_image(image)
        # image.show()
        image_save_path = f"data/cache/img/out/{img_name}.png"
        image.save(image_save_path, quality=95, subsampling=0)
        print(f"{image_save_path} saved.")
    elif mode == "k-means":
        np.random.seed(0)
        input_shape = [416, 416]
        anchors_num = 9
        data = load_data()
        cluster, near = k_means(data, anchors_num)
        data = data * np.array([416, 416])
        cluster = cluster * np.array([416, 416])
        for j in range(anchors_num):
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
    elif mode == "onnx":
        yolo = Yolo()
        yolo.convert_to_onnx(simplify=False)
    elif mode == "sum":
        input_shape = [416, 416]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        m = YoloBody(80).to(device)
        summary(m, (3, 416, 416))
        dummy_input = torch.randn(1, 3, 416, 416).to(device)
        flops, params = profile(m.to(device), (dummy_input,), verbose=False)
        flops = flops * 2
        flops, params = clever_format([flops, params], "%.3f")
        print(f"Total flops: {flops}")
        print(f"Total params: {params}")
    elif mode == "video":
        yolo = Yolo()
        video_path = input("Input video path, default camera.: ")
        capture = cv2.VideoCapture(0 if video_path == "" else video_path)
        os.makedirs("data/cache/video", exist_ok=True)
        video_save_path = f"data/cache/video/{datetime.strftime(datetime.now(), '%Y%m%d%H%M%S')}.avi"
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        out = cv2.VideoWriter(video_save_path, fourcc, 25.0, size)
        ref, frame = capture.read()
        if not ref:
            raise ValueError("Failed to read the camera/video.")
        fps = 0.0
        while True:
            t1 = time.time()
            ref, frame = capture.read()
            if not ref:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(np.uint8(frame))
            image, _ = yolo.detect_image(frame)
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
        out.release()
        capture.release()
        cv2.destroyAllWindows()
        print(video_save_path + " saved")
    else:
        raise ValueError("Input mode in [app, dir, fps, heatmap, img, k-means, map, onnx, sum, video]")
