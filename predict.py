import cv2
import matplotlib
import numpy as np
import os
import shutil
import time
import torch
import uvicorn
from datetime import datetime
from matplotlib import pyplot as plt
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from starlette.responses import FileResponse
from thop import clever_format, profile
from torchsummary import summary
from tqdm import tqdm
from util import avg_iou, get_classes, get_map, kmeans, load_data
from xml.etree import ElementTree as ET
from yolo import YOLO, YoloBody

if os.name == "nt":
    matplotlib.use("Agg")
else:
    matplotlib.use("TkAgg")
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["GET", "POST"],
                   allow_headers=["Content-Type", "X-Requested-With"])


@app.post("/upload", response_model=dict)
def upload(file: UploadFile):
    if file is None:
        return {"status": 0}
    file_name, extend_name = file.filename.split(".")
    original_path = os.path.join("tmp/original", file.filename)
    detected_path = os.path.join("tmp/detected", f"{file_name}.png")
    try:
        with open(original_path, "wb+") as buffer:
            shutil.copyfileobj(file.file, buffer)
    finally:
        file.file.close()
    if extend_name.lower() in ("bmp", "dib", "jpeg", "jpg", "pbm", "pgm", "png", "ppm", "tif", "tiff"):
        r_image, image_info = yolo.detect_image(Image.open(original_path))
        r_image.save(detected_path, quality=95, subsampling=0)
        return {"status": 1,
                "image_url": "http://127.0.0.1:8081/" + original_path,
                "draw_url": "http://127.0.0.1:8081/" + detected_path,
                "image_info": image_info}


@app.get("/tmp/{fpath:path}", response_class=FileResponse)
def tmp(fpath):
    print(fpath)
    return FileResponse(os.path.join("tmp", fpath), headers={"Content-Type": "image/png"})


if __name__ == "__main__":
    yolo = YOLO()
    mode = input("Input mode in [app, dir, fps, heatmap, img, kmeans, map, onnx, video], default app.: ")
    crop = True
    fps_interval = 100  # mode = "fps"
    # dir
    if mode == "dir":
        img_names = os.listdir("tmp/imgs")
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(
                    (".bmp", ".dib", ".png", ".jpg", ".jpeg", ".pbm", ".pgm", ".ppm", ".tif", ".tiff")):
                image_path = os.path.join("tmp/imgs", img_name)
                image = Image.open(image_path)
                r_image, _ = yolo.detect_image(image)
                os.makedirs("tmp/imgs_out", exist_ok=True)
                r_image.save(os.path.join("tmp/imgs_out", img_name.replace(".jpg", ".png")), quality=95, subsampling=0)
    # fps
    elif mode == "fps":
        fps_image_path = input("Input image path: ")
        img = Image.open(fps_image_path)
        tact_time = yolo.get_fps(img, fps_interval)
        print(str(tact_time) + " seconds, " + str(1 / tact_time) + " fps, @batch_size 1")
    # heatmap
    elif mode == "heatmap":
        img = input("Input image path: ")
        try:
            image = Image.open(img)
        except:
            pass
        else:
            yolo.detect_heatmap(image, "tmp/heatmap.png")
    # img
    elif mode == "img":
        img = input("Input image path: ")
        try:
            image = Image.open(img)
        except:
            pass
        else:
            r_image, _ = yolo.detect_image(image, crop=crop)
            r_image.show()
            r_image.save(os.path.join("tmp/imgs_out", img.split(".")[-2] + ".png"), quality=95, subsampling=0)
    # kmeans
    elif mode == "kmeans":
        np.random.seed(0)
        input_shape = [416, 416]
        anchors_num = 9
        data = load_data("VOC/Annotations")
        cluster, near = kmeans(data, anchors_num)
        data = data * np.array([input_shape[1], input_shape[0]])
        cluster = cluster * np.array([input_shape[1], input_shape[0]])
        for j in range(anchors_num):
            plt.scatter(data[near == j][:, 0], data[near == j][:, 1])
            plt.scatter(cluster[j][0], cluster[j][1], marker="x", c="black")
        os.makedirs("tmp", exist_ok=True)
        plt.savefig("tmp/kmeans.jpg")
        print("tmp/kmeans.jpg saved.")
        cluster = cluster[np.argsort(cluster[:, 0] * cluster[:, 1])]
        print("avg_ratio: %.2f" % (avg_iou(data, cluster)))
        print(f"cluster:\n{cluster}")
        f = open("tmp/anchors.txt", "w")
        row = np.shape(cluster)[0]
        for i in range(row):
            if i == 0:
                xy = "%d,%d" % (cluster[i][0], cluster[i][1])
            else:
                xy = ", %d,%d" % (cluster[i][0], cluster[i][1])
            f.write(xy)
        f.close()
    # map
    elif mode == "map":
        min_overlap = 0.5
        confidence = 0.001
        nms_iou = 0.5
        score_threshold = 0.5
        image_ids = open("VOC/ImageSets/Main/test.txt").read().strip().split()
        os.makedirs("tmp/maps_out/.gt", exist_ok=True)
        os.makedirs("tmp/maps_out/.dr", exist_ok=True)
        class_names, _ = get_classes("data/classes.txt")
        yolo = YOLO(confidence=confidence, nms_iou=nms_iou)
        for image_id in tqdm(image_ids):
            image_path = f"VOC/JPEGImages/{image_id}.jpg"
            image = Image.open(image_path)
            yolo.get_map_txt(image_id, image, class_names, "tmp/maps_out")
            with open(f"tmp/maps_out/.gt/{image_id}.txt", "w") as new_f:
                root = ET.parse(f"VOC/Annotations/{image_id}.xml").getroot()
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
        get_map(min_overlap, True, score_threshold=score_threshold, path="tmp/maps_out")
        shutil.rmtree("tmp/maps_out/.dr", ignore_errors=True)
        shutil.rmtree("tmp/maps_out/.gt", ignore_errors=True)
    # onnx
    elif mode == "onnx":
        yolo.convert_to_onnx(False, "data/model.onnx")
    # summary
    elif mode == "summary":
        input_shape = [416, 416]
        anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        num_classes = 80
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        m = YoloBody(anchors_mask, num_classes).to(device)
        summary(m, (3, input_shape[0], input_shape[1]))
        dummy_input = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
        flops, params = profile(m.to(device), (dummy_input,), verbose=False)
        flops = flops * 2
        flops, params = clever_format([flops, params], "%.3f")
        print(f"Total flops: {flops}")
        print(f"Total params: {params}")
    # video
    elif mode == "video":
        video_path = input("Input video path: ")
        capture = cv2.VideoCapture(0 if video_path == "" else video_path)
        os.makedirs("tmp/videos_out", exist_ok=True)
        video_save_path = f"tmp/videos_out/{datetime.strftime(datetime.now(), '%H%M%S')}.avi"
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
            frame = cv2.putText(frame, "fps= %.2f" % fps, (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if video_path == "":
                cv2.imshow("video", frame)
            c = cv2.waitKey(1) & 0xff
            out.write(frame)
            if c == 27:
                break
        capture.release()
        out.release()
        cv2.destroyAllWindows()
        print(video_save_path + " saved")
    # app
    else:
        shutil.rmtree("tmp", ignore_errors=True)
        for _dir in ["tmp/imgs", "tmp/imgs_out", "tmp/videos_out", "tmp/maps_out", "tmp/original", "tmp/detected"]:
            os.makedirs(_dir, exist_ok=True)
        uvicorn.run(app, host="0.0.0.0", port=8081)
