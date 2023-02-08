import os
import shutil
import sys
import time
from typing import Union

import cv2
import numpy as np
import uvicorn
from PIL import Image
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse

from yolo import YOLO

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Requested-With"],
)


@app.post('/upload', response_model=dict)
def upload_file(file: Union[UploadFile, None] = None):
    if file is None:
        return {'status': 0}
    file_name, extend_name = file.filename.split('.')
    ori_path = os.path.join('tmp/ori', file.filename)
    try:
        with open(ori_path, "wb+") as buffer:
            shutil.copyfileobj(file.file, buffer)
    finally:
        file.file.close()
    if extend_name in {'jpg', 'png'}:
        det_path = os.path.join('tmp/det', f'{file_name}.png')
        r_image, image_info = yolo.detect_image(Image.open(ori_path), info=True)
        r_image.save(det_path, quality=95, subsampling=0)
        return {'status': 1,
                'image_url': 'http://127.0.0.1:8081/' + ori_path,
                'draw_url': 'http://127.0.0.1:8081/' + det_path,
                'image_info': image_info}
    if extend_name in {'mp4', 'avi'}:
        det_path = os.path.join('tmp/det', f'{file_name}.avi')
        video_fps = 25.0
        capture = cv2.VideoCapture(ori_path)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        out = cv2.VideoWriter(det_path, fourcc, video_fps, size)
        ref, frame = capture.read()
        if not ref:
            raise ValueError("Failed to read the video!")
        fps = 0.0
        while True:
            t1 = time.time()
            ref, frame = capture.read()
            if not ref:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(np.uint8(frame))
            frame = np.array(yolo.detect_image(frame))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            fps = (fps + (1. / (time.time() - t1))) / 2
            frame = cv2.putText(frame, "fps= %.2f" % fps, (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            c = cv2.waitKey(1) & 0xff
            out.write(frame)
            if c == 27:
                capture.release()
                break
        capture.release()
        out.release()
        cv2.destroyAllWindows()
        return {'status': 1,
                'image_url': 'http://127.0.0.1:8081/' + ori_path,
                'draw_url': 'http://127.0.0.1:8081/' + det_path,
                'image_info': {}}


@app.get('/tmp/{path:path}')
def show_photo(path):
    if path is None:
        return {'status': 0}
    filename = path.split('/')[1]
    extend_name = filename.split('.')[1]
    if extend_name in {'jpg', 'png'}:
        return FileResponse(path=f'tmp/{path}', headers={'Content-Type': 'image/png'}, filename=filename)
    if extend_name in {'mp4', 'avi'}:
        return FileResponse(path=f'tmp/{path}', headers={'Content-Type': 'video/mp4'}, filename=filename)


if __name__ == '__main__':
    # append
    os.chdir(sys.path[0])

    dirs = ['tmp/ori/', 'tmp/det/']
    for dire in dirs:
        if not os.path.exists(dire):
            os.makedirs(dire)
        for file in os.listdir(dire):
            os.remove(os.path.join(dire, file))
    yolo = YOLO()
    mode = "predict"
    uvicorn.run(app, host="0.0.0.0", port=8081)
