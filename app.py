import os
import shutil
import sys
from typing import Union

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
def upload(file: Union[UploadFile, None] = None):
    if file is None:
        return {'status': 0}
    file_name, extend_name = file.filename.split(".")
    src_path = os.path.join("tmp/src", file.filename)
    dest_path = os.path.join("tmp/dest", f"{file_name}.png")
    try:
        with open(src_path, "wb+") as buffer:
            shutil.copyfileobj(file.file, buffer)
    finally:
        file.file.close()
    if extend_name.lower() in ("bmp", "dib", "jpeg", "jpg", "pbm", "pgm", "png", "ppm", "tif", "tiff"):
        r_image, image_info = yolo.detect_image(Image.open(src_path))
        r_image.save(dest_path, quality=95, subsampling=0)
        return {"status": 1,
                "image_url": "http://127.0.0.1:8081/show/" + src_path,
                "draw_url": "http://127.0.0.1:8081/show/" + dest_path,
                "image_info": image_info}


@app.get('/show/{fpath:path}', response_class=FileResponse)
def show(fpath):
    return FileResponse(path=fpath, headers={'Content-Type': 'image/png'})


if __name__ == '__main__':
    os.chdir(sys.path[0])
    dirs = ["tmp/src", "tmp/dest"]
    for _dir in dirs:
        if not os.path.exists(_dir):
            os.makedirs(_dir)
        for file in os.listdir(_dir):
            os.remove(os.path.join(_dir, file))
    yolo = YOLO()
    uvicorn.run(app, host="0.0.0.0", port=8081)
