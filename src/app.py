import os
import shutil
import sys
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from starlette.responses import FileResponse

proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not sys.path.__contains__(proj_dir):
    sys.path.append(proj_dir)
from utils.yolo import YOLO

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-Requested-With"],
)
yolo = YOLO()


@app.post("/upload", response_model=dict)
def upload(file=None):
    if file is None:
        return {"status": 0}
    file_name, extend_name = file.filename.split(".")
    original_path = os.path.join("../tmp/original", file.filename)
    detected_path = os.path.join("../tmp/detected", f"{file_name}.png")
    try:
        with open(original_path, "wb+") as buffer:
            shutil.copyfileobj(file.file, buffer)
    finally:
        file.file.close()
    if extend_name.lower() in ("bmp", "dib", "jpeg", "jpg", "pbm", "pgm", "png", "ppm", "tif", "tiff"):
        r_image, image_info = yolo.detect_image(Image.open(original_path))
        r_image.save(detected_path, quality=95, subsampling=0)
        return {"status": 1,
                "image_url": "http://127.0.0.1:8081//" + original_path,
                "draw_url": "http://127.0.0.1:8081//" + detected_path,
                "image_info": image_info}


@app.get("/tmp/{fpath:path}", response_class=FileResponse)
def tmp(fpath):
    print(fpath)
    return FileResponse(path=os.path.join("../tmp/", fpath), headers={"Content-Type": "image/png"})


if __name__ == "__main__":
    os.chdir(sys.path[0])
    shutil.rmtree("../tmp", ignore_errors=True)
    dirs = ["../tmp/imgs", "../tmp/imgs_out", "../tmp/maps_out", "../tmp/original", "../tmp/detected"]
    for _dir_ in dirs:
        os.makedirs(_dir_, exist_ok=True)
    uvicorn.run(app, host="0.0.0.0", port=8081)
