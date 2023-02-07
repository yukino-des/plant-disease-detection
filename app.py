import os
import shutil
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
def upload_file(file: Union[UploadFile, None] = None):
    print(file)
    if file and ('.' in file.filename) and (file.filename.rsplit('.', 1)[1] in {'png', 'jpg'}):
        src_path = os.path.join(r'./uploads', file.filename)
        try:
            with open(src_path, "wb+") as buffer:
                shutil.copyfileobj(file.file, buffer)
        finally:
            file.file.close()
        shutil.copy(src_path, './tmp/ct')
        image_path = os.path.join('./tmp/ct', file.filename)
        r_image, image_info = yolo.detect_image(Image.open(image_path), crop=crop, count=count, api=True)
        img_name = r_image.filename.split('/')[-1]
        r_image.save(os.path.join('./tmp/draw', img_name), quality=95, subsampling=0)
        return {'status': 1,
                'image_url': 'http://127.0.0.1:8081/tmp/ct/' + img_name,
                'draw_url': 'http://127.0.0.1:8081/tmp/draw/' + img_name,
                'image_info': image_info}
    return {'status': 0}


@app.get('/tmp/{file:path}')
def show_photo(file):
    if file is not None:
        return FileResponse(path=f'tmp/{file}', headers={'Content-Type': 'image/png'})


if __name__ == '__main__':
    files = ['uploads', 'tmp/ct', 'tmp/draw']
    for ff in files:
        if not os.path.exists(ff):
            os.makedirs(ff)
    yolo = YOLO()
    mode = "predict"
    crop = False
    count = False
    uvicorn.run(app, host="0.0.0.0", port=8081)
