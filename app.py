import os
import sys
from datetime import timedelta

from PIL import Image
from flask import *
from gevent import pywsgi

from yolo import YOLO

app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = timedelta(seconds=1)


@app.after_request
def after_request(response):
    response.headers["Access-Control-Allow-Credentials"] = "true"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, X-Requested-With"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST"
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response


@app.route("/upload", methods=["GET", "POST"])
def upload():
    file = request.files["file"]
    if file is None:
        return jsonify({"status": 0})
    file_name, extend_name = file.filename.split(".")
    src_path = os.path.join("tmp/src", file.filename)
    file.save(src_path)
    if extend_name.lower() in ("bmp", "dib", "jpeg", "jpg", "pbm", "pgm", "png", "ppm", "tif", "tiff"):
        dest_path = os.path.join("tmp/dest", f"{file_name}.png")
        r_image, image_info = yolo.detect_image(Image.open(src_path))
        r_image.save(dest_path, quality=95, subsampling=0)
        return jsonify({"status": 1,
                        "image_url": "http://127.0.0.1:8081/show/" + src_path,
                        "draw_url": "http://127.0.0.1:8081/show/" + dest_path,
                        "image_info": image_info})


@app.route("/show/<path:img_path>", methods=["GET"])
def show(img_path):
    image = open(img_path, "rb").read()
    response = make_response(image)
    response.headers["Content-Type"] = "image/png"
    return response


if __name__ == "__main__":
    os.chdir(sys.path[0])
    dirs = ["tmp/src", "tmp/dest"]
    for _dir in dirs:
        if not os.path.exists(_dir):
            os.makedirs(_dir)
        for file in os.listdir(_dir):
            os.remove(os.path.join(_dir, file))
    yolo = YOLO()
    server = pywsgi.WSGIServer(listener=('0.0.0.0', 8081), application=app)
    server.serve_forever()
