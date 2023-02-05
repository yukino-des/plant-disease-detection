import logging as rel_log
import os
import shutil
from datetime import timedelta

from PIL import Image
from flask import *

from yolo import YOLO

UPLOAD_FOLDER = r'./uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg'}
app = Flask(__name__)
app.secret_key = 'secret!'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
werkzeug_logger = rel_log.getLogger('werkzeug')
werkzeug_logger.setLevel(rel_log.ERROR)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)


@app.after_request
def after_request(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    response.headers['Access-Control-Allow-Methods'] = 'POST'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, X-Requested-With'
    return response


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    file = request.files['file']
    if file and allowed_file(file.filename):
        src_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(src_path)
        shutil.copy(src_path, './tmp/ct')
        image_path = os.path.join('./tmp/ct', file.filename)

        # modify
        r_image, image_info = yolo.detect_image(Image.open(image_path), crop=crop, count=count)
        img_name = r_image.filename.split('/')[-1]
        r_image.save(os.path.join('./tmp/draw', img_name), quality=95, subsampling=0)
        ########

        return jsonify({'status': 1,
                        'image_url': 'http://127.0.0.1:8081/tmp/ct/' + img_name,
                        'draw_url': 'http://127.0.0.1:8081/tmp/draw/' + img_name,
                        'image_info': image_info})
    return jsonify({'status': 0})


@app.route('/tmp/<path:file>', methods=['GET'])
def show_photo(file):
    if request.method == 'GET':
        if not file is None:
            image_data = open(f'tmp/{file}', "rb").read()
            response = make_response(image_data)
            response.headers['Content-Type'] = 'image/png'
            return response


if __name__ == '__main__':
    files = ['uploads', 'tmp/ct', 'tmp/draw']  # 'tmp/image', 'tmp/mask', 'tmp/uploads'
    for ff in files:
        if not os.path.exists(ff):
            os.makedirs(ff)
    with app.app_context():
        yolo = YOLO()
        mode = "predict"
        crop = False
        count = True
    app.run(host='127.0.0.1', port=8081, debug=False)
