# 农作物病虫害检测
使用MobileNet V2代替YOLOv4中主干特征提取网络DarkNet

### 结构
```text
.
├── LICENSE
├── README.md
├── app.py
├── data
│         ├── VOC
│         ├── anchors.txt
│         ├── cache
│         ├── classes.txt
├── model.py
├── present.ipynb
├── requirements.txt
├── train.py
├── utils.py
```

### 开始
1. 下载依赖
```shell
cd plant-disease-detection
pip install -r requirements.txt
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```

2. 制作data/anchors.txt文件
```shell
touch data/anchors.txt # 写入下方文本
```
```text
12, 16,  19, 36,  40, 28,  36, 75,  76, 55,  72, 146,  142, 110,  192, 243,  459, 401
```

3. 制作data/classes.txt文件
```shell
touch data/classes.txt # 写入下方文本
```
```text
Apple Scab Leaf
Apple leaf
Apple rust leaf
Bell_pepper leaf
Bell_pepper leaf spot
Blueberry leaf
Cherry leaf
Corn Gray leaf spot
Corn leaf blight
Corn rust leaf
Peach leaf
Potato leaf early blight
Potato leaf late blight
Raspberry leaf
Soybean leaf
Squash Powdery mildew leaf
Strawberry leaf
Tomato Early blight leaf
Tomato Septoria leaf spot
Tomato leaf
Tomato leaf bacterial spot
Tomato leaf late blight
Tomato leaf mosaic virus
Tomato leaf yellow virus
Tomato mold leaf
grape leaf
grape leaf black rot
```
data/VOC数据集请邮件作者：3181137349@qq.com

4. 训练
```shell
python train.py
```

5. 农作物病虫害检测，支持 app, dir, fps, heatmap, img, kmeans, map, onnx, sum, video 10种模式
```shell
python app.py
```

