# 农业病虫害检测

### 结构
```text
├── LICENSE
├── README.md
├── app.py
├── data
│   ├── VOC
│   │   ├── Annotations
│   │   ├── ImageSets/Main
│   │   │   ├── test.txt
│   │   │   ├── train.txt
│   │   │   ├── trainval.txt
│   │   │   └── val.txt
│   │   └── JPEGImages
│   ├── anchors.txt
│   ├── classes.txt
│   ├── mobilenet_v2-b0353104.pth
│   ├── model.pth
│   ├── train.txt
│   ├── あ.txt
│   └── val.txt
├── model.py
├── requirements.txt
├── train.py
└── utils.py
```

1. 依赖
```shell
cd plant-disease-detection
pip install -r requirements.txt
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```

2. 训练
```shell
python train.py
```

3. 检测
```shell
python app.py
```

4. 字典
```text
bndbox, trainval, upsample, xmax, xmin, ymax, ymin, xvid
```