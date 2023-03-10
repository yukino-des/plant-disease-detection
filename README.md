# 农业病虫害检测
```angular2html
ヽ｀、ヽ｀｀、ヽ｀ヽ｀、、ヽ｀ヽ、ヽ｀🌙｀ヽヽ
｀ヽ、ヽ｀ヽ｀、ヽ｀｀、ヽ、｀｀、｀、ヽ｀、｀
ヽ｀ヽ、ヽ｀、ヽ｀｀、ヽ、｀｀、｀、ヽ｀｀、、
ヽ｀、｀、、ヽヽ、｀｀、、ヽ｀、ヽ｀｀、ヽ｀ヽ
｀、、ヽ｀ヽ、ヽ｀｀ヽ、｀｀ヽ｀、、🚶｀ヽ｀、
```

### 结构
```text
├── LICENSE
├── README.md
├── app.py
├── data
│   ├── VOC
│   │   ├── Annotations
│   │   ├── ImageSets/Main
│   │   └── JPEGImages
│   ├── anchors.txt
│   ├── classes.txt
│   ├── font.ttc
│   └── model.pth
├── model.py
├── requirements.txt
├── train.py
└── utils.py
```

### 开始
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