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
# bndbox, trainval, upsample, xmax, xmin, ymax, ymin, xvid
```

2. 训练、检测
```shell
python train.py
python app.py
```
