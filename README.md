# 农业病害检测

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
│   ├── simhei.ttf
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
```

2. 训练、检测
```shell
python train.py
python app.py
```
