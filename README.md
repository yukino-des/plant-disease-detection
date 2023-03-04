# 作物病虫害の検出

### プロジェクト構造
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
│   └── val.txt
├── model.py
├── requirements.txt
├── train.py
└── utils.py
```

### はじめましょう！

1. インストールパッケージ。
```shell
cd plant-disease-detection
pip install -r requirements.txt
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```

2. トレーニングモデル
```shell
python train.py
```

3. 检测が始まり、27種類の作物病虫害と多重检测モードをサポートします。
```shell
python app.py
```