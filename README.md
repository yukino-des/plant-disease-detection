# 農作物の病気と害虫の検出

### プロジェクト構造
```text
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

### はじめましょう

1. インストールパッケージ
```shell
cd plant-disease-detection
pip install -r requirements.txt
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```

2. data/anchors.txtファイルを作成
```shell
touch data/anchors.txt # 以下を書いてください
```
```text
12, 16,  19, 36,  40, 28,  36, 75,  76, 55,  72, 146,  142, 110,  192, 243,  459, 401
```

3. data/classes.txtファイルを作成
```shell
touch data/classes.txt # 以下を書いてください
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
data/VOC datasetについては、作成者に連絡してください。

4. トレーニングモデル
```shell
python train.py
```

5. 農作物の病気と害虫の検出を開始し、27 種類の病気と害虫をサポートし、複数の検出モードをサポートします。
```shell
python app.py
```

