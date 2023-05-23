# 农业病害检测

### 1. 下载依赖

```shell
cd plant-disease-detection
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

### 2. 训练、预测

```shell
python train.py
python app.py
```

### 3. 附录

##### 调整的PlantDoc数据集

classes.txt

```text
apple leaf
apple rust leaf
apple scab leaf
bell pepper leaf
bell pepper leaf spot
blueberry leaf
cherry leaf
corn gray leaf spot
corn leaf blight
corn rust leaf
grape leaf
grape leaf black rot
peach leaf
potato leaf early blight
potato leaf late blight
raspberry leaf
soybean leaf
squash powdery mildew leaf
strawberry leaf
tomato early blight leaf
tomato leaf
tomato leaf bacterial spot
tomato leaf late blight
tomato leaf mosaic virus
tomato leaf yellow virus
tomato mold leaf
tomato septoria leaf spot

```

##### MS COCO数据集

classes.txt

```
aeroplane
bicycle
bird
boat
bottle
bus
car
cat
chair
cow
diningtable
dog
horse
motorbike
person
pottedplant
sheep
sofa
train
tvmonitor
```