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

@Deprecated
```python
import os

import numpy as np
import torch


def corr2d(in_tensor, kernel):
    c, ih, iw = in_tensor.shape
    _, kh, kw = kernel.shape
    out_tensor = np.zeros(shape=(ih - kh + 1, iw - kw + 1))
    for y in range(out_tensor.shape[0]):
        for x in range(out_tensor.shape[1]):
            for z in range(c):
                out_tensor[y, x] += (in_tensor[z, y:y + kh, x:x + kw]
                                     * kernel[z]).sum()
    return out_tensor


def pointwise(in_tensor, kernel):
    _, ih, iw = in_tensor.shape
    out_tensor = np.zeros(shape=(ih, iw))
    for y in range(ih):
        for x in range(iw):
            # np.matmul(in_tensor[:, y, x], kernel[:, 0, 0])
            out_tensor[y, x] += np.dot(in_tensor[:, y, x], kernel[:, 0, 0])
    return out_tensor


def relu(in_tensor):
    tensor0 = torch.zeros_like(in_tensor)
    out_tensor = torch.max(in_tensor, tensor0)
    return out_tensor


def relu6(in_tensor):
    tensor0 = torch.zeros_like(in_tensor)
    tensor6 = torch.ones_like(in_tensor) * 6
    out_tensor = torch.min(torch.max(in_tensor, tensor0), tensor6)
    return out_tensor


from xml.etree import ElementTree as et

if __name__ == "__main__":
    dir = "data/VOC/Annotations"
    for xml in os.listdir(dir):
        tree = et.parse(os.path.join(dir, xml))
        root = tree.getroot()
        for obj in root.findall("object"):
            for name in obj.findall("name"):
                pass
```