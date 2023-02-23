### Plant disease detection, using MobileNetV2 and YOLOv4, Pytorch implementation.

```shell
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
pip install python-multipart tensorboard
pip install plant-disease-detection-3181137349

python setup.py sdist --formats=gztar
twine upload dist/* -u yukino-des -p Tch228jp 
```