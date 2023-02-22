from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as readme:
    long_description = readme.read()
classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OSX"
]

setup_kwargs = {
    "name": "plant-disease-detection-3181137349",
    "version": "0.0.1",
    "author": "YukinoShita Yukino",
    "author_email": "3181137349go@gmail.com",
    "description": "Plant disease detection, using MobileNetV2 and YOLOv4, Pytorch implementation.",
    "long_description": long_description,
    "url": "https://github.com/yukin-des/"

}
