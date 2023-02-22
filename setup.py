from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as readme:
    long_description = readme.read()
classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: Unix"
]
setup_kwargs = {
    "name": "plant-disease-detection-3181137349",
    "version": "0.0.1",
    "author": "YukinoShita Yukino",
    "author_email": "3181137349go@gmail.com",
    "description": "Plant disease detection, using MobileNetV2 and YOLOv4, Pytorch implementation.",
    "long_description": long_description,
    "url": "https://github.com/yukin-des/plant-disease-detection.git",
    "packages": find_packages(),
    "include_package_data": True,
    "entry_points": {
        "console_scripts": [
            "backbone = nets:get_backbone"
        ]
    },
    "install_requires": ["torch==1.13.1",
                         "numpy==1.24.2",
                         "Pillow==9.4.0",
                         "matplotlib==3.7.0",
                         "scipy==1.10.1",
                         "tqdm==4.64.1",
                         "opencv-python==4.7.0.68",
                         "torchvision==0.14.1",
                         "onnx==1.13.0",
                         "onnxsim==0.4.17",
                         "setuptools==58.0.4",
                         "thop==0.1.1.post2209072238",
                         "torchsummary==1.5.1",
                         "python-multipart==0.0.5"],
    "classifiers": classifiers
}
setup(**setup_kwargs)
