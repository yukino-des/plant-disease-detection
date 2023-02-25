from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as readme:
    long_description = readme.read()
classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: Unix"
]
setup_kwargs = {
    "author": "yukino-des",
    "author_email": "3181137349go@gmail.com",
    "classifiers": classifiers,
    "description": "Plant disease detection, using MobileNetV2 and YOLOv4, Pytorch implementation.",
    "entry_points": {},
    "include_package_data": True,
    "long_description": long_description,
    "long_description_content_type": "text/plain",
    "name": "plant-disease-detection-3181137349",
    "packages": find_packages(),
    "url": "https://github.com/yukin-des/plant-disease-detection.git",
    "version": "0.0.1",
    "install_requires": ["fastapi==0.92.0",
                         "matplotlib==3.7.0",
                         "numpy==1.24.2",
                         "onnx==1.13.0",
                         "onnxsim==0.4.17",
                         "opencv-python==4.7.0.68",
                         "Pillow==9.4.0",
                         "python-multipart==0.0.5",
                         "scipy==1.10.1",
                         "setuptools==58.0.4",
                         "starlette==0.25.0",
                         "tensorboard==2.12.0",
                         "thop==0.1.1.post2209072238",
                         "torch==1.13.1",
                         "torchsummary==1.5.1",
                         "torchvision==0.14.1",
                         "tqdm==4.64.1",
                         "uvicorn==0.20.0"],
}
setup(**setup_kwargs)
