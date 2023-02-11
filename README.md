### backend
Backend project for plant disease detection (Python 3.9.6)

1. install requirements
```shell
# see requirements.txt
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
```

2. run backend project
```shell
python app.py
```

3. if `address already in use`
```shell
lsof -i:8081
kill -9 {PID}
```