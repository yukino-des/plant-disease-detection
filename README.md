# backend

### Backend project for plant disease detection: Pytorch + FastAPI

1. install requirements (GPU version)

```shell
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
pip install tensorboard

pip freeze > requirements.txt
pip uninstall -r requirements.txt -y
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

4. if `os.name == "Windows"`
```text
HTTP_PROXY http://127.0.0.1:7890
HTTPS_PROXY http://127.0.0.1:7890
```

5. git
```shell
git stash
git pull
git rm -r --cached .
```

### class blance

<img src="VOCdevkit/class_balance.png" alt="class balance">

### size distribution

<img src="VOCdevkit/size_distribution.png" alt="size distribution">

### aspect radio distribution

<img src="VOCdevkit/aspect_ratio_distribution.png" alt="aspect ratio distribution">