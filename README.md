# backend

### Backend project for plant disease detection (Pytorch + FastAPI)

1. install requirements (GPU version)

```shell
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
pip install python-multipart

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