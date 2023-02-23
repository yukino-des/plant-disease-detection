import map
import predict
import summary
import train
import annotation
import app
import kmeans
from utils import mobilenet_v2


def run(mode: str):
    if mode == "annotation":
        annotation.main()
    elif mode == "kmeans":
        kmeans.main()
    elif mode == "map":
        map.main()
    elif mode == "predict":
        predict.main()
    elif mode == "summary":
        summary.main()
    elif mode == "train":
        train.main()
    elif mode == "backbone":
        print(mobilenet_v2.mobilenet_v2())
    else:
        raise AssertionError("Use mode: 'annotation', 'kmeans',  'map', 'predict', 'summary', 'train', 'backbone'.")


if __name__ == '__main__':
    run(input())
