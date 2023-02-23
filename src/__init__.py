from src import map
from src import predict
from src import summary
from src import train
from src import annotation
from src import app
from src import kmeans


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
    else:
        raise AssertionError("Use mode: 'annotation', 'kmeans',  'map', 'predict', 'summary', 'train'.")


if __name__ == "__main__":
    run(input())
