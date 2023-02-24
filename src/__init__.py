# import maps
# import predict
# import summary
# import train
# import annotations
# import app
# import kmeans
# from utils import mobilenetv2
#
#
# def run(mode: str):
#     if mode == "annotation":
#         annotations.main()
#     elif mode == "kmeans":
#         kmeans.main()
#     elif mode == "map":
#         maps.main()
#     elif mode == "predict":
#         predict.main()
#     elif mode == "summary":
#         summary.main()
#     elif mode == "train":
#         train.main()
#     elif mode == "backbone":
#         print(mobilenetv2.mobilenet_v2())
#     else:
#         raise AssertionError("Use mode: 'annotation', 'kmeans',  'map', 'predict', 'summary', 'train', 'backbone'.")
#
#
# if __name__ == '__main__':
#     run(input())
