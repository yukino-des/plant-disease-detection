import torch

pth = dict(torch.load("data/model.pth", map_location="cpu"))
npth = dict()
for k in pth.keys():
    if k.__contains__("make_five_conv"):
        npth[k.replace("make_five_conv", "make5conv")] = pth[k]
    elif k.__contains__("make_three_conv"):
        npth[k.replace("make_three_conv", "make3conv")] = pth[k]
    elif k.__contains__("down_sample"):
        npth[k.replace("down_sample", "downsample")] = pth[k]
    else:
        npth[k] = pth[k]
torch.save(npth, "data/nmodel.pth")
