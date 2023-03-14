import collections
import torch

pth = torch.load("data/model.pth", map_location=torch.device('cpu'))
n_pth = collections.OrderedDict()
print(type(pth))
for k, v in pth.items():
    if str.startswith(k, "make_five_conv"):
        nk = str.replace(k, "make_five_conv", "make5conv")
        k = nk
    elif str.startswith(k, "conv_for_P"):
        nk = str.replace(k, "P", "p")
    else:
        nk = k
    n_pth[nk] = v
for k in n_pth.keys():
    print(k)
torch.save(n_pth, "data/model.pth")
