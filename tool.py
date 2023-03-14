# import collections
# import torch
#
# pth = torch.load("data/model.pth", map_location=torch.device('cpu'))
# n_pth = collections.OrderedDict()
# print(type(pth))
# for k, v in pth.items():
#     if str.startswith(k, "make_five_conv"):
#         nk = str.replace(k, "make_five_conv", "make5conv")
#         k = nk
#     elif str.startswith(k, "conv_for_P"):
#         nk = str.replace(k, "P", "p")
#     else:
#         nk = k
#     n_pth[nk] = v
# for k in n_pth.keys():
#     print(k)
# torch.save(n_pth, "data/model.pth")
import os
from xml.etree import ElementTree

for xml_name in os.listdir("data/VOC/Annotations"):
    xml_path = f"data/VOC/Annotations/{xml_name}"
    xml_file = ElementTree.parse(xml_path)
    root = xml_file.getroot()
    for _object in root.findall("object"):
        for name in _object.findall("name"):
            name.text = str.replace(name.text, "_", " ")
            tree = ElementTree.ElementTree(root)
            tree.write(xml_path, encoding="utf-8")
