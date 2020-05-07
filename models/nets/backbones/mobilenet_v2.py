# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Project: torch-anti-spoofing-pipeline
# @Author : panjq
# @E-mail : pan_jinquan@163.com
# @Date   : 2020-04-15 17:03:03
# --------------------------------------------------------
"""

import torch
from torch import nn
from torchvision import models
from collections import OrderedDict


def mobilenet_v2(pretrained, num_classes=None, width_mult=1.0):
    """
    :param pretrained: <bool> pretrained
    :param num_classes: if None ,return no-classifier-layers backbone
    :param last_channel:
    :param width_mult:
    :return:
    """
    model = models.mobilenet_v2(pretrained=pretrained, width_mult=width_mult)
    # state_dict1 = model.state_dict()
    if num_classes:
        last_channel = 1280
        # replace mobilenet_v2  classifier layers
        classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes),
        )
        model.classifier = classifier
    else:
        # remove mobilenet_v2  classifier layers
        model_dict = OrderedDict(model.named_children())
        model_dict.pop("classifier")
        model = torch.nn.Sequential(model_dict)
        # state_dict2 = model.state_dict()
    return model


if __name__ == "__main__":
    import numpy as np
    from torchviz import make_dot
    from torchsummary import summary

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 1
    num_classes = 2
    input_size = [112, 112]
    input_type = ["depth", "ir"]
    x = torch.randn(size=(batch_size, 3, input_size[0], input_size[1])).to(device)
    print("x.shape:{}".format(x.shape))
    model_name = "mobilenet_v2"
    # model_name = "multi_mobilenet_v2"
    model = mobilenet_v2(num_classes=num_classes, pretrained=True).to(device)
    out = model(x)
    torch.save(model.state_dict(), "model.pth")
    summary(model, input_size=(3, input_size[0], input_size[1]), batch_size=batch_size, device="cuda")
    for k, v in model.named_parameters():
        # print(k,v)
        print(k)
    print(out)
    print(out.shape)
    g = make_dot(out)
    g.view()
