# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Project: torch-anti-spoofing-pipeline
# @Author : panjq
# @E-mail : pan_jinquan@163.com
# @Date   : 2020-04-15 17:04:25
# --------------------------------------------------------
"""
import torch
import torch.nn as nn
from torchvision import models
from collections import OrderedDict


def resnet(model_name, num_classes, pretrained=True):
    """
    :param model_name: resnet18,resnet34
    :param num_classes: if None ,return no-classifier-layers backbone
    :param pretrained: <bool> pretrained
    :return:
    """
    if model_name == "resnet18":
        model = models.resnet18(pretrained=pretrained)
    elif model_name == "resnet34":
        model = models.resnet34(pretrained=pretrained)
    else:
        raise Exception("Error: model_name:{}".format(model_name))

    if num_classes:
        expansion = 1
        model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model.fc = nn.Linear(512 * expansion, num_classes)
    else:
        # remove mobilenet_v2  classifier layers
        model_dict = OrderedDict(model.named_children())
        model_dict.pop("avgpool")
        model_dict.pop("fc")
        model = torch.nn.Sequential(model_dict)
    return model


if __name__ == "__main__":
    import numpy as np
    from torchviz import make_dot
    from torchsummary import summary

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 1
    num_classes = 2
    input_size = [112 * 2, 112 * 2]
    x = torch.randn(size=(batch_size, 3, input_size[0], input_size[1])).to(device)
    print("x.shape:{}".format(x.shape))
    model_name = "resnet18"
    model = resnet18(model_name, num_classes=num_classes, pretrained=True).to(device)
    out = model(x)
    torch.save(model.state_dict(), "model.pth")
    summary(model, input_size=(3, input_size[0], input_size[1]), batch_size=batch_size, device="cuda")
    for k, v in model.named_parameters():
        # print(k,v)
        print(k)
    print(out.shape)
    g = make_dot(out)
    g.view()
