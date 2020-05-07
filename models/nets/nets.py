# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# Copyright (c) DMAI Inc. and its affiliates. All Rights Reserved.
# Licensed under The MIT License [see LICENSE for details]
# Written by panjinquan@dm-ai.cn
# @Project: torch-Face-Recognize-Pipeline
# @Author : panjq
# @Date   : 2019-9-20 13:18:34
# --------------------------------------------------------
"""

import torch
from torchvision import models
from models.nets.backbones import resnet
from models.nets.backbones import mobilenet_v2


def build_net(model_name, input_size, num_classes, width_mult=1.0, pretrained=True):
    if "resnet" in model_name:
        model = resnet.resnet(model_name, pretrained=pretrained, num_classes=num_classes)
    elif model_name == "mobilenet_v2":
        model = mobilenet_v2.mobilenet_v2(pretrained, num_classes=num_classes, width_mult=width_mult)
    else:
        raise Exception("Error: model_name:{}".format(model_name))
    return model


if __name__ == "__main__":
    import numpy as np
    from torchviz import make_dot
    from torchsummary import summary

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 1
    num_classes = 2
    input_size = [64, 64]
    x = torch.randn(size=(batch_size, 3, input_size[0], input_size[1])).to(device)
    print("x.shape:{}".format(x.shape))
    model_name = "resnet18"
    model = build_net(model_name, input_size, num_classes, width_mult=1.0).to(device)
    out = model(x)

    summary(model, input_size=(3, input_size[0], input_size[1]), batch_size=batch_size, device="cuda")
    for k, v in model.named_parameters():
        # print(k,v)
        print(k)

    print(out.shape)
    g = make_dot(out)
    g.view()
