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


def build_net(model_name, input_size, num_classes, width_mult=1.0):
    model = models.resnet18(pretrained=False, num_classes=num_classes)
    return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 1
    embedding_size = 512
    input_size = [64, 64]
    x = torch.randn(size=(batch_size, 3, input_size[0], input_size[1])).to(device)
    print("x.shape:{}".format(x.shape))
    backbone_name = "ResNet_18"
    model = build_net(backbone_name, embedding_size, input_size, width_mult=1.0).to(device)
    out = model(x)
    from torchsummary import summary

    summary(model, input_size=(3, input_size[0], input_size[1]), batch_size=batch_size, device=device)
    for k, v in model.named_parameters():
        # print(k,v)
        print(k)
