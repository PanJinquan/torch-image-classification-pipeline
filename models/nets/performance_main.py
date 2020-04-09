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
from utils import debug
from models.nets import model_mixnet
from models.nets import model_irse
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
print("-----device:{}".format(device))
print("-----Pytorch version:{}".format(torch.__version__))


# @debug.run_time_decorator()
def model_forward(model, input_tensor):
    T0 = debug.TIME()
    out = model(input_tensor)
    torch.cuda.synchronize()
    T1 = debug.TIME()
    time = debug.RUN_TIME(T1 - T0)
    return out, time


def iter_model(model, input_tensor, iter):
    out, time = model_forward(model, input_tensor)
    all_time = 0
    for i in range(iter):
        out, time = model_forward(model, input_tensor)
        all_time += time
    return all_time


def squeezenet1_0(input_tensor, out_features, iter=10):
    model = models.squeezenet.squeezenet1_0(pretrained=False, num_classes=out_features).to(device)
    model.eval()
    all_time = iter_model(model, input_tensor, iter)
    print("squeezenet1_0,mean run time :{:.3f}".format(all_time / iter))


def squeezenet1_1(input_tensor, out_features, iter=10):
    model = models.squeezenet.squeezenet1_1(pretrained=False, num_classes=out_features).to(device)
    model.eval()
    all_time = iter_model(model, input_tensor, iter)
    print("squeezenet1_1,mean run time :{:.3f}".format(all_time / iter))


def mnasnet1_0(input_tensor, out_features, iter=10):
    model = models.mnasnet.mnasnet1_0(pretrained=False, num_classes=out_features).to(device)
    model.eval()
    all_time = iter_model(model, input_tensor, iter)
    print("mnasnet1_0,mean run time :{:.3f}".format(all_time / iter))


def shufflenet_v2_x1_0(input_tensor, out_features, iter=10):
    model = models.shufflenetv2.shufflenet_v2_x1_0(pretrained=False, num_classes=out_features).to(device)
    model.eval()
    all_time = iter_model(model, input_tensor, iter)
    print("shufflenet_v2_x1_0,mean run time :{:.3f}".format(all_time / iter))


def mobilenet_v2(input_tensor, out_features, iter=10):
    model = models.mobilenet_v2(pretrained=False, num_classes=out_features).to(device)
    model.eval()
    all_time = iter_model(model, input_tensor, iter)
    print("mobilenet_v2,mean run time :{:.3f}".format(all_time / iter))


def resnet18(input_tensor, out_features, iter=10):
    model = models.resnet18(pretrained=False, num_classes=out_features).to(device)
    model.eval()
    all_time = iter_model(model, input_tensor, iter)
    print("reset18,mean run time :{:.3f}".format(all_time / iter))


def ir_resnet18(input_tensor,input_size, out_features, iter=10):
    model = model_irse.IR_18(input_size, out_features).to(device)
    model.eval()
    all_time = iter_model(model, input_tensor, iter)
    print("reset18,mean run time :{:.3f}".format(all_time / iter))

def resnet34(input_tensor, out_features, iter=10):
    model = models.resnet34(pretrained=False, num_classes=out_features).to(device)
    model.eval()
    all_time = iter_model(model, input_tensor, iter)
    print("resnet34,mean run time :{:.3f}".format(all_time / iter))


def vgg16(input_tensor, out_features, iter=10):
    model = models.vgg16(pretrained=False, num_classes=out_features).to(device)
    model.eval()
    all_time = iter_model(model, input_tensor, iter)
    print("vgg16,mean run time :{:.3f}".format(all_time / iter))


def MixNet_L(input_tensor, input_size, out_features, iter=10):
    model = model_mixnet.MixNet_L(input_size, out_features).to(device)
    model.eval()
    all_time = iter_model(model, input_tensor, iter)
    print("MixNet_L,mean run time :{:.3f}".format(all_time / iter))


def MixNet_M(input_tensor, input_size, out_features, iter=10):
    model = model_mixnet.MixNet_M(input_size, out_features).to(device)
    model.eval()
    all_time = iter_model(model, input_tensor, iter)
    print("MixNet_M,mean run time :{:.3f}".format(all_time / iter))


def MixNet_S(input_tensor, input_size, out_features, iter=10):
    model = model_mixnet.MixNet_S(input_size, out_features).to(device)
    model.eval()
    all_time = iter_model(model, input_tensor, iter)
    print("MixNet_S,mean run time :{:.3f}".format(all_time / iter))


def inception_v3(input_tensor, out_features, iter=10):
    model = models.inception.inception_v3(pretrained=False, num_classes=out_features).to(device)
    model.eval()
    all_time = iter_model(model, input_tensor, iter)
    print("inception_v3,mean run time :{:.3f}".format(all_time / iter))

def googlenet(input_tensor, out_features, iter=10):
    model = models.googlenet(pretrained=False, num_classes=out_features).to(device)
    model.eval()
    all_time = iter_model(model, input_tensor, iter)
    print("googlenet,mean run time :{:.3f}".format(all_time / iter))



if __name__ == "__main__":
    input_size = [64, 64]
    out_features = 256
    input_tensor = torch.randn(1, 3, input_size[0], input_size[1]).to(device)
    print('input_tensor:', input_tensor.shape)
    iter = 100
    # mobilenet_v2(input_tensor, out_features, iter)
    # resnet18(input_tensor, out_features, iter)
    # resnet34(input_tensor, out_features, iter)
    # vgg16(input_tensor, out_features, iter)
    # squeezenet1_0(input_tensor, out_features, iter)
    # squeezenet1_1(input_tensor, out_features, iter)
    # inception_v3(input_tensor, out_features, iter)
    # googlenet(input_tensor, out_features, iter)
    ir_resnet18(input_tensor, input_size, out_features, iter)
    # mnasnet1_0(input_tensor, out_features, iter)
    # shufflenet_v2_x1_0(input_tensor, out_features, iter)
    # MixNet_S(input_tensor, input_size, out_features, iter)
    # MixNet_M(input_tensor, input_size, out_features, iter)
    # MixNet_L(input_tensor, input_size, out_features, iter)

