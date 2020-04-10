# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Project: torch-Face-Recognize-Pipeline
# @File   : validation.py
# @Author : panjq
# @Date   : 2019-6-20 13:18:34
# --------------------------------------------------------
"""

import os
import sys
import torch
import numpy as np
import PIL.Image as Image
from models.nets import nets
from models.dataloader import custom_transform
from utils import image_processing, file_processing

sys.path.append(os.getcwd())


class Predictor(object):
    def __init__(self, model_path, model_name, input_size, num_classes, device):
        self.device = device
        RGB_MEAN = [0.5, 0.5, 0.5]
        RGB_STD = [0.5, 0.5, 0.5]
        self.val_transform = self.transform(input_size, RGB_MEAN, RGB_STD)
        self.model = self.build_net(model_name, input_size, num_classes)
        self.load_model(model_path)
        self.model.to(device)
        self.model.eval()  # set to val mode

    def build_net(self, model_name, input_size, num_classes):
        """
        :param model_name:
        :param input_size:
        :param num_classes:
        :return:
        """
        model = nets.build_net(model_name, input_size, num_classes)
        return model

    def transform(self, input_size, RGB_MEAN, RGB_STD):
        val_transform = custom_transform.custom_transform(input_size,
                                                          RGB_MEAN,
                                                          RGB_STD,
                                                          transform_type="val")
        return val_transform

    def load_model(self, model_path):
        print("Loading  Checkpoint '{}'".format(model_path))
        # self.model.load_state_dict(torch.load(backbone_resume_pth))
        # state_dict = torch.load(model_path, map_location=lambda storage, loc: storage.cuda(device))
        state_dict = torch.load(model_path)
        self.model.load_state_dict(state_dict)

    def forward(self, input_tensor):
        """
        :param input_tensor: input tensor
        :return:
        """
        with torch.no_grad():
            out_tensor = self.model(input_tensor.to(self.device))
        return out_tensor

    def pre_process(self, image):
        """
        :param image:
        :return:
        """
        image = Image.fromarray(image)
        image_tensor = self.val_transform(image)
        image_tensor = torch.unsqueeze(image_tensor, dim=0)
        return image_tensor

    def post_process(self, output):
        """
        :param output:
        :return:
        """
        pred_score = self.softmax(output, axis=1)
        index = np.argmax(pred_score, axis=1)
        score = pred_score[:, index]
        return index, score

    def predict(self, image):
        """
        :param image:
        :return:
        """
        input_tensor = self.pre_process(image)
        output = self.forward(input_tensor)
        output = output.cpu().data.numpy()
        pred_index, pred_score = self.post_process(output)
        return pred_index, pred_score

    @staticmethod
    def softmax(x, axis=1):
        """
        :param x:
        :param axis:
        :return:
        """
        # 计算每行的最大值
        row_max = x.max(axis=axis)

        # 每行元素都需要减去对应的最大值，否则求exp(x)会溢出，导致inf情况
        row_max = row_max.reshape(-1, 1)
        x = x - row_max

        # 计算e的指数次幂
        x_exp = np.exp(x)
        x_sum = np.sum(x_exp, axis=axis, keepdims=True)
        s = x_exp / x_sum
        return s

    def image_predict(self, image_dir):
        image_list = file_processing.get_images_list(image_dir, postfix=["*.jpg"])
        for path in image_list:
            image = image_processing.read_image(path, colorSpace="RGB")
            pre_index, max_score = self.predict(image)
            info = "path:{} pre labels:{},score:{}".format(path, pre_index, max_score)
            print(info)
            image_processing.cv_show_image("predict", image)


if __name__ == "__main__":
    model_path = "data/pretrained/model_ResNet18_099.pth"
    image_dir = "data/test_images"
    model_name = "ResNet18"
    input_size = [112, 112]
    num_classes = 4
    device = "cuda:0"
    p = Predictor(model_path, model_name, input_size, num_classes, device)
    p.image_predict(image_dir)
