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
import predictor
from utils import image_processing, file_processing

sys.path.append(os.getcwd())


class Validation(predictor.Predictor):
    def __init__(self, model_path, model_name, input_size, num_classes, device):
        args = model_path, model_name, input_size, num_classes, device
        super(Validation, self).__init__(*args)


if __name__ == "__main__":
    model_path = "data/pretrained/model_ResNet18_099.pth"
    image_dir = "data/test_images"
    model_name = "ResNet18"
    input_size = [112, 112]
    num_classes = 4
    device = "cuda:0"
    v = Validation(model_path, model_name, input_size, num_classes, device)
    v.image_predict(image_dir)
