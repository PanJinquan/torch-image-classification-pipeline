# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# Copyright (c) DMAI Inc. and its affiliates. All Rights Reserved.
# Licensed under The MIT License [see LICENSE for details]
# Written by panjinquan@dm-ai.cn
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

