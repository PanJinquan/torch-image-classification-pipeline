# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Project: torch-image-classification-pipeline
# @Author : panjq
# @E-mail : pan_jinquan@163.com
# @Date   : 2020-04-14 11:47:47
# --------------------------------------------------------
"""
import os
import math
import PIL.Image as Image
import numpy as np
import random
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from utils import image_processing, file_processing
from models.dataloader import balanced_classes


class MultiModalDataset(Dataset):
    """多模态 dataset"""

    def __init__(self, filename, image_dir=None, input_type=["ir"], transform=None, shuffle=True, repeat=1):
        """
        :param filename:
        :param image_dir:
        :param input_type:模态,list ["color","depth","ir"]
        :param transform:
        :param shuffle:
        :param repeat:
        """
        if not os.path.exists(filename):
            raise Exception("Error:no file {}.".format(filename))
        self.filename = filename
        self.image_dir = image_dir
        self.input_type = input_type
        self.data_list = self.read_file(self.filename, shuffle)
        self.transform = transform
        self.repeat = repeat
        self.classes = list(set(self.get_label_list()))
        self.num_class = len(self.classes)
        self.num_images = len(self.data_list)
        print("num_images :{},class_num:{}".format(self.num_images, self.num_class))

    def __getitem__(self, idx):
        '''
        :param idx:
        :return: RGB image,label id
        '''
        item = self.data_list[idx]
        color_path, depth_path, ir_path, label_id = self.get_item_data(item, image_dir=self.image_dir)
        data = {}
        if "color" in self.input_type:
            color_image = self.read_image(color_path)
            color_image = self.data_preproccess(color_image)
            data["color"] = color_image
        if "depth" in self.input_type:
            depth_image = self.read_image(depth_path)
            depth_image = self.data_preproccess(depth_image)
            data["depth"] = depth_image
        if "ir" in self.input_type:
            ir_image = self.read_image(ir_path)
            ir_image = self.data_preproccess(ir_image)
            data["ir"] = ir_image
        data["label"] = label_id
        return data

    @staticmethod
    def get_item_data(item, image_dir=None):
        """
        :param item:
        :param image_dir:
        :return:
        """
        color_path = item[0]
        depth_path = item[1]
        ir_path = item[2]
        label_id = item[3]
        if image_dir:
            color_path = os.path.join(image_dir, color_path)
            depth_path = os.path.join(image_dir, depth_path)
            ir_path = os.path.join(image_dir, ir_path)
        return color_path, depth_path, ir_path, label_id

    def __len__(self):
        if self.repeat is None:
            data_len = 10000000
        else:
            data_len = len(self.data_list) * self.repeat
        return data_len

    @staticmethod
    def read_file(filename, shuffle):
        """
        :param filename:
        :param shuffle:
        :return:
        """
        content = file_processing.read_data(filename)
        if shuffle:
            random.seed(100)
            random.shuffle(content)

        return content

    def get_label_list(self, label_index=3):
        labels_list = []
        for item in self.data_list:
            label_id = item[label_index]
            labels_list.append(label_id)
        return labels_list

    @staticmethod
    def read_image(path, mode='RGB'):
        '''
        读取图片的函数
        :param path:
        :param mode: RGB or L
        :return:
        '''
        try:
            image = image_processing.read_image(path, colorSpace=mode)
            # image = Image.open(path).convert('RGB')
            # image_processing.show_image("test", image)
        except Exception as e:
            print(e)
            image = None
        return image

    def data_preproccess(self, image):
        """
        数据预处理
        :param image:
        :return:
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        image = self.transform(image)
        return image

    def get_classes_weights(self):
        labels_list = self.get_label_list(label_index=3)
        weight = balanced_classes.create_sample_weight_torch(labels_list)
        return weight


if __name__ == "__main__":
    filename = "../../data/list/train_list.txt"
    image_dir = "/media/dm/dm/FaceRecognition/anti-spoofing/CASIA-SURF/phase1"
    input_size = [112, 112]
    train_transform = transforms.Compose([
        transforms.Resize((input_size[0], input_size[1])),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    PIN_MEMORY = True
    NUM_WORKERS = 2
    DROP_LAST = True
    shuffle = False
    modal = ["color", "depth"]
    dataset_train = MultiModalDataset(filename, image_dir, input_type=modal, transform=train_transform, shuffle=shuffle)
    sampler = None
    batch_size = 2
    weights = dataset_train.get_classes_weights()
    weights = torch.DoubleTensor(weights)
    # 自定义从数据集中取样本的策略，如果指定这个参数，那么shuffle必须为False
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    dataloader = DataLoader(dataset_train, batch_size, sampler=sampler, pin_memory=PIN_MEMORY,
                            num_workers=NUM_WORKERS, drop_last=DROP_LAST, shuffle=shuffle)

    for data in iter(dataloader):
        batch_image = data["depth"]
        batch_label = data["label"]
        image = batch_image[0, :]
        image = image.numpy()  #
        image = np.array(image, dtype=np.float32)
        image = image.transpose(1, 2, 0)  # 通道由[c,h,w]->[h,w,c]
        print("batch_image.shape:{},batch_label:{}".format(batch_image.shape, batch_label))
        image_processing.cv_show_image("image", image)
        # batch_x, batch_y = Variable(batch_x), Variable(batch_y)
