# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Project: torch-Face-Recognize-Pipeline
# @File   : custom_transform.py
# @Author : panjq
# @Date   : 2019-6-20 13:18:34
# --------------------------------------------------------
"""

import os
import torch
import random
import PIL.Image as Image
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from models.dataloader import imagefolder_dataset
from utils import image_processing
import cv2


class RandomResize(object):
    """ random resize images"""

    def __init__(self, resize_range, interpolation=Image.BILINEAR):
        """
        :param resize_range: range size range
        :param interpolation:
        """
        self.interpolation = interpolation
        self.resize_range = resize_range

    def __call__(self, img):
        r = int(random.uniform(self.resize_range[0], self.resize_range[1]))
        size = (r, r)
        # print("RandomResize:{}".format(size))
        return transforms.functional.resize(img, size, self.interpolation)

    def __repr__(self):
        interpolation = transforms._pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolation)


class GaussianBlur(object):
    """Gaussian Blur for image"""

    def __init__(self):
        pass

    def __call__(self, img, ksize=(3, 3), sigmaX=0):
        img = np.asarray(img)
        img = cv2.GaussianBlur(img, ksize, sigmaX)
        img = Image.fromarray(img)
        return img


class RandomGaussianBlur(object):
    """Random Gaussian Blur for image"""

    def __init__(self, ksize_range=(0, 1, 1, 3, 3, 5), sigmaX=0):
        """
        :param ksize_range: Gaussian kernel size. ksize.width and ksize.height can differ but they both must be
    .   positive and odd. Or, they can be zero's and then they are computed from sigma.
        :param sigmaX:
        """
        self.ksize_range = ksize_range
        self.sigmaX = sigmaX

    def __call__(self, img):
        index = int(random.uniform(0, len(self.ksize_range)))
        r = self.ksize_range[index]
        # print(r)
        if r > 0:
            ksize = (r, r)
            img = np.asarray(img)
            img = cv2.GaussianBlur(img, ksize, self.sigmaX)
            img = Image.fromarray(img)
        return img


def custom_transform(input_size, RGB_MEAN, RGB_STD, transform_type):
    '''
    :param input_size:
    :param RGB_MEAN:
    :param RGB_STD:
    :param transform_type: [default,scale20_50,scale30]
    :return:
    '''
    if "scale" in transform_type:
        resize_range = transform_type[len('scale'):].split("_")
        resize_range = (int(resize_range[0]), int(resize_range[1]))
        transform = transforms.Compose([
            RandomResize(resize_range),
            transforms.Resize([int(128 * input_size[0] / 112), int(128 * input_size[0] / 112)]),
            transforms.RandomCrop([input_size[0], input_size[1]]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=RGB_MEAN, std=RGB_STD),
        ])

    elif transform_type == "default":
        transform = transforms.Compose([
            transforms.Resize([int(128 * input_size[0] / 112), int(128 * input_size[0] / 112)]),
            transforms.RandomCrop([input_size[0], input_size[1]]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=RGB_MEAN, std=RGB_STD),
        ])
    else:
        raise Exception("transform_type ERROR:{}".format(transform_type))
    return transform


def kd_transform(input_size, RGB_MEAN, RGB_STD, transform_type):
    '''
    :param input_size:
    :param RGB_MEAN:
    :param RGB_STD:
    :param transform_type: [default,scale20_50,scale30]
    :return:
    '''
    if "scale" in transform_type:
        resize_range = transform_type[len('scale'):].split("_")
        resize_range = (int(resize_range[0]), int(resize_range[1]))
        transform = transforms.Compose([
            RandomResize(resize_range),
            transforms.Resize([input_size[0], input_size[1]]),
            # GaussianBlur(),
            RandomGaussianBlur(),
            transforms.ToTensor(),
            transforms.Normalize(mean=RGB_MEAN, std=RGB_STD),
        ])

    elif transform_type == "default":
        transform = transforms.Compose([
            transforms.Resize([input_size[0], input_size[1]]),
            transforms.ToTensor(),
            transforms.Normalize(mean=RGB_MEAN, std=RGB_STD),
        ])
    elif transform_type == "comment":
        transform = transforms.Compose([
            transforms.Resize([int(128 * input_size[0] / 112), int(128 * input_size[0] / 112)]),
            transforms.RandomCrop([input_size[0], input_size[1]]),
            transforms.RandomHorizontalFlip(),
        ])
    else:
        raise Exception("transform_type ERROR:{}".format(transform_type))
    return transform


def make_weights_for_balanced_classes(images, nclasses):
    """
    Make a vector of weights for each image in the dataset, based
    on class frequency. The returned vector of weights can be used
    to create a WeightedRandomSampler for a DataLoader to have
    class balancing when sampling for a training batch.
    https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3
    :param images:  torchvisionDataset.imgs
    :param nclasses: len(torchvisionDataset.classes)
    :return:
    """
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1  # item is (img-data, label-id)
    weight_per_class = [0.] * nclasses
    N = float(sum(count))  # total number of images
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]

    return weight


def custom_transform_test():
    image_path = "/media/dm/dm1/XMC/tf-Face-Recognize-Pipeline/data/dataset/0/1.jpg"
    image = image_processing.read_image(image_path)
    input_size = [112, 112]
    train_transform = custom_transform(input_size, RGB_MEAN=[0.5, 0.5, 0.5], RGB_STD=[0.5, 0.5, 0.5],
                                       transform_type="default")
    image = Image.fromarray(image)
    image = train_transform(image)
    image = np.array(image, dtype=np.float32)
    image = image.transpose(1, 2, 0)  # 通道由[c,h,w]->[h,w,c]
    image_processing.cv_show_image("image", image)
    # batch_x, batch_y = Variable(batch_x), Variable(batch_y)


if __name__ == '__main__':
    custom_transform_test()
    # image_dir1 = "/media/dm/dm1/project/InsightFace_Pytorch/custom_insightFace/data/facebank"
    image_dir2 = "/media/dm/dm1/XMC/torch-Face-Recognize-Pipeline/data/dataset"
    # 图像预处理Rescale，RandomCrop，ToTensor
    input_size = [112, 112]
    image_dir_list = [image_dir2]
    train_transform = custom_transform(input_size, RGB_MEAN=[0.5, 0.5, 0.5], RGB_STD=[0.5, 0.5, 0.5],
                                       transform_type="default")
    PIN_MEMORY = True
    NUM_WORKERS = 2
    DROP_LAST = True
    dataset_train = datasets.ImageFolder(image_dir2, transform=train_transform)
    # dataset_train = imagefolder_dataset.ImageFolderDataset(image_dir_list=image_dir_list, transform=train_transform)

    print("num images:{},num classs:{}".format(len(dataset_train.imgs), len(dataset_train.classes)))
    weights = imagefolder_dataset.make_weights_for_balanced_classes(dataset_train.imgs, len(dataset_train.classes))
    weights = torch.DoubleTensor(weights)
    # 自定义从数据集中取样本的策略，如果指定这个参数，那么shuffle必须为False
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    # dataloader = DataLoader(dataset_train, batch_size=8, sampler=sampler, pin_memory=PIN_MEMORY, num_workers=NUM_WORKERS, drop_last=DROP_LAST, shuffle=False)
    dataloader = DataLoader(dataset_train, batch_size=1, pin_memory=PIN_MEMORY, num_workers=NUM_WORKERS,
                            drop_last=DROP_LAST, shuffle=False)

    for batch_image, batch_label in iter(dataloader):
        image = batch_image[0, :]
        # image = image.numpy()  #
        image = np.array(image, dtype=np.float32)
        image = image.transpose(1, 2, 0)  # 通道由[c,h,w]->[h,w,c]
        print("batch_image.shape:{},batch_label:{}".format(batch_image.shape, batch_label))
        image_processing.cv_show_image("image", image)
        # batch_x, batch_y = Variable(batch_x), Variable(batch_y)
