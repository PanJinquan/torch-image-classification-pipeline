# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Project: torch-Face-Recognize-Pipeline
# @File   : dali_datasets.py
# @Author : panjq
# @Date   : 2019-9-20 13:18:34
# --------------------------------------------------------
"""

import os
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import distributed, DataLoader
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, DALIGenericIterator
from utils import file_processing


class HybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, world_size, rank, rgb_mean, rgb_std,
                 dali_cpu=False, shuffle=True):
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        self.num_class = len(file_processing.get_sub_directory_list(data_dir))
        self.world_size = world_size
        self.input = ops.FileReader(file_root=data_dir, shard_id=rank, num_shards=world_size, random_shuffle=shuffle)
        # let user decide which pipeline works him bets for RN version he runs
        dali_device = 'cpu' if dali_cpu else 'gpu'
        decoder_device = 'cpu' if dali_cpu else 'mixed'
        # This padding sets the size of the internal nvJPEG buffers to be able to handle all images from full-sized
        # ImageNet without additional reallocations
        device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
        host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
        self.decode = ops.ImageDecoderRandomCrop(device=decoder_device,
                                                 output_type=types.RGB,
                                                 device_memory_padding=device_memory_padding,
                                                 host_memory_padding=host_memory_padding,
                                                 random_aspect_ratio=[0.8, 1.0],
                                                 random_area=[0.1, 1.0],
                                                 num_attempts=100)
        # self.decode = ops.ImageDecoder(device=decoder_device, 
        #                                 output_type=types.RGB,
        #                                 device_memory_padding=device_memory_padding,
        #                                 host_memory_padding=host_memory_padding)
        # self.rrc = ops.RandomResizedCrop(device=dali_cpu,
        #                                 size=crop,
        #                                 minibatch_size=batch_size,
        #                                 num_attempts=100)
        self.resize = ops.Resize(device=dali_device,
                                 resize_x=int(128 * crop / 112),
                                 resize_y=int(128 * crop / 112),
                                 interp_type=types.INTERP_TRIANGULAR)
        # self.flip = ops.Flip(device=dali_device)  # RandomHorizontalFlip
        self.cmnp = ops.CropMirrorNormalize(device=dali_device,
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mean=[255 * a for a in rgb_mean],
                                            std=[255 * a for a in rgb_std])
        self.coin = ops.CoinFlip(probability=0.5)
        print('DALI "{0}" variant'.format(dali_device))

    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.resize(images)
        # images = self.flip(images)
        output = self.cmnp(images, mirror=self.coin())
        return [output, self.labels]

    @property
    def num_imgs(self):
        return self.epoch_size("Reader")


class DaliDataset():
    def __init__(self, image_dir, batch_size, num_workers, world_size, device_id, rank, crop_size, rgb_mean, rgb_std,
                 shuffle=True):
        self.image_dir = image_dir
        self.device_id = device_id
        self.rank = rank
        self.crop_size = crop_size
        self.world_size = world_size
        self.rgb_mean = rgb_mean
        self.rgb_std = rgb_std
        self.batch_size = batch_size
        self.pip_train = HybridTrainPipe(batch_size=batch_size,
                                         num_threads=num_workers,
                                         device_id=self.device_id,
                                         data_dir=self.image_dir,
                                         crop=self.crop_size,
                                         world_size=self.world_size,
                                         rank=self.rank,
                                         rgb_mean=self.rgb_mean,
                                         rgb_std=self.rgb_std,
                                         shuffle=shuffle)
        self.pip_train.build()
        # self.num_imgs = self.pip_train.epoch_size("Reader") // self.world_size
        # self.num_imgs = self.pip_train.get_num_imgs()
        # self.num_class = self.pip_train.get_num_class()
        print("have:num_class:{},num_imgs:{}".format(self.num_class, self.num_imgs))

    @property
    def num_imgs(self):
        return self.pip_train.num_imgs

    @property
    def num_class(self):
        return self.pip_train.num_class

    # def get_num_class(self):
    #     return self.num_class

    # def get_num_imgs(self):
    #     return self.num_imgs

    def get_dataloader(self):
        def get_len(self):
            return self._size // self.batch_size

        DALIClassificationIterator.__len__ = get_len
        train_loader = DALIClassificationIterator(self.pip_train,
                                                  size=self.pip_train.epoch_size("Reader") // self.world_size,
                                                  auto_reset=True)
        #         import ipdb
        #         ipdb.set_trace()
        train_loader.num_samples = train_loader._size
        train_loader.batch_size = self.batch_size
        return train_loader


if __name__ == "__main__":
    # Use nvidia DALI dataloader:
    BATCH_SIZE = 8
    DATA_ROOT = ["/media/dm/dm/XMC/torch-Face-Recognize-Pipeline/data/dataset"]
    # DATA_ROOT = ["/media/dm/dm/XMC/torch-Face-Recognize-Pipeline/data/dataset1"]
    VAL_ROOT = ""
    NUM_WORKERS = 8
    GPU_ID = [0]
    INPUT_SIZE = [112, 112]
    RGB_MEAN = [0.5, 0.5, 0.5]
    RGB_STD = [0.5, 0.5, 0.5]
    device = "cuda:0"
    shuffle = False
    NUM_EPOCH = 4
    dali = DaliDataset(image_dir=DATA_ROOT[0],
                       batch_size=BATCH_SIZE,
                       num_workers=NUM_WORKERS,
                       device_id=GPU_ID[0],
                       world_size=1,
                       crop_size=INPUT_SIZE[0],
                       rgb_mean=RGB_MEAN,
                       rgb_std=RGB_STD,
                       shuffle=shuffle)
    num_class = dali.get_num_class()
    num_imgs = dali.get_num_imgs()
    print("have:num_class:{},num_imgs:{}".format(num_class, num_imgs))
    train_loader = dali.get_dataloader()
    print(len(train_loader))
    step = 0
    for epoch in range(NUM_EPOCH):  # start training process
        for data in train_loader:
            batch_label = data[0]['label'].squeeze().long().cpu().numpy()
            batch_image = data[0]['data'].cpu().numpy()
            image = batch_image[0, :]
            # image = image.numpy()  #
            image = np.array(image, dtype=np.float32)
            image = image.transpose(1, 2, 0)  # 通道由[c,h,w]->[h,w,c]
            print("epoch:{}/{},batch_image.shape:{},batch_label:{}".format(epoch, len(train_loader) * NUM_EPOCH,
                                                                           batch_image.shape, batch_label))
            # image_processing.cv_show_image("image1", image)
            # batch_x, batch_y = Variable(batch_x), Variable(batch_y)
