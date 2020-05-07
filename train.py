# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Project: torch-anti-spoofing-Pipeline
# @Author : panjq
# @E-mail : pan_jinquan@163.com
# @Date   : 2020-04-09 10:54:14
# --------------------------------------------------------
"""

import os
import argparse
import torch
import torch.utils.data as torch_utils
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from models.nets import nets
from models.dataloader import imagefolder_dataset
from models.dataloader import custom_transform
from models.core import lr_scheduler
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from evaluation.eval_tools.metrics import AverageMeter, accuracy
from utils import file_processing, json_utils

print("torch", torch.__version__)


class Trainer(object):
    """ Face Recognize Pipeline """

    def __init__(self, cfg):
        """
        :param cfg: train config
        """
        self.SEED = 2020
        torch.manual_seed(self.SEED)
        torch.cuda.manual_seed(self.SEED)

        self.data_root = cfg["data_root"]
        self.val_root = cfg["val_root"]
        self.work_dir = cfg["work_dir"]
        self.model_name = cfg["model_name"]
        self.resume = cfg["resume"]
        self.input_size = cfg["input_size"]
        self.RGB_MEAN = cfg["RGB_MEAN"]
        self.RGB_STD = cfg["RGB_STD"]
        self.batch_size = cfg["batch_size"]
        self.lr = cfg["lr"]  # initial LR
        self.momentum = cfg["momentum"]
        self.optim_tpye = cfg["optim_tpye"]
        self.num_epoch = cfg["num_epoch"]
        self.resample = cfg["resample"]
        self.num_epoch_warm_up = cfg["num_epoch_warm_up"]
        self.weight_decay = float(cfg["weight_decay"])
        self.stages = cfg["stages"]
        self.gpu_id = cfg["gpu_id"]
        self.num_workers = cfg["num_workers"]
        self.disp_freq = cfg["disp_freq"]
        self.verbose = cfg["verbose"]
        self.config_file = cfg["config_file"]
        self.time = file_processing.get_time()
        self.work_dir = os.path.join(self.work_dir, "{}_{}".format(self.model_name, self.time))
        self.model_root = os.path.join(self.work_dir, "model")
        self.log_root = os.path.join(self.work_dir, "log")
        file_processing.create_dir(self.model_root)
        file_processing.create_dir(self.log_root)
        file_processing.copy_file_to_dir(self.config_file, self.log_root)
        self.writer = SummaryWriter(self.log_root)
        self.val_log = file_processing.WriterTXT(filename=os.path.join(self.log_root, "val_log.yaml"))

        self.device = torch.device("cuda:{}".format(self.gpu_id[0]) if torch.cuda.is_available() else "cpu")

        self.train_dataloader()
        self.val_dataloader()
        self.model = self.build_net(self.model_name, input_size=self.input_size, num_classes=self.num_class)
        self.model.to(self.device)
        self.optimizer = self.get_optimizer(optim_tpye=self.optim_tpye)
        self.loss = nn.CrossEntropyLoss()
        self.lr_scheduler = lr_scheduler.multi_step_lr(self.optimizer, self.stages, gamma=self.lr)

    def train_dataloader(self):
        """
        load train data
        :return:
        """
        train_transform = custom_transform.custom_transform(self.input_size,
                                                            self.RGB_MEAN,
                                                            self.RGB_STD,
                                                            transform_type="train"
                                                            )
        dataset_train = imagefolder_dataset.ImageFolderDataset(self.data_root,
                                                               train_transform)
        # create a weighted random sampler to process imbalanced data
        if self.resample:
            weights = custom_transform.make_weights_for_balanced_classes(dataset_train.imgs, len(dataset_train.classes))
            # weights = dataset_train.get_classes_weights()
            sampler = torch_utils.sampler.WeightedRandomSampler(weights, len(weights))
            shuffle = False  # sampler option is mutually exclusive with shuffle
        else:
            sampler = None
            shuffle = True

        self.train_loader = torch_utils.DataLoader(dataset_train,
                                                   batch_size=self.batch_size,
                                                   sampler=sampler,
                                                   num_workers=self.num_workers,
                                                   shuffle=shuffle)
        self.num_class = len(self.train_loader.dataset.classes)
        self.num_images = len(self.train_loader) * self.batch_size
        print("train num_images :{},class_num:{}".format(self.num_images, self.num_class))

    def val_dataloader(self):
        """
        load val data
        :return:
        """
        val_transform = custom_transform.custom_transform(self.input_size,
                                                          self.RGB_MEAN,
                                                          self.RGB_STD,
                                                          transform_type="val")
        dataset_val = imagefolder_dataset.ImageFolderDataset(self.val_root,
                                                             val_transform)

        self.val_loader = torch_utils.DataLoader(dataset_val,
                                                 batch_size=self.batch_size,
                                                 num_workers=self.num_workers,
                                                 shuffle=False)

        val_num_class = len(self.val_loader.dataset.classes)
        val_num_images = len(self.val_loader) * self.batch_size
        print("val num_images :{},class_num:{}".format(val_num_images, val_num_class))

    def get_optimizer(self, optim_tpye="SGD"):
        """
        :param optim_tpye:
        :return:
        """
        if optim_tpye == "SGD":
            optimizer = optim.SGD(self.model.parameters(),
                                  lr=self.lr,
                                  momentum=self.momentum,
                                  weight_decay=self.weight_decay)
        elif optim_tpye == "Adam":
            betas = (0.5, 0.999)
            optimizer = optim.Adam(self.model.parameters(),
                                   lr=self.lr,
                                   weight_decay=self.weight_decay)
        else:
            raise Exception("Error:{}".format(optim_tpye))
        return optimizer

    def build_net(self, model_name, input_size, num_classes):
        """
        :param model_name:
        :param input_size:
        :param num_classes:
        :return:
        """
        model = nets.build_net(model_name, input_size, num_classes, pretrained=True)
        if len(self.gpu_id) > 1:
            model = nn.DataParallel(model, device_ids=self.gpu_id)
        model = self.resume_model(model)
        return model

    def resume_model(self, model):
        """
        resume or finetune model
        :return:
        """
        self.start_epoch = 0
        self.start_step = 0
        return model

    def train(self):
        """
        train, validation and save checkpoint
        :return:
        """
        self.disp_freq = (len(self.train_loader) // self.disp_freq)
        self.num_step_warm_up = (len(self.train_loader) * self.num_epoch_warm_up)
        print("start_epoch:{},start_step:{}".format(self.start_epoch, self.start_step))
        step = self.start_step
        start_epoch = self.start_epoch
        for epoch in range(start_epoch, self.num_epoch):
            self.lr_scheduler.step()
            step = self.train_step(epoch, step)
            lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar("lr_epoch", lr, epoch)
            self.evaluation(epoch)
            self.save_model(self.model, self.optimizer, self.model_root, self.model_name, epoch, self.gpu_id)

    def train_step(self, epoch, step):
        """
        training loop
        :param epoch:
        :param step:
        :return:
        """
        self.model.train()  # set to training mode
        losses = AverageMeter()
        top1 = AverageMeter()
        with tqdm(total=len(self.train_loader), desc="Train Epoch #{}".format(epoch), disable=not self.verbose) as t:
            for data in self.train_loader:
                step += 1
                # adjust LR for each training batch during warm up
                if step < self.num_step_warm_up:
                    lr_scheduler.warm_up_lr(self.optimizer, step, self.num_step_warm_up, self.lr)
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss(outputs, labels)
                # measure accuracy and record loss
                acc, = accuracy(outputs.data, labels, topk=(1,))
                losses.update(loss.data.item(), inputs.size(0))
                top1.update(acc.data.item(), inputs.size(0))

                # compute gradient and do SGD step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # t.set_postfix({"prec1 ": "{:6.5f}s".format(prec1)})
                t.update(1)
                # dispaly training loss & acc
                if step % self.disp_freq == 0 and step > 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    train_log = "epoch/step: {:0=3}/{} lr: {:.6f} loss: {:.4f} Acc: {:.4f}%".format(epoch,
                                                                                                   step,
                                                                                                   lr,
                                                                                                   losses.avg,
                                                                                                   top1.avg)
                    self.val_log.write_line_str(train_log)
                    print(train_log)
            self.writer.add_scalar("Training_Loss_epoch", losses.avg, epoch)
            self.writer.add_scalar("Training_Accuracy", top1.avg, epoch)
            return step

    def evaluation(self, epoch):
        """
        val data metrics
        :param epoch:
        :return:
        """
        self.model.eval()  # set to training mode
        losses = AverageMeter()
        top1 = AverageMeter()
        with torch.no_grad():
            for data in self.val_loader:
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss(outputs, labels)
                # measure accuracy and record loss
                acc, = accuracy(outputs.data, labels, topk=(1,))
                losses.update(loss.data.item(), inputs.size(0))
                top1.update(acc.data.item(), inputs.size(0))
        val_log = "evaluation-epoch: {:0=3} loss: {:.4f} Acc: {:.4f}".format(epoch, losses.avg, top1.avg)
        print(val_log)
        self.val_log.write_line_str(val_log)
        self.writer.add_scalar("Val_Loss_epoch", losses.avg, epoch)
        self.writer.add_scalar("Val_Accuracy", top1.avg, epoch)
        return None

    @staticmethod
    def save_model(model, optimizer, model_root, model_name, epoch, gpu_id):
        """
        :param model:
        :param optimizer:
        :param model_root:
        :param model_name:
        :param epoch:
        :param gpu_id:
        :return:
        """
        model_file = os.path.join(model_root, "model_{}_{:0=3d}.pth".format(model_name, epoch))
        optimizer_pth = os.path.join(model_root, "optimizer_{}.pth".format(model_name))
        if len(gpu_id) > 1:
            model = model.module
        torch.save(model.state_dict(), model_file)
        torch.save({
            "epoch": epoch,
            "optimizer_state_dict": optimizer.state_dict()},
            optimizer_pth,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="for face verification train")

    parser.add_argument(
        "-c", "--config", help="configs file", default="configs/config.yaml", type=str
    )
    args = parser.parse_args()
    cfg = json_utils.load_config(args.config)
    cfg["config_file"] = args.config
    json_utils.print_dict(cfg)
    t = Trainer(cfg)
    t.train()
