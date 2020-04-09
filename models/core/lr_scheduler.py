# -*- coding: utf-8 -*-
"""
# --------------------------------------------------------
# @Project: torch-anti-spoofing-Pipeline
# @Author : panjq
# @E-mail : pan_jinquan@163.com
# @Date   : 2020-04-09 11:07:42
# --------------------------------------------------------
"""

from torch import optim


def warm_up_lr(optimizer, step, num_step_warm_up, init_lr):
    """
    Learning Rate warm up
    :param step:
    :param num_step_warm_up:
    :param init_lr:
    :param optimizer:
    :return:
    """
    for params in optimizer.param_groups:
        params['lr'] = step * init_lr / num_step_warm_up
    lr = step * init_lr / num_step_warm_up
    return lr


def step_lr(optimizer, step_size, gamma=0.1):
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    return scheduler


def multi_step_lr(optimizer, milestones, gamma=0.1):
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               milestones=milestones,
                                               gamma=gamma)
    return scheduler


if __name__ == "__main__":
    from torchvision import models
    import matplotlib.pyplot as plt

    lr = 0.1
    device = "cuda:0"
    model = models.resnet18(pretrained=False, num_classes=2).to(device)
    optimizer = optim.SGD(model.parameters(),
                          lr=lr,
                          weight_decay=5e-4,
                          momentum=0.9)

    milestones = [20, 40, 60, 80]
    num_epoch_warm_up = 10
    num_per_epoch = 50
    num_step_warm_up = (num_per_epoch * num_epoch_warm_up)
    scheduler = multi_step_lr(optimizer, milestones, gamma=0.1)
    lr_list = []
    step = 0
    for epoch in range(100):
        scheduler.step()
        for i in range(num_per_epoch):
            if step < num_step_warm_up:
                print("warm_up_lr:{}".format(epoch))
                warm_up_lr(optimizer, step, num_step_warm_up, init_lr=lr)
            optimizer.zero_grad()
            optimizer.step()
            step += 1
        lr_ = optimizer.param_groups[0]["lr"]
        lr_list.append(lr_)
        # print("epoch:{},lr:{}".format(epoch, lr_))
    plt.plot(list(range(len(lr_list))), lr_list, color='r')
    plt.grid(True)  # 显示网格;
    plt.show()
