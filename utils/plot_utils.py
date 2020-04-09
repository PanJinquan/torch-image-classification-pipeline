# -*-coding: utf-8 -*-
"""
    @Project: python-learning-notes
    @File   : plot_utils.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-07-13 16:30:10
"""
# 导入需要用到的库
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image


def plot_bar(x_data, y_data, title, xlabel, ylabel, isshow=False):
    # 准备数据
    # 用 Matplotlib 画条形图
    plt.bar(x_data, y_data)
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # 设置横纵坐标的名称以及对应字体格式
    font = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 10,
            }
    plt.xlabel(xlabel, font)
    plt.ylabel(ylabel, font)

    plt.title(title)
    plt.legend(loc="lower right")  # "upper right"
    # plt.legend(loc="upper right")#"upper right"
    plt.grid(True)  # 显示网格;
    plt.savefig('out.png')
    if isshow:
        plt.show()


def plot_multi_line(x_data_list, y_data_list, line_names=None, title="", xlabel="", ylabel=""):
    # 绘图
    # plt.figure()
    lw = 2
    plt.figure(figsize=(10, 10))
    colors = ["b", "r", "c", "m", "g", "y", "k", "w"]
    xlim_max = 0
    ylim_max = 0

    xlim_min = 0
    ylim_min = 0
    if not line_names:
        line_names = " " * len(x_data_list)
    for x, y, color, line_name in zip(x_data_list, y_data_list, colors, line_names):
        plt.plot(x, y, color=color, lw=lw, label=line_name)  # 假正率为横坐标，真正率为纵坐标做曲线
        if xlim_max < max(x):
            xlim_max = max(x)
        if ylim_max < max(y):
            ylim_max = max(y)
        if xlim_min > min(x):
            xlim_min = min(x)
        if ylim_min > min(y):
            ylim_min = min(y)
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.plot([0, 1], [1, 0], color='navy', lw=lw, linestyle='--')  # 绘制y=1-x的直线
    x_deta = xlim_max - xlim_min
    y_deta = ylim_max - ylim_min
    plt.xlim([xlim_min - 0.01 * x_deta, xlim_max + 0.1 * x_deta])
    plt.ylim([ylim_min - 0.01 * y_deta, ylim_max + 0.1 * y_deta])
    # 设置横纵坐标的名称以及对应字体格式
    font = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 20,
            }
    plt.xlabel(xlabel, font)
    plt.ylabel(ylabel, font)

    plt.title(title)
    plt.legend(loc="lower right")  # "upper right"
    # plt.legend(loc="upper right")#"upper right"
    plt.grid(True)  # 显示网格;
    plt.show()
