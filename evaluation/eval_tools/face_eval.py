# -*-coding: utf-8 -*-
"""
    @Project: python-learning-notes
    @File   : face_eval.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-07-10 09:46:40
"""
import os, sys

sys.path.append(os.path.dirname(__file__))
import numpy as np
import torch
import torchvision.transforms as transforms

import itertools
import PIL.Image as Image
from evaluation.eval import pr
from evaluation.eval.iou import cal_iou_list
from models.core import face_recognition, face_detection
from utils import image_processing, file_processing
from scipy import interpolate
import scipy.io as sio
import yaml
from evaluation.eval.verification import evaluate
from utils import util


def cal_face_recognition(pred_bboxes, pred_labels, true_bboxes, true_labels, iou_threshold):
    '''
    iou_mat shape=(num_pred_bboxes,num_true_bboxes)
    :param pred_bboxes:
    :param pred_labels:
    :param true_bboxes:
    :param true_labels:
    :param iou_threshold:
    :param score_threshold:
    :return:
    '''
    num_pred_bboxes = len(pred_bboxes)
    num_true_bboxes = len(true_bboxes)
    iou_mat = []
    for pred_bbox in pred_bboxes:
        iou = cal_iou_list(pred_bbox, true_bboxes)
        iou_mat.append(iou)
    iou_mat = np.asarray(iou_mat)
    # print(iou_mat)
    max_index = np.argmax(iou_mat, axis=1)
    max_iou = np.max(iou_mat, axis=1)
    # print(max_index)
    # print(max_iou)
    _true_labels = np.asarray(true_labels)[max_index]
    # print(_true_labels)
    tp = get_tp(pred_labels, _true_labels.tolist(), max_iou, iou_threshold=iou_threshold)
    fp = num_pred_bboxes - tp
    precision = tp / num_pred_bboxes
    recall = tp / num_true_bboxes
    print("precision:{}".format(precision))
    print("recall   :{}".format(recall))
    return precision, recall


def get_face_precision_recall_acc(true_labels, pred_labels, average="binary"):
    recision, recall, acc = pr.get_precision_recall_acc(true_labels, pred_labels, average)
    return recision, recall, acc


def get_tp(pred_labels, true_labels, iou_list, iou_threshold):
    assert isinstance(pred_labels, list), "must be list"
    assert isinstance(true_labels, list), "must be list"
    tp = 0
    for iou, pred_label, true_label in zip(iou_list, pred_labels, true_labels):
        if iou > iou_threshold and pred_label == true_label:
            tp += 1
    return tp


def split_data(data):
    '''
    按照奇偶项分割数据
    :param data:
    :return:
    '''
    data1 = data[0::2]
    data2 = data[1::2]
    return data1, data2


def get_pair_scores_for_bin(faces_data, issames_data, model_path, backbone_name, input_size, embedding_size,
                            save_path=None, eval_prefix="",
                            width_mult=1.0, alignment=False):
    '''
    计算分数
    :param faces_data:
    :param issames_data:
    :param model_path: insightFace模型路径
    :param conf:
    :param save_path:
    :return:
    '''
    detect_model = "mtcnn"
    # 初始化人脸检测
    face_detect = face_detection.FaceDetection(detect_model=detect_model)

    device = "cuda:0"
    multi_gpu = False
    faces_list1, faces_list2 = split_data(faces_data)
    faceRec = face_recognition.FaceRecognition(model_path, backbone_name, input_size, embedding_size, device, multi_gpu,
                                               width_mult)
    pred_score = []
    i = 0
    for face1, face2, issame in zip(faces_list1, faces_list2, issames_data):
        # pred_id, pred_scores = faceRec.predict(faces)
        # 或者使用get_faces_embedding()获得embedding，再比较compare_embedding()
        if alignment:
            face1 = face_detect.face_alignment([face1], face_resize=[112, 112])[0]
            face2 = face_detect.face_alignment([face2], face_resize=[112, 112])[0]
        face_emb1 = faceRec.get_faces_embedding([face1])
        face_emb2 = faceRec.get_faces_embedding([face2])
        # 0.06317729
        score = face_recognition.CompareEmbedding.compare_embedding_scores(face_emb1, face_emb2)
        pred_score.append(score)
        i += 1
        if i % 3000 == 0 or i == len(faces_list1) - 1:
            print('processing data :{}/{}'.format(i, len(faces_list1)))

    pred_score = np.array(pred_score).reshape(-1)
    issames_data = issames_data + 0  # 将true和false转为1/0
    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        issames_path = os.path.join(save_path, "{}_issames.npy".format(eval_prefix))
        pred_score_path = os.path.join(save_path, "{}_pred_score.npy".format(eval_prefix))
        np.save(issames_path, issames_data)
        np.save(pred_score_path, pred_score)
    return pred_score, issames_data


def perform_val_kfold(faces_data, issames_data, model_path, backbone_name, input_size, embedding_size,
                      save_path=None, eval_prefix="",
                      width_mult=1.0, alignment=False, info=""):
    '''
       计算分数
       :param faces_data:
       :param issames_data:
       :param model_path: insightFace模型路径
       :param conf:
       :param save_path:
       :return:
       '''
    detect_model = "mtcnn"
    # 初始化人脸检测
    face_detect = face_detection.FaceDetection(detect_model=detect_model)

    device = "cuda:0"
    multi_gpu = False
    faces_list1, faces_list2 = split_data(faces_data)
    faceRec = face_recognition.FaceRecognition(model_path, backbone_name, input_size, embedding_size, device, multi_gpu,
                                               width_mult)
    pred_score = []
    embeddings = []
    i = 0
    for face1, face2, issame in zip(faces_list1, faces_list2, issames_data):
        # pred_id, pred_scores = faceRec.predict(faces)
        # 或者使用get_faces_embedding()获得embedding，再比较compare_embedding()
        if alignment:
            face1 = face_detect.face_alignment([face1], face_resize=[112, 112])[0]
            face2 = face_detect.face_alignment([face2], face_resize=[112, 112])[0]
        face_emb1 = faceRec.get_faces_embedding([face1])
        face_emb2 = faceRec.get_faces_embedding([face2])
        embeddings.append(face_emb1.cpu())
        embeddings.append(face_emb2.cpu())
    embeddings = np.concatenate(embeddings, axis=0)
    tpr, fpr, accuracy, best_thresholds = evaluate(embeddings, issames_data, nrof_folds=10)
    buf = util.gen_plot(fpr, tpr)
    roc_curve = Image.open(buf)
    roc_curve_tensor = transforms.ToTensor()(roc_curve)

    print("=" * 50)
    info += "best_threshold:{:.4f}\n".format(best_thresholds.mean())
    info += "accuracy:{:.4f}\n".format(accuracy.mean())
    info += "max accuracy:{:.4f}\n".format(max(accuracy))
    print(info)
    print("=" * 50)
    filename = os.path.join(save_path, "result.txt")
    file_processing.write_list_data(filename, [info], mode="w")

    return accuracy.mean(), best_thresholds.mean(), roc_curve_tensor


def get_pair_scores_for_image(faces_list1, faces_list2, issames_data, model_path, backbone_name, input_size,
                              embedding_size,
                              image_dir=None, save_path=None, eval_prefix="", width_mult=1.0, alignment=True):
    '''
    计算分数
    :param faces_data:
    :param issames_data:
    :param model_path: insightFace模型路径
    :param conf:
    :param save_path:
    :return:
    '''
    detect_model = "mtcnn"
    # 初始化人脸检测
    face_detect = face_detection.FaceDetection(detect_model=detect_model)

    device = "cuda:0"
    multi_gpu = False
    faceRec = face_recognition.FaceRecognition(model_path, backbone_name, input_size, embedding_size, device, multi_gpu,
                                               width_mult)
    pred_score = []
    i = 0
    for face1_path, face2_path, issame in zip(faces_list1, faces_list2, issames_data):
        # pred_id, pred_scores = faceRec.predict(faces)
        # 或者使用get_faces_embedding()获得embedding，再比较compare_embedding()
        if image_dir:
            face1_path = os.path.join(image_dir, face1_path)
            face2_path = os.path.join(image_dir, face2_path)
        face1 = image_processing.read_image_gbk(face1_path)
        face2 = image_processing.read_image_gbk(face2_path)
        if face1 is None:
            print("ERROR:{}".format(face1_path))
        if face2 is None:
            print("ERROR:{}".format(face2_path))
        face1 = image_processing.resize_image(face1, resize_height=input_size[0], resize_width=input_size[1])
        face2 = image_processing.resize_image(face2, resize_height=input_size[0], resize_width=input_size[1])
        if alignment:
            face1 = face_detect.face_alignment([face1], face_resize=[112, 112])[0]
            face2 = face_detect.face_alignment([face2], face_resize=[112, 112])[0]
        face_emb1 = faceRec.get_faces_embedding([face1])
        face_emb2 = faceRec.get_faces_embedding([face2])
        diff = face_emb1 - face_emb2
        scores = torch.sum(torch.pow(diff, 2), dim=1)
        scores = scores.detach().cpu().numpy()
        scores = face_recognition.CompareEmbedding.get_scores(scores)
        pred_score.append(scores)
        i += 1
        # if i % 100 == 0:
        #     print('processing data :', i)
    print("processing done ...:")
    pred_score = np.array(pred_score).reshape(-1)
    if not isinstance(issames_data, np.ndarray):
        issames_data = np.asarray(issames_data, dtype=np.int)
    issames_data = issames_data  # 将true和false转为1/0
    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        issames_path = os.path.join(save_path, "{}_issames.npy".format(eval_prefix))
        pred_score_path = os.path.join(save_path, "{}_pred_score.npy".format(eval_prefix))
        np.save(issames_path, issames_data)
        np.save(pred_score_path, pred_score)
    return pred_score, issames_data


def load_npy(dir_path, eval_prefix):
    issames_path = os.path.join(dir_path, "{}_issames.npy".format(eval_prefix))
    pred_score_path = os.path.join(dir_path, "{}_pred_score.npy".format(eval_prefix))
    issames = np.load(issames_path)
    pred_score = np.load(pred_score_path)
    return pred_score, issames


def get_combinations_pair_data(image_dir):
    '''
    get image_dir image list,combinations image
    :param image_dir:
    :return:
    '''
    image_list = file_processing.get_files_list(image_dir, postfix=["*.jpg"])

    pair_issame = []
    for paths in itertools.combinations(image_list, 2):
        image_path1, image_path2 = paths
        name1 = os.path.basename(image_path1)
        name2 = os.path.basename(image_path2)

        label1 = image_path1.split(os.sep)[-2]
        label2 = image_path2.split(os.sep)[-2]
        if label1 == label2:
            issame = 1
        else:
            issame = 0
        image_id1 = os.path.join(label1, name1)
        image_id2 = os.path.join(label2, name2)
        # pair_issame.append([image_id1, image_id2, issame])
        pair_issame.append([image_path1, image_path2, issame])
    pair_issame = np.asarray(pair_issame)
    pair_issame = pair_issame[np.lexsort(pair_issame.T)]
    pair_issame_0 = pair_issame[pair_issame[:, -1] == "0", :]
    pair_issame_1 = pair_issame[pair_issame[:, -1] == "1", :]
    num_pair_issame_1 = len(pair_issame_1)
    per = np.random.permutation(pair_issame_0.shape[0])[:num_pair_issame_1]  # 打乱后的行号
    pair_issame_0 = pair_issame_0[per, :]  # 获取打乱后的训练数据

    pair_issame = np.concatenate([pair_issame_0, pair_issame_1], axis=0)
    image_list1 = pair_issame[:, 0]
    image_list2 = pair_issame[:, 1]
    issame_list = pair_issame[:, 2]
    print("have images:{},combinations :{} pairs".format(len(image_list), len(pair_issame)))
    return image_list1, image_list2, issame_list


def face_far_tar(fpr, tpr, threshold, cache_dir=None):
    fpr_levels = [0.0001, 0.001, 0.01, 0.1]
    f_interp = interpolate.interp1d(fpr, tpr) # default
    # 'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'
    # f_interp = interpolate.interp1d(fpr, tpr,kind="slinear")
    tpr_at_fpr = [float(f_interp(x)) for x in fpr_levels]
    result = ""
    for (far, tar) in zip(fpr_levels, tpr_at_fpr):
        result += 'TAR @ FAR=%.4f : %.4f\n' % (far, tar)

    res = {}
    res['TAR'] = tpr_at_fpr
    res['FAR'] = fpr_levels
    if not cache_dir is None:
        with open(os.path.join(cache_dir, 'result_tar_far.yaml'), 'w') as f:
            yaml.dump(res, f, default_flow_style=False)
        sio.savemat(os.path.join(cache_dir, 'fpr_tpr_threshold.mat'),
                    {'fpr': fpr, 'tpr': tpr, 'thresholds': threshold,
                     'tpr_at_fpr': tpr_at_fpr})
    return result


if __name__ == "__main__":
    # fpr = [0.1, 0.2, 0.3, 0.4, 0.5]
    # tpr = [0.9, 0.8, 0.7, 0.6, 0.5]
    # threshold = [0.7, 0.8, 0.9]
    # face_far_tar(fpr, tpr, threshold, cache_dir=None)
    pass
