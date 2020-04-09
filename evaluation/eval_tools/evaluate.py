# -*-coding: utf-8 -*-
"""
    @Project: torch-Face-Recognize-Pipeline
    @File   : evaluate.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-12-26 16:32:47
"""
import os
import math
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from evaluation.eval import verification
from utils import bin_tools
from utils import util


class EvaluationMetrics(object):
    def __init__(self, issr=False):
        """
        :param issr: is Super-Resolution(SR) Evaluation,default(False)
        """
        self.issr = issr
        self.model = None

    def update_model(self, model, device, multi_gpu):
        """
        updata model for test
        :param model:
        :param device:
        :param multi_gpu: bool True or False
        :return:
        """
        self.device = device
        if multi_gpu:
            model = model.module  # unpackage model from DataParallel
            self.model = model.to(self.device)
        else:
            self.model = model.to(self.device)
        # switch to evaluation mode
        self.model.eval()

    def forward(self, input_tensor):
        """
        :param input_tensor: input tensor
        :return:
        """
        with torch.no_grad():
            out_tensor = self.model(input_tensor.to(self.device))
            if self.issr:
                de_image, out_tensor = out_tensor
        return out_tensor

    @staticmethod
    def de_preprocess(tensor):
        """
        tensor convert to image
        :param tensor:
        :return:
        """
        return tensor * 0.5 + 0.5

    @staticmethod
    def default_transform(input_size, RGB_MEAN=[0.5, 0.5, 0.5], RGB_STD=[0.5, 0.5, 0.5]):
        """
        人脸识别默认的预处理方法
        :param input_size:resize大小
        :param RGB_MEAN:均值
        :param RGB_STD: 方差
        :return:
        """
        transform = transforms.Compose([
            # EvaluationMetrics.de_preprocess,
            transforms.ToPILImage(),
            transforms.Resize([int(128 * input_size[0] / 112), int(128 * input_size[0] / 112)]),
            transforms.CenterCrop([input_size[0], input_size[1]]),
            transforms.ToTensor(),
            transforms.Normalize(mean=RGB_MEAN, std=RGB_STD),
        ])
        return transform

    @staticmethod
    def batch_transform(imgs_tensor, input_size):
        """
        :param imgs_tensor:
        :param input_size:
        :return:
        """
        ccrop = EvaluationMetrics.default_transform(input_size,
                                                    RGB_MEAN=[0.5, 0.5, 0.5],
                                                    RGB_STD=[0.5, 0.5, 0.5])
        ccropped_imgs = torch.empty_like(imgs_tensor)
        for i, img_ten in enumerate(imgs_tensor):
            ccropped_imgs[i] = ccrop(img_ten)
        return ccropped_imgs

    @staticmethod
    def hflip_transform(input_size, RGB_MEAN=[0.5, 0.5, 0.5], RGB_STD=[0.5, 0.5, 0.5]):
        """
        hflip预处理方法
        :param input_size:resize大小
        :param RGB_MEAN:均值
        :param RGB_STD: 方差
        :return:
        """
        transform = transforms.Compose([
            # EvaluationMetrics.de_preprocess,
            transforms.ToPILImage(),
            transforms.functional.hflip,
            transforms.Resize([int(128 * input_size[0] / 112), int(128 * input_size[0] / 112)]),
            transforms.CenterCrop([input_size[0], input_size[1]]),
            transforms.ToTensor(),
            transforms.Normalize(mean=RGB_MEAN, std=RGB_STD),
        ])
        return transform

    @staticmethod
    def batch_hflip_transform(imgs_tensor, input_size):
        """
        flipped image
        :param imgs_tensor:
        :return:
        """
        hflip = EvaluationMetrics.hflip_transform(input_size,
                                                  RGB_MEAN=[0.5, 0.5, 0.5],
                                                  RGB_STD=[0.5, 0.5, 0.5])

        hfliped_imgs = torch.empty_like(imgs_tensor)
        for i, img_ten in enumerate(imgs_tensor):
            hfliped_imgs[i] = hflip(img_ten)

        return hfliped_imgs

    @staticmethod
    def pre_process(faces_list, transform):
        """
        :param faces_list:
        :param face_resize:
        :param transform:
        :return:
        """
        imgs_tensor = np.empty_like(faces_list)
        for i, face in enumerate(faces_list):
            face = torch.tensor(face)
            face = transform(face)
            imgs_tensor[i] = face.unsqueeze(0)  # 增加一个维度
        imgs_tensor = torch.from_numpy(imgs_tensor)
        return imgs_tensor

    @staticmethod
    def post_process(input, axis=1):
        """
        l2_norm
        :param input:
        :param axis:
        :return:
        """
        norm = torch.norm(input, 2, axis, True)
        output = torch.div(input, norm)
        return output

    def get_embedding(self, faces, transform):
        """
        :param faces: faces
        :param transform: torch transform,for processing
        :return:
        """
        faces_tensor = self.pre_process(faces, transform=transform)
        embeddings = self.forward(faces_tensor).cpu()
        return embeddings

    def get_images_embedding(self, images, batch_size, input_size, tta=False):
        """
        :param images: input images
        :param batch_size: batch size
        :param input_size: input size,[112,112] or others
        :param tta:
        :return:
        """
        transform = self.default_transform(input_size,
                                           RGB_MEAN=[0.5, 0.5, 0.5],
                                           RGB_STD=[0.5, 0.5, 0.5])

        hflip_transform = self.hflip_transform(input_size,
                                               RGB_MEAN=[0.5, 0.5, 0.5],
                                               RGB_STD=[0.5, 0.5, 0.5])
        sample_num = len(images)
        batch_num = int(math.ceil(sample_num / batch_size))
        embds = []
        for i in range(batch_num):
            start = i * batch_size
            end = min((i + 1) * batch_size, sample_num)
            batch_image = images[start: end]
            # batch_image = images[i * batch_size:min((i + 1) * batch_size, sample_num)]

            cur_emb = self.get_embedding(batch_image, transform)
            if tta:
                hflip_emb = self.get_embedding(batch_image, hflip_transform)
                cur_emb = cur_emb + hflip_emb
            cur_emb = self.post_process(cur_emb, axis=1)
            embds += list(cur_emb.detach().numpy())
            # embds = cur_emb

        return np.array(embds)

    def metrics(self, images, issame, batch_size, input_size, tta=True):
        """
        get val data metrics result
        :param images:
        :param issame:
        :param batch_size:
        :param input_size:
        :param tta:
        :return:
        """
        embds = self.get_images_embedding(images, batch_size, input_size, tta=tta)
        # tpr, fpr, acc_mean, acc_std, tar, tar_std, far = eval_utils.get_evaluate_report(embds,
        #                                                                                 issame,
        #                                                                                 far_target=1e-3)
        # same as:
        tpr, fpr, accuracy, best_thresholds = verification.evaluate(embds, issame, nrof_folds=10)
        buf = util.gen_plot(fpr, tpr)
        roc_curve = Image.open(buf)
        roc_curve_tensor = transforms.ToTensor()(roc_curve)
        # return acc_mean
        return accuracy.mean(), best_thresholds.mean(), roc_curve_tensor

    @staticmethod
    def perform_val(
            multi_gpu,
            device,
            embedding_size,
            batch_size,
            input_size,
            backbone,
            carray,
            issame,
            nrof_folds=10,
            tta=True
    ):
        """
        :param multi_gpu:
        :param device:
        :param embedding_size:
        :param batch_size:
        :param input_size:
        :param backbone:
        :param carray:
        :param issame:
        :param nrof_folds:
        :param tta:
        :return:
        """
        if multi_gpu:
            backbone = backbone.module  # unpackage model from DataParallel
            backbone = backbone.to(device)
        else:
            backbone = backbone.to(device)
        backbone.eval()  # switch to evaluation mode

        idx = 0
        embeddings = np.zeros([len(carray), embedding_size])
        with torch.no_grad():
            while idx + batch_size <= len(carray):
                # batch = torch.tensor(carray[idx:idx + batch_size][:, [2, 1, 0], :, :])
                batch = torch.tensor(carray[idx:idx + batch_size])

                # image_processing.show_batch_image("image",batch,index=0)
                if tta:
                    ccropped = EvaluationMetrics.batch_transform(batch, input_size)
                    fliped = EvaluationMetrics.batch_hflip_transform(batch, input_size)
                    # inputs = inputs.cuda(device, non_blocking=True)
                    # labels = labels.cuda(device, non_blocking=True).long()
                    emb_batch = backbone(ccropped.to(device)).cpu() + backbone(fliped.to(device)).cpu()
                    embeddings[idx:idx + batch_size] = util.l2_norm(emb_batch)
                else:
                    ccropped = EvaluationMetrics.batch_transform(batch, input_size)
                    embeddings[idx:idx + batch_size] = util.l2_norm(backbone(ccropped.to(device))).cpu()
                idx += batch_size
            if idx < len(carray):
                batch = torch.tensor(carray[idx:])
                if tta:
                    ccropped = EvaluationMetrics.batch_transform(batch, input_size)
                    fliped = EvaluationMetrics.batch_hflip_transform(batch, input_size)
                    emb_batch = backbone(ccropped.to(device)).cpu() + backbone(fliped.to(device)).cpu()
                    embeddings[idx:] = util.l2_norm(emb_batch)
                else:
                    ccropped = EvaluationMetrics.batch_transform(batch, input_size)
                    embeddings[idx:] = util.l2_norm(backbone(ccropped.to(device))).cpu()
        tpr, fpr, accuracy, best_thresholds = util.evaluate(embeddings, issame, nrof_folds)
        buf = util.gen_plot(fpr, tpr)
        roc_curve = Image.open(buf)
        roc_curve_tensor = transforms.ToTensor()(roc_curve)

        return accuracy.mean(), best_thresholds.mean(), roc_curve_tensor

    def metrics_writer(self, writer, epoch, val_dataset, input_size, batch_size, val_data_resize, tta=True):
        '''
        :param writer:
        :param epoch:
        :param val_dataset:
        :param input_size:
        :param batch_size:
        :param val_data_resize:
        :param tta:
        :return:
        '''
        val_acc_info = ""
        for dataset in val_dataset:
            name = dataset["name"]
            faces = dataset["faces"]
            issame = dataset["issame"]
            name_list, val_accuracys_list, best_threshold_list, roc_curve_list = (
                [],
                [],
                [],
                [],
            )

            for val_size in val_data_resize:
                faces, issame = util.reszie_val_face_data(faces, issame, resize=[val_size, val_size])
                faces, issame = util.reszie_val_face_data(faces, issame, resize=input_size)
                val_accuracy, best_threshold, roc_curve = self.metrics(faces, issame, batch_size, input_size, tta=tta)
                val_name = "{}_{}".format(name, val_size)
                name_list.append(val_name)
                val_accuracys_list.append(val_accuracy)
                best_threshold_list.append(best_threshold)
                roc_curve_list.append(roc_curve)
                val_acc_info += "_{}_{:.3f}".format(val_name, val_accuracy)
            util.buffer_val_list(
                writer,
                name,
                name_list,
                val_accuracys_list,
                best_threshold_list,
                roc_curve_list,
                epoch,
            )
        return val_acc_info


def perform_val_test(val_root, dataset, model_path, backbone_name, embedding_size, input_size, width_mult):
    '''
    :param val_root:
    :param dataset:
    :param model_path:
    :param backbone_name:
    :param embedding_size:
    :param input_size:
    :param width_mult:
    :return:
    '''
    from models.nets import nets
    device = "cuda:0"
    MULTI_GPU = False
    BATCH_SIZE = 64
    tta = True
    BACKBONE = nets.build_net(backbone_name=backbone_name,
                              embedding_size=embedding_size,
                              input_size=input_size,
                              width_mult=width_mult)
    BACKBONE.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage.cuda(device)))
    # # BACKBONE.eval()
    # faces, issame = util.get_val_face_data_bin(val_root, face_dataset=dataset, input_size=input_size)
    faces, issame = util.get_val_face_data_bcolz(val_root, face_dataset=dataset, input_size=input_size)
    # # faces, issame = reszie_val_face_data(faces, issame, resize=[112, 112])
    # # faces, issame = reszie_val_face_data(faces, issame, resize=input_size)
    eva = EvaluationMetrics()
    # 0.9666666666666666
    val_accuracy, best_threshold, roc_curve = eva.perform_val(MULTI_GPU,
                                                              device,
                                                              embedding_size,
                                                              BATCH_SIZE,
                                                              input_size,
                                                              BACKBONE,
                                                              faces,
                                                              issame,
                                                              tta=tta)
    print("perform_val_test", val_accuracy)

    eva.update_model(BACKBONE, device, multi_gpu=MULTI_GPU)
    val_accuracy, best_threshold, roc_curve = eva.metrics(faces,
                                                          issame,
                                                          batch_size=BATCH_SIZE,
                                                          input_size=input_size,
                                                          tta=tta)
    print("EvaluationMetrics", val_accuracy)

    return val_accuracy


def perform_val_test_bin(VAL_ROOT, dataset, model_path, backbone_name, embedding_size, input_size, width_mult):
    '''

    :param VAL_ROOT:
    :param dataset:
    :param model_path:
    :param backbone_name:
    :param embedding_size:
    :param input_size:
    :param width_mult:
    :return:
    '''
    from evaluation.eval import face_eval
    from evaluation import evaluation_test_for_11_pair
    PREFIX = "dmai_model"
    save_path = "./"
    eval_prefix = "tmp"
    alignment = False
    bin_path = os.path.join(VAL_ROOT, dataset + ".bin")
    faces_data, issames_data = bin_tools.load_bin(bin_path, image_size=input_size)  # dataset is [112,112]
    val_accuracy, best_threshold, roc_curve = face_eval.perform_val_kfold(faces_data, issames_data, model_path,
                                                                          backbone_name, input_size, embedding_size,
                                                                          save_path=save_path, eval_prefix=eval_prefix,
                                                                          width_mult=width_mult, alignment=alignment)
    print("perform_val_kfold", val_accuracy)
    print("=" * 50)
    pred_score, issames_data = face_eval.get_pair_scores_for_bin(faces_data, issames_data, model_path, backbone_name,
                                                                 input_size=input_size, embedding_size=embedding_size,
                                                                 save_path=save_path,
                                                                 eval_prefix=eval_prefix,
                                                                 width_mult=width_mult, alignment=alignment)
    info = "alignment:{}\n".format(alignment)
    info += "val_filename:{}\n".format(dataset)
    info += "eval_prefix:{}(1:1)\n".format(eval_prefix)
    max_acc = evaluation_test_for_11_pair.get_pair_eval_data(issames_data, pred_score, save_path, info=info)
    return max_acc


def val_demo(val_root, model_root, save_path):
    """
    :param val_root:
    :param model_root:
    :param save_path:
    :return:
    """
    # define model
    backbone_name = "IR_26"
    input_size = [112, 112]
    PREFIX = "IR_MB_V2"
    model_root = model_root + "Asian/ResNet26/IR_18_ArcFace_Focal_112_512_None_default_ms1m_align_112_Asian_Celeb_20190925_090904/models/"
    model_path = model_root + "backbone_IR_18_ArcFace_Focal_112_512_None_Epoch_152_X4_112_0.996_X4_64_0.996_NVR1_Alig_112_0.929_NVR1_Alig_64_0.947_NVR1_112_0.860_NVR1_64_0.849_agedb_30_112_0.968_agedb_30_64_0.962.pth"
    # model_path = "/media/dm/dm2/FaceRecognition/torch-Face-Recognize-Pipeline/work_space_ArcFace/IR_MB_V2_ArcFace_Focal_112_512_1.0_default_Asian_Celeb_20191210_144848/models/backbone_IR_MB_V2_ArcFace_Focal_112_512_1.0_Epoch_159_X4_112_0.974_lfw_112_0.989_agedb_30_112_0.913.pth"
    width_mult = 1.0
    dataset = "agedb_30"
    embedding_size = 512
    ##########################################################################
    perform_val_test(val_root, dataset, model_path, backbone_name, embedding_size, input_size, width_mult)
    # perform_val_test_bin(val_root, dataset, model_path, backbone_name, embedding_size, input_size, width_mult)


if __name__ == "__main__":
    val_root = "/media/dm/dm1/FaceDataset/bin/"
    model_root = '/media/dm/dm1/FaceRecModel/'
    save_path = "./eval_output"
    val_demo(val_root, model_root, save_path)
