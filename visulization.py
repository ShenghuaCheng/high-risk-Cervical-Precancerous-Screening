# -*- coding: utf-8 -*-
"""
see top 200 instance in a view of image and the attention heat map in the instances
"""
import os
import cv2
import torch
import numpy as np
from utils.smallwsiread import SmallWsiRead


def check_stage_one_top_10(result_dir_name, slide_name):
    slide_infer_re = torch.load(os.path.join(result_dir_name, '{}_slide_infer_re.pth'.format(slide_name)))
    infer = slide_infer_re[0]

    boxes = []
    for name, cor_w, cor_h, im_batch, A, merged_feature, class_out in infer[:10]:
        boxes.append([name, cor_w, cor_h, A])

    return boxes


def check_stage_two_top_10(result_dir_name, slide_name):
    slide_infer_re = torch.load(os.path.join(result_dir_name, '{}_slide_infer_re.pth'.format(slide_name)))
    attention = slide_infer_re[-2]
    _, top_10_index = torch.sort(attention)
    infer = slide_infer_re[0]

    boxes = []
    for index, (name, cor_w, cor_h, im_batch, A, merged_feature, class_out) in enumerate(infer):
        if top_10_index[index] < 10:
            boxes.append([name, cor_w, cor_h, A])

    return boxes


def check_view(result_dir_name, slide_name, view_name):
    slide_infer_re = torch.load(os.path.join(result_dir_name, '{}_slide_infer_re.pth'.format(slide_name)))
    infer = slide_infer_re[0]

    boxes = []
    for name, cor_w, cor_h, im_batch, A, merged_feature, class_out in infer:
        if name == view_name:
            boxes.append([cor_w, cor_h, A])

    return boxes


def see_a_view(result_dir_path, check_view_full_path, save_dir):
    slide_name = check_view_full_path.split(os.sep)[-2]
    view_name = check_view_full_path.split(os.sep)[-1].split('.')[0]

    im_view = cv2.imread(check_view_full_path)
    im_view = np.ascontiguousarray(SmallWsiRead.crop_preprocess(im_view))
    ori_view = im_view.copy()

    boxes = check_view(result_dir_path, slide_name, view_name)
    if len(boxes) == 0:
        print('there is no top 200 instance in: ', check_view_full_path)
    else:
        for cor_w, cor_h, A in boxes:
            im_view = cv2.rectangle(im_view, (int(cor_w), int(cor_h)), (int(cor_w) + 256, int(cor_h) + 256), (255, 0, 0), 3)
            region = cv2.resize(ori_view[int(cor_h): int(cor_h) + 256, int(cor_w): int(cor_w) + 256, ::].copy(), (224, 224))
            A = (A - torch.min(A)) / (torch.max(A) - torch.min(A))
            patch_size = 16
            im_size = 224
            mask = np.zeros_like(region)
            for i in range(im_size // patch_size):
                for j in range(im_size // patch_size):
                    value = A[i * (im_size // patch_size) + j].item() * 255
                    mask[i * patch_size: i * patch_size + patch_size, j * patch_size: j * patch_size + patch_size, ::] = np.array([0, 0, value]).astype(np.uint8)

            add_im = cv2.addWeighted(region, 1, mask, 0.7, 0)

            cv2.imwrite(os.path.join(save_dir, 'instance_w{}_h{}_mask.jpg'.format(cor_w, cor_h)), add_im)
            cv2.imwrite(os.path.join(save_dir, 'instance_w{}_h{}.jpg'.format(cor_w, cor_h)), region)

            print('save: ', os.path.join(save_dir, 'instance_w{}_h{}.jpg'.format(cor_w, cor_h)))
            print('save: ', os.path.join(save_dir, 'instance_w{}_h{}_mask.jpg'.format(cor_w, cor_h)))

        cv2.imwrite(os.path.join(save_dir, 'view_re.jpg'), im_view)
        print('save: ', os.path.join(save_dir, 'view_re.jpg'))


def see_stage_one_top_10(result_dir_path, slide_dir_path, format, save_dir):
    slide_name = slide_dir_path.split(os.sep)[-1]
    boxes = check_stage_one_top_10(result_dir_path, slide_name)

    for index, (name, cor_w, cor_h, A) in enumerate(boxes):
        print('top {} instance in {}.{}'.format(index + 1, name, format))

        view_im = SmallWsiRead.crop_preprocess(cv2.imread(os.path.join(slide_dir_path, name + '.{}'.format(format))))

        region = cv2.resize(view_im[int(cor_h): int(cor_h) + 256, int(cor_w): int(cor_w) + 256, ::].copy(), (224, 224))
        A = (A - torch.min(A)) / (torch.max(A) - torch.min(A))
        patch_size = 16
        im_size = 224
        mask = np.zeros_like(region)
        for i in range(im_size // patch_size):
            for j in range(im_size // patch_size):
                value = A[i * (im_size // patch_size) + j].item() * 255
                mask[i * patch_size: i * patch_size + patch_size, j * patch_size: j * patch_size + patch_size,
                ::] = np.array([0, 0, value]).astype(np.uint8)

        add_im = cv2.addWeighted(region, 1, mask, 0.7, 0)

        cv2.imwrite(os.path.join(save_dir, 'top_{}_instance_w{}_h{}_mask.jpg'.format(index + 1, cor_w, cor_h)), add_im)
        cv2.imwrite(os.path.join(save_dir, 'top_{}_instance_w{}_h{}.jpg'.format(index + 1, cor_w, cor_h)), region)


def see_stage_two_top_10(result_dir_path, slide_dir_path, format, save_dir):
    slide_name = slide_dir_path.split(os.sep)[-1]
    boxes = check_stage_two_top_10(result_dir_path, slide_name)

    for index, (name, cor_w, cor_h, A) in enumerate(boxes):
        print('top {} instance in {}.{}'.format(index + 1, name, format))

        view_im = SmallWsiRead.crop_preprocess(cv2.imread(os.path.join(slide_dir_path, name + '.{}'.format(format))))

        region = cv2.resize(view_im[int(cor_h): int(cor_h) + 256, int(cor_w): int(cor_w) + 256, ::].copy(), (224, 224))
        A = (A - torch.min(A)) / (torch.max(A) - torch.min(A))
        patch_size = 16
        im_size = 224
        mask = np.zeros_like(region)
        for i in range(im_size // patch_size):
            for j in range(im_size // patch_size):
                value = A[i * (im_size // patch_size) + j].item() * 255
                mask[i * patch_size: i * patch_size + patch_size, j * patch_size: j * patch_size + patch_size,
                ::] = np.array([0, 0, value]).astype(np.uint8)

        add_im = cv2.addWeighted(region, 1, mask, 0.7, 0)

        cv2.imwrite(os.path.join(save_dir, 'top_{}_instance_w{}_h{}_mask.jpg'.format(index + 1, cor_w, cor_h)), add_im)
        cv2.imwrite(os.path.join(save_dir, 'top_{}_instance_w{}_h{}.jpg'.format(index + 1, cor_w, cor_h)), region)
