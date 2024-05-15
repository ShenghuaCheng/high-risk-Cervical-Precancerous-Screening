# -*- coding:utf-8 -*-
from utils.smallwsiread import SmallWsiRead
from core.mae_models import MyEncoder
from core.slide_level_model import AttentionFeatureClassifier

import torch
import torch.nn as nn
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.nn.functional import softmax as f_softmax

import os
import cv2
import multiprocessing.dummy as mp
import multiprocessing


class TDataSet(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        name, cor_w, cor_h, im = self.data[item]
        return name, cor_w, cor_h, im

    def __len__(self):
        return len(self.data)


def prepare_lr_wsi(data_path, save_path):
    if os.path.exists(save_path):
        print('file exists: {}'.format(save_path))

        return 0

    print('start to prepare LRWSI: {}'.format(data_path))

    pre_norm = ToTensor()
    lock = multiprocessing.Lock()

    def _read(_name, _numpy_im):
        _index_2 = -_name[::-1].index('_')
        _index_1 = -_name[:_index_2 - 1][::-1].index('_') + _index_2 - 1
        _block = _name[:_index_1 - 1]
        _cor_w = _name[_index_1:_index_2 - 1]
        _cor_h = _name[_index_2:]
        _im = cv2.resize(_numpy_im, (224, 224))
        _im = pre_norm(_im)
        with lock:
            data.append([_block, _cor_w, _cor_h, _im])

    lr_wsi = SmallWsiRead(data_path)
    lr_wsi.read()

    data = []
    pool = mp.Pool()
    for name, numpy_im in lr_wsi.instances.items():
        pool.apply_async(_read, (name, numpy_im))
    pool.close()
    pool.join()

    # saved as: [[image_name, cor_w, cor_h, normed_tensor_im], ]
    torch.save(data, save_path)

    print('save file: ', save_path)


def instance_inference(data_path, save_path, batch_size):
    if os.path.exists(save_path):
        print('file exists: {}'.format(save_path))
        return 0

    print('start to inference instances: {}'.format(data_path))
    device = torch.device('cuda', index=0)

    model = MyEncoder(
        img_size=224,
        patch_size=16,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_head_out_dim=0,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        device=device
    )
    load_weights = torch.load('weigths/encoder_b4d4_aug_mix_270.pth', map_location='cpu')['model_weights']
    model.load_state_dict(load_weights)
    model.to(device)
    model.eval()
    data = torch.load(data_path, map_location='cpu')
    dataset = TDataSet(data)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
    )

    feature_data = []
    with torch.no_grad():
        for name, cor_w, cor_h, im_batch in loader:
            im_batch_ = im_batch.to(device)
            merged_feature, A, class_out = model(im_batch_, 0)

            class_out = class_out.detach().clone().cpu()
            class_out = f_softmax(class_out, dim=1)

            for index in range(im_batch.shape[0]):
                feature_data.append([
                    name[index],
                    cor_w[index],
                    cor_h[index],
                    im_batch[index].detach().clone().cpu(),
                    A[index].detach().clone().cpu(),
                    merged_feature[index].detach().clone().cpu(),
                    class_out[index][1].item()
                ])

    feature_data.sort(key=lambda x: x[-1], reverse=True)
    torch.save(feature_data, os.path.join(save_path))

    print('save: ', save_path)


def slide_inference(data_path, save_path):
    if os.path.exists(save_path):
        print('file exists: {}'.format(save_path))
        return 0

    print('start to inference lrwsi: {}'.format(data_path))

    device = torch.device('cuda', index=0)

    model = AttentionFeatureClassifier(
        pre_head=None,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        device=None
    )

    load_weights = torch.load('weigths/slide_classifier_exp_6_restart_better_34.pth', map_location='cpu')['model_weights']
    model.load_state_dict(load_weights)
    model.to(device)
    model.eval()

    all_instances = torch.load(data_path)
    top_200_instances = torch.stack([_[-2] for _ in all_instances[:200]])
    top_200_instances = top_200_instances.to(device)

    with torch.no_grad():
        merged_feature, A, class_out = model(top_200_instances.unsqueeze(0))

    torch.save([
        all_instances[:200],
        merged_feature.detach().clone().cpu(),
        A.detach().clone().cpu(),
        f_softmax(class_out, dim=0)[1].item()
    ], save_path)

    p = f_softmax(class_out, dim=0)[1].item()
    print('positive probability: {} '.format('POS' if p > 0.5 else 'NEG'), p)


def instance_inference_single(im_list):
    """
    im_list: [normed_0-1_tesor]
    return: [[im_merge_feature, A, cls]]
    """
    device = torch.device('cuda', index=0)
    model = MyEncoder(
        img_size=224,
        patch_size=16,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_head_out_dim=0,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        device=device
    )
    load_weights = torch.load('weigths/encoder_b4d4_aug_mix_270.pth', map_location='cpu')['model_weights']
    model.load_state_dict(load_weights)
    model.to(device)
    model.eval()

    results = []
    with torch.no_grad():
        for single_im in im_list:
            im_batch_ = single_im.unsqueeze(0).to(device)
            attention_feature, A, class_out = model(im_batch_, 0)
            attention_feature = attention_feature.detach().clone().cpu()
            class_out = class_out.detach().clone().cpu()
            A = A.detach().clone().cpu()

            results.append([attention_feature, A, class_out])

    return results


def infer_slide(dir_full_path: str, save_dir, batch_size):
    name = dir_full_path.split(os.sep)[-1]
    prepare_lr_wsi(dir_full_path, os.path.join(save_dir, '{}_instances.pth'.format(name)))
    instance_inference(
        os.path.join(save_dir, '{}_instances.pth'.format(name)),
        os.path.join(save_dir, '{}_instances_infer_re.pth'.format(name)),
        batch_size
    )
    slide_inference(
        os.path.join(save_dir, '{}_instances_infer_re.pth'.format(name)),
        os.path.join(save_dir, '{}_slide_infer_re.pth'.format(name))
    )



