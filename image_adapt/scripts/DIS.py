import argparse
import os
import random
from torch.optim import AdamW  # 引入 AdamW 优化器
import torch.nn as nn
from tqdm import tqdm

import numpy as np
import torch as th
import torch.distributed as dist
from torch.nn import DataParallel
import sys
from torch.optim import SGD

from mmcv import Config
from mmcls.models import build_classifier
from mmcv.runner import load_checkpoint

from mmcv import Config
from mmcls.models import build_classifier
from image_adapt.model_use import Classifier_covnextT

from image_adapt.guided_diffusion import dist_util, logger
from image_adapt.guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from torchvision import transforms, datasets
import torch.nn.functional as F
from image_adapt.resize_right import resize

import warnings
warnings.filterwarnings("ignore", module='mmcv')
import os
import lpips
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np


class ImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


def compute_kernel_matrix(x, y, kernel='rbf', sigma=None):
    """
    计算核矩阵
    x, y: torch.Tensor, shape [n, d] 和 [m, d]
    """
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)

    if kernel == 'rbf':
        if sigma is None:
            sigma = dim ** 0.5
        x_kernel = x.unsqueeze(1).expand(x_size, y_size, dim)
        y_kernel = y.unsqueeze(0).expand(x_size, y_size, dim)
        kernel_matrix = torch.exp(-((x_kernel - y_kernel).pow(2).sum(2) / (2 * sigma ** 2)))
    else:
        raise NotImplementedError(f"Kernel {kernel} not implemented.")

    return kernel_matrix


def compute_mmd(source_features, target_features, kernel='rbf'):
    """
    计算MMD距离
    source_features, target_features: torch.Tensor
    """
    n_source = source_features.size(0)
    n_target = target_features.size(0)

    # 计算核矩阵
    K_ss = compute_kernel_matrix(source_features, source_features)
    K_tt = compute_kernel_matrix(target_features, target_features)
    K_st = compute_kernel_matrix(source_features, target_features)

    # 计算MMD
    mmd = (K_ss.sum() / (n_source * n_source) +
           K_tt.sum() / (n_target * n_target) -
           2 * K_st.sum() / (n_source * n_target))

    return mmd


def extract_features(model, dataloader, device):
    """
    使用模型提取特征
    """
    features_list = []
    model.eval()
    with torch.no_grad():
        for images in dataloader:
            images = images.to(device)
            output = model.backbone(images)
            if isinstance(output, tuple):
                features = output[0]
            else:
                features = output

            features_list.append(features.cpu())


    return torch.cat(features_list, dim=0)


def calculate_domain_distance(source_path, target_path, sourcenet, batch_size=32):
    """
    计算两个域之间的MMD距离

    参数:
    source_path: 源域图片目录
    target_path: 目标域图片目录
    sourcenet: 预训练的特征提取器
    batch_size: 批次大小

    返回:
    mmd_distance: float, MMD距离
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sourcenet = sourcenet.to(device)

    # 获取图片路径
    source_images = [os.path.join(source_path, f) for f in os.listdir(source_path)
                     if f.endswith(('.jpg', '.png', '.JPEG'))]
    target_images = [os.path.join(target_path, f) for f in os.listdir(target_path)
                     if f.endswith(('.jpg', '.png', '.JPEG'))]

    # 创建数据加载器
    source_dataset = ImageDataset(source_images)
    target_dataset = ImageDataset(target_images)

    source_loader = DataLoader(source_dataset, batch_size=batch_size,
                               shuffle=False, num_workers=4)
    target_loader = DataLoader(target_dataset, batch_size=batch_size,
                               shuffle=False, num_workers=4)

    # 提取特征
    source_features = extract_features(sourcenet, source_loader, device)
    target_features = extract_features(sourcenet, target_loader, device)

    # 计算MMD距离
    mmd_distance = compute_mmd(source_features, target_features)

    return mmd_distance.item()


# 使用示例
if __name__ == "__main__":
    # 假设sourcenet已经定义并加载了预训练权重
    source_path = "/media/shared_space/wuyanzu/ImageNet-C-Syn/gaussian_noise/5/n01440764"
    target_path = "/media/shared_space/wuyanzu/DDA-watch/dataset/test/DiT-XL-2-DiT-XL-2-256x256-size-256-class0/5/1"
    classifier_config_path = "/media/shared_space/wuyanzu/DDA-main/model_adapt/configs/ensemble/convnextT_ensemble_b64_imagenet.py"
    pretrained_weights_path = "/media/shared_space/wuyanzu/DDA-main/ckpt/convnext-tiny_3rdparty_32xb128_in1k_20220124-18abde00.pth"

    # 加载模型配置和权重
    cfg = Config.fromfile(classifier_config_path)
    source_net = build_classifier(cfg.model)
    source_net.to('cuda')

    # 加载预训练权重
    checkpoint = th.load(pretrained_weights_path, map_location='cuda')
    model_state_dict = checkpoint['state_dict']

    backbone_state_dict = {}
    head_state_dict = {}
    for k, v in model_state_dict.items():
        if k.startswith('backbone.'):
            backbone_state_dict[k.replace('backbone.', '')] = v
        elif k.startswith('head.'):
            head_state_dict[k.replace('head.', '')] = v

    # 加载 backbone 的权重到三个模型中
    for model2 in [source_net]:
        model2.backbone.load_state_dict(backbone_state_dict, strict=True)

    mmd_dist = calculate_domain_distance(source_path, target_path, source_net)
    print(f"MMD distance between domains: {mmd_dist:.4f}")