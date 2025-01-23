import argparse
import os
import random
from torch.optim import AdamW
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import torch
import torch.distributed as dist
from torch.nn import DataParallel
import sys
from torch.optim import SGD
from mmcv import Config
from mmcls.models import build_classifier
from mmcv.runner import load_checkpoint
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from scipy.stats import wasserstein_distance
import warnings

warnings.filterwarnings("ignore", module='mmcv')


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


def compute_wasserstein(source_features, target_features):
    """
    计算 Wasserstein 距离
    source_features, target_features: torch.Tensor
    """
    # 确保数据在CPU上且转换为numpy数组
    source_features = source_features.cpu().numpy()
    target_features = target_features.cpu().numpy()

    # 计算每个特征维度的 Wasserstein 距离
    n_features = source_features.shape[1]
    distances = []

    for i in range(n_features):
        # 对每个特征维度计算一维 Wasserstein 距离
        dist = wasserstein_distance(source_features[:, i], target_features[:, i])
        distances.append(dist)

    # 返回所有维度 Wasserstein 距离的平均值
    return np.mean(distances)


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
    计算两个域之间的 Wasserstein 距离

    参数:
    source_path: 源域图片目录
    target_path: 目标域图片目录
    sourcenet: 预训练的特征提取器
    batch_size: 批次大小

    返回:
    wasserstein_distance: float, Wasserstein 距离
    """
    # 检查路径是否存在
    if not os.path.exists(source_path):
        raise ValueError(f"Source path does not exist: {source_path}")
    if not os.path.exists(target_path):
        raise ValueError(f"Target path does not exist: {target_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sourcenet = sourcenet.to(device)

    # 获取图片路径
    source_images = [os.path.join(source_path, f) for f in os.listdir(source_path)
                     if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    target_images = [os.path.join(target_path, f) for f in os.listdir(target_path)
                     if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    print(f"Found {len(source_images)} source images and {len(target_images)} target images")

    if not source_images:
        raise ValueError(f"No valid images found in source path: {source_path}")
    if not target_images:
        raise ValueError(f"No valid images found in target path: {target_path}")

    # 创建数据加载器
    source_dataset = ImageDataset(source_images)
    target_dataset = ImageDataset(target_images)

    source_loader = DataLoader(source_dataset, batch_size=batch_size,
                               shuffle=False, num_workers=4)
    target_loader = DataLoader(target_dataset, batch_size=batch_size,
                               shuffle=False, num_workers=4)

    print(f"Source loader has {len(source_loader)} batches")
    print(f"Target loader has {len(target_loader)} batches")

    try:
        # 提取特征
        print("Extracting source features...")
        source_features = extract_features(sourcenet, source_loader, device)
        print(f"Extracted source features shape: {source_features.shape}")

        print("Extracting target features...")
        target_features = extract_features(sourcenet, target_loader, device)
        print(f"Extracted target features shape: {target_features.shape}")

        # 计算 Wasserstein 距离
        wasserstein_dist = compute_wasserstein(source_features, target_features)
        return float(wasserstein_dist)

    except Exception as e:
        print(f"Error during feature extraction or distance calculation: {str(e)}")
        raise


if __name__ == "__main__":
    # 设置路径
    source_path = "/media/shared_space/wuyanzu/ImageNet-C-Syn/n01440764"
    # target_path = "/media/shared_space/wuyanzu/ImageNet-C-Syn/n01440764"
    # target_path = "/media/shared_space/dongxiaolin/maple1/data/imagenet/images/val/n01440764"
    target_path = "/media/shared_space/wuyanzu/DDA-watch/dataset/test/DiT-XL-2-DiT-XL-2-256x256-size-256-class0/5/1"
    classifier_config_path = "/media/shared_space/wuyanzu/DDA-main/model_adapt/configs/ensemble/convnextT_ensemble_b64_imagenet.py"
    pretrained_weights_path = "/media/shared_space/wuyanzu/DDA-main/ckpt/convnext-tiny_3rdparty_32xb128_in1k_20220124-18abde00.pth"

    # 设置 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # 加载模型配置和权重
    cfg = Config.fromfile(classifier_config_path)
    source_net = build_classifier(cfg.model)
    source_net.to('cuda')

    # 加载预训练权重
    checkpoint = torch.load(pretrained_weights_path, map_location='cuda')
    model_state_dict = checkpoint['state_dict']

    # 分离 backbone 和 head 的权重
    backbone_state_dict = {}
    head_state_dict = {}
    for k, v in model_state_dict.items():
        if k.startswith('backbone.'):
            backbone_state_dict[k.replace('backbone.', '')] = v
        elif k.startswith('head.'):
            head_state_dict[k.replace('head.', '')] = v

    # 加载 backbone 的权重
    source_net.backbone.load_state_dict(backbone_state_dict, strict=True)

    # 计算域间距离
    wasserstein_dist = calculate_domain_distance(source_path, target_path, source_net)
    print(f"Wasserstein distance between domains: {wasserstein_dist:.4f}")