import argparse
import os
import random
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from mmcv import Config
from mmcls.models import build_classifier
from scipy.stats import gaussian_kde
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


def compute_kl_divergence(source_features, target_features):
    """
    计算KL散度
    source_features, target_features: torch.Tensor
    """
    # 转换为numpy数组
    source_features = source_features.cpu().numpy()
    target_features = target_features.cpu().numpy()

    # 计算每个维度的KL散度
    n_features = source_features.shape[1]
    kl_div_total = 0

    for i in range(n_features):
        # 获取当前维度的特征
        source_samples = source_features[:, i]
        target_samples = target_features[:, i]

        # 使用高斯核密度估计来估计概率密度
        try:
            source_kde = gaussian_kde(source_samples)
            target_kde = gaussian_kde(target_samples)

            # 在一个固定的点集上评估密度
            x_eval = np.linspace(min(np.min(source_samples), np.min(target_samples)),
                                 max(np.max(source_samples), np.max(target_samples)),
                                 num=100)

            # 计算概率密度
            p = source_kde(x_eval)
            q = target_kde(x_eval)

            # 为了数值稳定性，添加小的常数
            epsilon = 1e-10
            p = p + epsilon
            q = q + epsilon

            # 计算KL散度
            kl_div = np.sum(p * np.log(p / q)) * (x_eval[1] - x_eval[0])
            kl_div_total += kl_div

        except np.linalg.LinAlgError:
            # 处理奇异矩阵错误
            print(f"Warning: Singular matrix in dimension {i}, skipping...")
            continue

    # 返回平均KL散度
    return kl_div_total / n_features


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
    计算两个域之间的KL散度

    参数:
    source_path: 源域图片目录
    target_path: 目标域图片目录
    sourcenet: 预训练的特征提取器
    batch_size: 批次大小

    返回:
    kl_divergence: float, KL散度
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

    try:
        # 提取特征
        print("Extracting source features...")
        source_features = extract_features(sourcenet, source_loader, device)
        print(f"Extracted source features shape: {source_features.shape}")

        print("Extracting target features...")
        target_features = extract_features(sourcenet, target_loader, device)
        print(f"Extracted target features shape: {target_features.shape}")

        # 计算KL散度
        kl_div = compute_kl_divergence(source_features, target_features)

        # 计算对称的KL散度 (Jensen-Shannon divergence)
        kl_div_reverse = compute_kl_divergence(target_features, source_features)
        symmetric_kl = (kl_div + kl_div_reverse) / 2

        return float(symmetric_kl)

    except Exception as e:
        print(f"Error during feature extraction or distance calculation: {str(e)}")
        raise


if __name__ == "__main__":
    # 设置路径
    source_path = "/media/shared_space/wuyanzu/ImageNet-C-Syn/gaussian_noise/5/n01440764"
    #target_path = "/media/shared_space/wuyanzu/ImageNet-C-Syn/gaussian_noise/5/n01440764"
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
    kl_div = calculate_domain_distance(source_path, target_path, source_net)
    print(f"Symmetric KL divergence between domains: {kl_div:.4f}")