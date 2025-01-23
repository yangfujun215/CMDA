import argparse
import os
import random
from torch.optim import AdamW  # 引入 AdamW 优化器

from tqdm import tqdm

import numpy as np
import torch as th
import torch.distributed as dist
from torch.nn import DataParallel
import sys
from torch.optim import SGD

from mmcv import Config
from mmcls.models import build_classifier
from mmcv.runner import load_checkpoint, wrap_fp16_model

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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import lpips
logger.log("纯model、用来计算天气正确率。")
import json  # 用于读取类别映射文件
from PIL import Image

# # 定义图像归一化参数
# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53],
#     std=[58.395, 57.12, 57.375],
#     to_rgb=True
# )
#
# def load_paired_data(data_dir1, data_dir2, batch_size, image_size, class_cond=False, max_samples=None):
#     # 定义图像变换
#     transform = transforms.Compose([
#         transforms.Resize(256),  # 保持宽高比，将宽度调整到 256
#         transforms.CenterCrop(224),
#         transforms.Lambda(lambda img: img.convert("RGB")),
#         transforms.ToTensor(),
#         transforms.Lambda(lambda x: x * 255),  # 将像素值从 [0, 1] 还原到 [0, 255]
#         transforms.Normalize(
#             mean=img_norm_cfg['mean'],
#             std=img_norm_cfg['std']
#         )
#     ])
#
#     # 加载两个数据集
#     dataset1 = datasets.ImageFolder(root=data_dir1, transform=transform)
#     dataset2 = datasets.ImageFolder(root=data_dir2, transform=transform)
#
#     # 如果需要，截取最大样本数
#     if max_samples is not None:
#         dataset1.samples = dataset1.samples[:max_samples]
#         dataset2.samples = dataset2.samples[:max_samples]
#
#     # 确保两个数据集的样本数量相同
#     assert len(dataset1) == len(dataset2), "两个数据集的样本数量不一致"
#
#     # 自定义数据集，返回图像对和标签
#     class PairedDataset(th.utils.data.Dataset):
#         def __init__(self, dataset1, dataset2):
#             self.dataset1 = dataset1
#             self.dataset2 = dataset2
#
#         def __len__(self):
#             return len(self.dataset1)
#
#         def __getitem__(self, idx):
#             img1, label1 = self.dataset1[idx]
#             img2, label2 = self.dataset2[idx]
#             # 可选：检查标签是否一致
#             assert label1 == label2, "样本标签不匹配"
#             return (img1, img2), label1, idx    # 返回样本和索引
#
#     # 创建成对的数据集
#     paired_dataset = PairedDataset(dataset1, dataset2)
#
#     # 创建数据加载器
#     data_loader = th.utils.data.DataLoader(paired_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
#
#     return data_loader

# 计算熵函数

# 定义图像归一化参数
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)

def get_wnid_to_idx_mapping():
    """
    获取 ImageNet 的 WNID（类别 ID）到索引（0-999）的映射。
    需要有 'imagenet_class_index.json' 文件，该文件包含了 ImageNet 的类别映射。
    """
    # 请将下面的路径替换为您本地 imagenet_class_index.json 文件的实际路径
    imagenet_class_index_path = "/media/shared_space/wuyanzu/DDA-main/imagenet_class_index.json"
    with open(imagenet_class_index_path) as f:
        class_idx = json.load(f)
    wnid_to_idx = {v[0]: int(k) for k, v in class_idx.items()}
    return wnid_to_idx

def build_paired_dataset(data_dir1, data_dir2, transform, wnid_to_idx, max_samples=None):
    """
    构建成对的数据集，只包含同时存在于两个数据集的样本。
    Args:
        data_dir1: 第一个数据集的根目录（目标域）
        data_dir2: 第二个数据集的根目录（虚拟域）
        transform: 图像变换
        wnid_to_idx: WNID 到索引的映射
        max_samples: 最大样本数
    Returns:
        samples: 样本列表，每个元素为 (img_path1, img_path2, label)
    """
    samples = []
    # 获取所有类别的 WNID（文件夹名）
    wnids = os.listdir(data_dir1)
    for wnid in wnids:
        class_dir1 = os.path.join(data_dir1, wnid)
        class_dir2 = os.path.join(data_dir2, wnid)
        # 检查类别文件夹是否存在
        if not os.path.isdir(class_dir1) or not os.path.isdir(class_dir2):
            continue
        # 获取类别索引
        class_idx = wnid_to_idx.get(wnid)
        if class_idx is None:
            continue  # 如果类别不在映射中，跳过
        # 获取类别文件夹中的所有文件名
        fnames = os.listdir(class_dir1)
        for fname in fnames:
            img_path1 = os.path.join(class_dir1, fname)
            img_path2 = os.path.join(class_dir2, fname)
            # 检查图像文件是否存在于两个数据集
            if not os.path.isfile(img_path1) or not os.path.isfile(img_path2):
                continue
            samples.append((img_path1, img_path2, class_idx))
            # 如果达到最大样本数，停止添加
            if max_samples is not None and len(samples) >= max_samples:
                break
        if max_samples is not None and len(samples) >= max_samples:
            break
    return samples

def load_paired_data(data_dir1, data_dir2, batch_size, image_size, class_cond=False, max_samples=None):
    # 定义图像变换
    transform = transforms.Compose([
        transforms.Resize(256),  # 保持宽高比，将宽度调整到 256
        transforms.CenterCrop(224),
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255),  # 将像素值从 [0, 1] 还原到 [0, 255]
        transforms.Normalize(
            mean=img_norm_cfg['mean'],
            std=img_norm_cfg['std']
        )
    ])

    # 获取 WNID 到索引的映射
    wnid_to_idx = get_wnid_to_idx_mapping()

    # 构建成对的数据集
    samples = build_paired_dataset(data_dir1, data_dir2, transform, wnid_to_idx, max_samples)

    # 创建数据集
    class PairedDataset(th.utils.data.Dataset):
        def __init__(self, samples, transform=None):
            self.samples = samples
            self.transform = transform

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            img_path1, img_path2, label = self.samples[idx]
            img1 = Image.open(img_path1).convert('RGB')
            img2 = Image.open(img_path2).convert('RGB')
            if self.transform is not None:
                img1 = self.transform(img1)
                img2 = self.transform(img2)
            return (img1, img2), label, idx

    paired_dataset = PairedDataset(samples, transform=transform)

    # 创建数据加载器
    data_loader = th.utils.data.DataLoader(paired_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    return data_loader



def Entropy(input_):
    epsilon = 1e-5
    entropy = -input_ * th.log(input_ + epsilon)
    entropy = th.sum(entropy, dim=1)
    return entropy

# 温度锐化函数
def temperature_sharpening(preds, temperature=0.5):
    """
    对预测结果进行温度缩放以锐化预测。
    Args:
        preds (torch.Tensor): 预测概率分布。
        temperature (float): 温度参数，值越小，分布越锐化。
    Returns:
        torch.Tensor: 锐化后的预测概率分布。
    """
    preds = preds ** (1 / temperature)
    preds = preds / preds.sum(dim=1, keepdim=True)
    return preds

# 验证机制
def ensemble_logits(logits1, logits2, logits3):
    # 简单相加
    # sum_logits = (logits1 + logits2 + logits3)/3
    # sum_logits2 = (logits1 + logits3)/2
    # sum_logits3 = (logits1 + logits2)/2
    # sum_logits4 = (logits2 + logits3)/2

    sum_logits = logits1 + logits2 + logits3
    sum_logits2 = logits1 + logits3
    sum_logits3 = logits1 + logits2
    sum_logits4 = logits2 + logits3

    # 基于熵加权相加
    ent1 = -(logits1.softmax(1) * logits1.log_softmax(1)).sum(1, keepdim=True)
    ent2 = -(logits2.softmax(1) * logits2.log_softmax(1)).sum(1, keepdim=True)
    ent3 = -(logits3.softmax(1) * logits3.log_softmax(1)).sum(1, keepdim=True)
    entropy_sum_logits = logits1 * (ent2 + ent3) + logits2 * (ent1 + ent3) + logits3 * (ent1 + ent2)

    # 基于置信度加权相加
    con1 = logits1.softmax(1).max(1, keepdim=True)[0]
    con2 = logits2.softmax(1).max(1, keepdim=True)[0]
    con3 = logits3.softmax(1).max(1, keepdim=True)[0]
    confidence_sum_logits = logits1 * con1 + logits2 * con2 + logits3 * con3
    confidence_sum_logits2 = logits1 * (con2 + con3) + logits2 * (con1 + con3) + logits3 * (con1 + con2)

    return sum_logits, sum_logits2, sum_logits3, sum_logits4, entropy_sum_logits, confidence_sum_logits, confidence_sum_logits2

def evaluate_model(model, model_frozen, model3,
                   original_inputs_list1, original_inputs_list2, updated_inputs_list,
                   labels_list, args):
    logger.log("Evaluating models on original and updated data...")

    # 将模型设置为评估模式
    model.eval()
    model_frozen.eval()
    model3.eval()

    # 获取模型所在的设备
    device = next(model.parameters()).device

    all_preds_trainable = []
    all_preds_frozen = []
    all_preds_frozen2 = []
    all_preds_ensemble_sum = []
    all_preds_ensemble_sum2 = []
    all_preds_ensemble_sum3 = []
    all_preds_ensemble_sum4 = []
    all_preds_ensemble_sum5 = []
    all_preds_ensemble_entropy_sum = []
    all_preds_ensemble_confidence_sum = []
    all_preds_ensemble_confidence_sum2 = []
    all_labels = []

    with th.no_grad():
        for original_x1, original_x2, updated_x2, labels in zip(original_inputs_list1, original_inputs_list2, updated_inputs_list, labels_list):
            original_x1 = original_x1.to(device)
            original_x2 = original_x2.to(device)
            updated_x2 = updated_x2.to(device)
            model3 = model3.to(device)
            labels = labels.to(device)

            logits_model = model3(original_x1, return_loss=False)
            logits_model2 = model3(original_x2, return_loss=False)

            # logits_add = logits_model + logits_model2
            logits_add = [np.add(a, b) for a, b in zip(logits_model, logits_model2)]

            # 更新后的中间域模型，处理中间域数据
            logits_trainable = model(updated_x2, return_loss=False)
            # 将 numpy.ndarray 转换为 torch.Tensor
            logits_trainable_tensors = [th.from_numpy(arr) for arr in logits_trainable]
            # 堆叠张量，得到形状为 (4, 1000)
            logits_trainable_tensors = th.stack(logits_trainable_tensors, dim=0)
            logits_trainable_tensors = logits_trainable_tensors.to(device)
            preds_trainable = logits_trainable_tensors.argmax(dim=1)

            # 原模型处理虚拟域数据
            logits_frozen = model_frozen(original_x2, return_loss=False)
            # 将 numpy.ndarray 转换为 torch.Tensor
            logits_frozen_tensors = [th.from_numpy(arr) for arr in logits_frozen]
            # 堆叠张量，得到形状为 (4, 1000)
            logits_frozen_tensors = th.stack(logits_frozen_tensors, dim=0)
            logits_frozen_tensors = logits_frozen_tensors.to(device)
            preds_frozen = logits_frozen_tensors.argmax(dim=1)

            # 原模型处理目标域数据
            logits_frozen2 = model_frozen(original_x1, return_loss=False)
            # 将 numpy.ndarray 转换为 torch.Tensor
            logits_frozen2_tensors = [th.from_numpy(arr) for arr in logits_frozen2]
            # 堆叠张量，得到形状为 (4, 1000)
            logits_frozen2_tensors = th.stack(logits_frozen2_tensors, dim=0)
            logits_frozen2_tensors = logits_frozen2_tensors.to(device)
            preds_frozen2 = logits_frozen2_tensors.argmax(dim=1)

            # 获取所有模式的集成结果
            y_ensemble_sum, y_ensemble_sum2, y_ensemble_sum3, y_ensemble_sum4, y_ensemble_entropy_sum, y_ensemble_confidence_sum, y_ensemble_confidence_sum2 = ensemble_logits(logits_trainable_tensors, logits_frozen_tensors, logits_frozen2_tensors)
            final_pred_sum = y_ensemble_sum.argmax(dim=1)
            final_pred_sum2 = y_ensemble_sum2.argmax(dim=1)
            final_pred_sum3 = y_ensemble_sum3.argmax(dim=1)
            final_pred_sum4 = y_ensemble_sum4.argmax(dim=1)
            final_pred_entropy_sum = y_ensemble_entropy_sum.argmax(dim=1)
            final_pred_confidence_sum = y_ensemble_confidence_sum.argmax(dim=1)
            final_pred_confidence_sum2 = y_ensemble_confidence_sum2.argmax(dim=1)
            # 将 logits_add 列表中的 numpy.ndarray 转换为 torch.Tensor
            logits_add = [th.tensor(item) if isinstance(item, np.ndarray) else item for item in logits_add]
            logits_add_tensor = th.stack(logits_add)
            pred_add = logits_add_tensor.argmax(dim=1)

            # 将当前批次的预测和标签添加到列表中
            all_preds_trainable.append(preds_trainable.cpu())
            all_preds_frozen.append(preds_frozen.cpu())
            all_preds_frozen2.append(preds_frozen2.cpu())
            all_preds_ensemble_sum.append(final_pred_sum.cpu())
            all_preds_ensemble_sum2.append(final_pred_sum2.cpu())
            all_preds_ensemble_sum3.append(final_pred_sum3.cpu())
            all_preds_ensemble_sum4.append(final_pred_sum4.cpu())
            all_preds_ensemble_sum5.append(pred_add.cpu())
            all_preds_ensemble_entropy_sum.append(final_pred_entropy_sum.cpu())
            all_preds_ensemble_confidence_sum.append(final_pred_confidence_sum.cpu())
            all_preds_ensemble_confidence_sum2.append(final_pred_confidence_sum2.cpu())
            all_labels.append(labels.cpu())

    # 将所有预测和标签拼接起来
    all_preds_trainable = th.cat(all_preds_trainable)
    all_preds_frozen = th.cat(all_preds_frozen)
    all_preds_frozen2 = th.cat(all_preds_frozen2)
    all_preds_ensemble_sum = th.cat(all_preds_ensemble_sum)
    all_preds_ensemble_sum2 = th.cat(all_preds_ensemble_sum2)
    all_preds_ensemble_sum3 = th.cat(all_preds_ensemble_sum3)
    all_preds_ensemble_sum4 = th.cat(all_preds_ensemble_sum4)
    all_preds_ensemble_sum5 = th.cat(all_preds_ensemble_sum5)
    all_preds_ensemble_entropy_sum = th.cat(all_preds_ensemble_entropy_sum)
    all_preds_ensemble_confidence_sum = th.cat(all_preds_ensemble_confidence_sum)
    all_preds_ensemble_confidence_sum2 = th.cat(all_preds_ensemble_confidence_sum2)
    all_labels = th.cat(all_labels)

    # 计算准确率
    accuracy_trainable = (all_preds_trainable == all_labels).float().mean().item()
    accuracy_frozen = (all_preds_frozen == all_labels).float().mean().item()
    accuracy_frozen2 = (all_preds_frozen2 == all_labels).float().mean().item()
    accuracy_ensemble_sum = (all_preds_ensemble_sum == all_labels).float().mean().item()
    accuracy_ensemble_sum2 = (all_preds_ensemble_sum2 == all_labels).float().mean().item()
    accuracy_ensemble_sum3 = (all_preds_ensemble_sum3 == all_labels).float().mean().item()
    accuracy_ensemble_sum4 = (all_preds_ensemble_sum4 == all_labels).float().mean().item()
    accuracy_ensemble_sum5 = (all_preds_ensemble_sum5 == all_labels).float().mean().item()
    accuracy_ensemble_entropy_sum = (all_preds_ensemble_entropy_sum == all_labels).float().mean().item()
    accuracy_ensemble_confidence_sum = (all_preds_ensemble_confidence_sum == all_labels).float().mean().item()
    accuracy_ensemble_confidence_sum2 = (all_preds_ensemble_confidence_sum2 == all_labels).float().mean().item()

    # 输出结果
    print(f"Trainable Model (Updated x2) Accuracy 中间域在中间域模型正确率: {accuracy_trainable * 100:.8f}%")
    print(f"Frozen Model (Original x2) Accuracy 虚拟域在原模型正确率: {accuracy_frozen * 100:.8f}%")
    print(f"Frozen Model (Original x1) Accuracy 目标域在原模型正确率: {accuracy_frozen2 * 100:.8f}%")
    print(f"中间域+虚拟域+目标域正确率: {accuracy_ensemble_sum * 100:.8f}%")
    print(f"中间域+目标域 正确率: {accuracy_ensemble_sum2 * 100:.8f}%")
    print(f"中间域+虚拟域 正确率: {accuracy_ensemble_sum3 * 100:.8f}%")
    print(f"DDA 正确率: {accuracy_ensemble_sum4 * 100:.8f}%")
    print(f"DDA 正确率2: {accuracy_ensemble_sum5 * 100:.8f}%")
    print(f"Combined Model Accuracy (entropy_sum) 正确率: {accuracy_ensemble_entropy_sum * 100:.8f}%")
    print(f"Combined Model Accuracy (confidence_sum) 正确率: {accuracy_ensemble_confidence_sum * 100:.8f}%")
    print(f"Combined Model Accuracy (confidence_sum2) 正确率: {accuracy_ensemble_confidence_sum2 * 100:.8f}%")

def denormalize_and_prepare_for_lpips(x):
    # x 当前为 (x - mean)/std 后的结果，并且 x原本范围为[0,255]
    # 因为 transforms 先 x * 255，再 Normalize(mean, std)
    # 先反归一化: x = x * std + mean => [0,255]
    # 再除以255: => [0,1]
    # 再 *2 -1 => [-1,1]
    mean = th.tensor(img_norm_cfg['mean']).view(1,3,1,1).to(x.device)
    std = th.tensor(img_norm_cfg['std']).view(1,3,1,1).to(x.device)
    x = x * std + mean
    x = x / 255.0
    x = x * 2 - 1
    return x

def main():
    args = create_argparser().parse_args()

    # 设置随机种子，保证结果可以复现
    random.seed(args.seed)
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    th.cuda.manual_seed_all(args.seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False

    dist_util.setup_dist()
    # logger.configure(dir=args.save_dir)

    # args.data_dir1 = f"/media/shared_space/yrzhen/data/ImageNet-C/{args.corruption}/{args.severity}"
    # args.data_dir2 = f"/media/shared_space/wuyanzu/ImageNet-C-Syn/{args.corruption}/{args.severity}"
    # logger.log(f"当前域： (corruption): {args.corruption}, severity: {args.severity}")

    # # 设置数据路径
    # args.data_dir1 = f"/home/yfj/imagenet-w/{args.corruption}/{args.severity}"
    # args.data_dir2 = f"/home/yfj/ImageNet-W-Syn/{args.corruption}/{args.severity}"
    # logger.log(f"当前域： (corruption): {args.corruption}, severity: {args.severity}")

    # ImageNet-R 数据集路径
    args.data_dir1 = f"/media/shared_space/wuyanzu/imagenet-r/{args.corruption}"
    args.data_dir2 = f"/media/shared_space/wuyanzu/imagenet-r-syn/{args.corruption}"
    logger.log(f"当前域： (corruption): {args.corruption}, severity: {args.severity}")

    logger.log("加载数据...")
    data_loader = load_paired_data(
        args.data_dir1,
        args.data_dir2,
        args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        max_samples=args.max_samples  # 添加最大样本数参数
    )

    # # 获取数据集的总样本数
    # dataset1 = datasets.ImageFolder(root=args.data_dir1)
    # total_samples = len(dataset1)  # 获取实际的样本总数
    # if args.max_samples is not None:
    #     total_samples = min(total_samples, args.max_samples)

    # 加载模型配置

    # 获取数据集的总样本数
    total_samples = len(data_loader.dataset)
    if args.max_samples is not None:
        total_samples = min(total_samples, args.max_samples)

    cfg = Config.fromfile(args.classifier_config_path)
    # 构建模型
    model = build_classifier(cfg.model)
    model.to('cuda')

    # 加载预训练权重
    load_checkpoint(model, args.pretrained_weights_path, map_location='cuda')

    # 创建冻结的模型副本
    import copy
    model_frozen = copy.deepcopy(model)
    for param in model_frozen.parameters():
        param.requires_grad = False
    model_frozen.to('cuda')

    logger.log("模型导入成功")

    learning_rate = 1e-5  # 设置学习率

    # 使用 AdamW 优化器
    optimizer = AdamW(model.parameters(), lr=learning_rate , weight_decay=1e-8)

    # 使用官方LPIPS度量
    lpips_loss_fn = lpips.LPIPS(net='alex').to('cuda')  # 可以根据需要改成'alex','squeeze','vgg'

    # optimizer = SGD(model.parameters(), lr=learning_rate)

    # 初始化模型和扩散过程
    model_diffusion, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model_diffusion.to(dist_util.dev())
    if args.use_fp16:
        model_diffusion.convert_to_fp16()
    model_diffusion = th.nn.DataParallel(model_diffusion)
    model_diffusion.eval()

    # 保存用于评估的输入和标签
    original_inputs_list1 = []  # 保存目标域数据
    original_inputs_list2 = []  # 保存虚拟域数据
    updated_inputs_list = []
    labels_list = []

    logger.log("开始初始化 Memory Bank...")

    # 初始化 Memory Bank
    mem_initialized = False
    mem_fea = None
    mem_cls = None
    feature_dim = None
    num_classes = None

    # 使用冻结的模型初始化 Memory Bank
    with tqdm(total=total_samples, desc="Initializing memory bank", unit="sample") as pbar:
        for idx, batch_data in enumerate(data_loader):
            (images1, images2), labels, indices = batch_data
            x1 = images1.to('cuda')
            indices = indices.to('cuda')
            batch_size = x1.size(0)
            with th.no_grad():
                logits_x1 = model(x1, return_loss=False)
                features_x1 = model.extract_feat(x1)
                features_x1 = features_x1[0]  # 现在 features_x1 是 torch.Size([4, 768])
                features_x1 = features_x1 / th.norm(features_x1, p=2, dim=1, keepdim=True)
                # 将 numpy.ndarray 列表转换为 torch.Tensor 列表
                logits_x1_tensors = [th.from_numpy(arr) for arr in logits_x1]
                # 堆叠成 (4, 1000) 的张量
                logits_x1_tensor = th.stack(logits_x1_tensors, dim=0)
                logits_x1_tensor = logits_x1_tensor.cuda()
                class_probs_x1 = F.softmax(logits_x1_tensor, dim=1)
            if mem_fea is None:
                feature_dim = features_x1.size(1)
                num_classes = class_probs_x1.size(1)
                mem_fea = th.zeros(total_samples, feature_dim).cuda()
                mem_cls = th.zeros(total_samples, num_classes).cuda()
            mem_fea[indices] = features_x1.detach()
            mem_cls[indices] = class_probs_x1.detach()
            pbar.update(batch_size)

    # # 使用冻结的模型初始化 Memory Bank
    # with tqdm(total=total_samples, desc="Initializing memory bank", unit="sample") as pbar:
    #     for idx, batch_data in enumerate(data_loader):
    #         (images1, images2), labels, indices = batch_data
    #         x2 = images2.to('cuda')
    #         indices = indices.to('cuda')
    #         batch_size = x2.size(0)
    #         with th.no_grad():
    #             logits_x2 = model(x2, return_loss=False)
    #             features_x2 = model.extract_feat(x2)
    #             features_x2 = features_x2[0]  # 现在 features_x2 是 torch.Size([4, 768])
    #             features_x2 = features_x2 / th.norm(features_x2, p=2, dim=1, keepdim=True)
    #             # 将 numpy.ndarray 列表转换为 torch.Tensor 列表
    #             logits_x2_tensors = [th.from_numpy(arr) for arr in logits_x2]
    #             # 堆叠成 (4, 1000) 的张量
    #             logits_x2_tensor = th.stack(logits_x2_tensors, dim=0)
    #             logits_x2_tensor = logits_x2_tensor.cuda()
    #             class_probs_x2 = F.softmax(logits_x2_tensor, dim=1)
    #         if mem_fea is None:
    #             feature_dim = features_x2.size(1)
    #             num_classes = class_probs_x2.size(1)
    #             mem_fea = th.zeros(total_samples, feature_dim).cuda()
    #             mem_cls = th.zeros(total_samples, num_classes).cuda()
    #         mem_fea[indices] = features_x2.detach()
    #         mem_cls[indices] = class_probs_x2.detach()
    #         pbar.update(batch_size)

    logger.log("Memory Bank 初始化完成")
    # 初始化优化器和计数器
    optimizer.zero_grad()

    with tqdm(total=total_samples, desc="Processing data", unit="sample") as pbar:
        for idx, batch_data in enumerate(data_loader):
            (images1, images2), labels, batch_indices = batch_data
            x1 = images1.to('cuda')  # 目标域数据，用于后续计算正确率
            x2 = images2.to('cuda')  # 虚拟域数据，用于进行中间域不确定性得分的比较，后续计算正确率
            x_orig = x1.clone().detach()  # 目标域数据
            x = x2.clone().detach()  # 利用虚拟域数据进行后续模型更新，然后更新模型到中间域
            x.requires_grad = True  # 对 x2 进行更新得到中间域数据
            labels = labels.to('cuda').detach()
            batch_indices = batch_indices.to('cuda')  # 将索引移动到 GPU

            batch_size = x.size(0)

            # 保存原始输入到列表
            original_inputs_list1.append(x_orig.clone().detach().cpu())
            original_inputs_list2.append(x2.clone().detach().cpu())
            labels_list.append(labels.clone().detach().cpu())

            # 模型设置为训练模式
            model.train()

            # 直接使用模型进行前向传播
            logits_current = model(x, return_loss=False)
            # 将 numpy.ndarray 转换为 torch.Tensor
            logits_current_tensors = [th.from_numpy(arr) for arr in logits_current]

            # 堆叠张量，得到形状为 (4, 1000)
            logits_current_tensor = th.stack(logits_current_tensors, dim=0)

            # 将张量移动到模型所在的设备
            device = next(model.parameters()).device
            logits_current_tensor = logits_current_tensor.to(device)
            softmax_out = th.nn.Softmax(dim=1)(logits_current_tensor)

            # 提取特征
            features_current = model.extract_feat(x)
            features_current = features_current[0]  # 现在 features_x1 是 torch.Size([4, 768])
            features_current = features_current / th.norm(features_current, p=2, dim=1, keepdim=True)

            # 应用温度锐化
            temperature = 0.8  # 可以根据需求调整温度参数
            sharpened_softmax_out = temperature_sharpening(softmax_out, temperature)

            # 计算信息熵损失
            loss_emin = th.mean(Entropy(sharpened_softmax_out))
            msoftmax_out = sharpened_softmax_out.mean(dim=0)
            loss_gemin = th.sum(-msoftmax_out * th.log(msoftmax_out + 1e-10))
            loss_im = loss_emin - loss_gemin

            # 提取特征
            with th.no_grad():
                features = features_current
                outputs_target = sharpened_softmax_out

            # 计算与 Memory Bank 的距离
            dis = -th.mm(features.detach(), mem_fea.t())

            # 排除自身（将自身距离设为最大值）
            for i in range(batch_size):
                dis[i, batch_indices[i]] = dis.max()  # 将自身距离设为最大值

            # 获取最近的 k 个邻居
            k = min(5 , mem_fea.size(0))
            _, p1 = th.topk(dis, k=k, dim=1, largest=False)
            w = th.zeros(batch_size, mem_fea.size(0)).cuda()
            for i in range(batch_size):
                w[i, p1[i]] = 1 / k

            # 计算加权分类输出，获取伪标签
            mem_cls_weighted = th.mm(w, mem_cls)
            weight_, pred = th.max(mem_cls_weighted, dim=1)

            # 插值一致性损失
            # 从 Beta 分布中采样 λ
            beta_distribution = th.distributions.Beta(0.5, 0.5)
            lam = beta_distribution.sample((batch_size,)).cuda()
            lam = lam.view(-1, 1)

            # 对特征和伪标签进行插值
            index = th.randperm(batch_size).cuda()
            features_interpolated = lam * features + (1 - lam) * features[index]
            pseudo_labels_interpolated = lam * mem_cls_weighted + (1 - lam) * mem_cls_weighted[index]

            # 通过模型的头部获得插值特征的预测
            logits_interpolated = model.head.fc(features_interpolated)

            # 使用 softmax 获得插值特征的预测概率
            predicted_probs_interpolated = F.softmax(logits_interpolated, dim=1)

            # 计算交叉熵损失
            consistency_loss = -th.mean(
                th.sum(pseudo_labels_interpolated * th.log(predicted_probs_interpolated + 1e-10), dim=1))

            # 使用LPIPS官方函数计算感知损失
            x1_lpips = denormalize_and_prepare_for_lpips(x1)
            x_lpips = denormalize_and_prepare_for_lpips(x)
            lpips_loss = lpips_loss_fn(x1_lpips, x_lpips).mean()

            # # 计算 norm 损失（不添加噪声，直接使用 x）
            # t = th.zeros(x.size(0), dtype=th.long, device=x.device)
            # out_mean_variance = diffusion.p_mean_variance(
            #     model_diffusion,
            #     x,
            #     t,
            #     clip_denoised=args.clip_denoised,
            #     denoised_fn=None,
            #     model_kwargs={"ref_img": x.clone()},
            # )
            # pred_xstart = out_mean_variance["pred_xstart"]
            #
            # shape = x.shape
            # shape_u = (shape[0], 3, shape[2], shape[3])
            # shape_d = (shape[0], 3, int(shape[2] / args.D), int(shape[3] / args.D))
            #
            # difference = resize(resize(x.clone(), scale_factors=1.0 / args.D, out_shape=shape_d),
            #                     scale_factors=args.D, out_shape=shape_u) - \
            #              resize(resize(pred_xstart.clone(), scale_factors=1.0 / args.D, out_shape=shape_d),
            #                     scale_factors=args.D, out_shape=shape_u)
            #
            # norm = th.linalg.norm(difference)

            # 计算总损失
            # a = 1
            # b = 0.3
            # c = 0.2
            # total_loss = a * consistency_loss + b * loss_im + c * norm
            #
            # # 将损失除以累积步数
            # loss = total_loss / accumulation_steps
            #
            # # 反向传播
            # loss.backward()
            #
            # # 更新样本 x
            # with th.no_grad():
            #     x_updated = x - x.grad * args.scale  # 更新后的样本
            #
            # x.grad.zero_()
            #
            # # 保存更新后的输入，保存中间域数据
            # updated_inputs_list.append(x_updated.clone().detach().cpu())
            #
            # # 更新计数器
            # step_counter += 1
            #
            #
            # # 仅当未达到最大更新样本数时，才更新模型参数
            # if updated_sample_count < max_update_samples:
            #     # 在达到累积步数时更新模型参数
            #     if step_counter % accumulation_steps == 0:
            #         optimizer.step()
            #         optimizer.zero_grad()
            #         step_counter = 0  # 重置计数器
            #
            #     # 更新已更新的样本数量
            #     updated_sample_count += batch_size
            #
            # # 更新 Memory Bank（动量更新）
            # mem_fea[batch_indices] = 0.4 * mem_fea[batch_indices] + 0.6 * features.detach()
            # mem_fea[batch_indices] = mem_fea[batch_indices] / th.norm(mem_fea[batch_indices], p=2, dim=1,
            #                                                           keepdim=True)
            # mem_cls[batch_indices] = 0.4 * mem_cls[batch_indices] + 0.6 * outputs_target.detach()
            #
            # # 更新进度条，增加处理过的样本数
            # pbar.update(batch_size)

            # 定义系数
            a = 1
            b = 0.3
            c = 1000

            # 用于更新模型的损失
            model_loss = a * consistency_loss + b * loss_im

            # 用于更新样本的损失
            sample_loss = a * consistency_loss + c * lpips_loss

            # 在更新样本之前确保梯度图的建立
            x.requires_grad = True

            optimizer.zero_grad()

            if x.grad is not None:
                x.grad.zero_()

            # 对 sample_loss backward, 保留计算图
            sample_loss.backward(retain_graph=True)

            # 使用 x.grad 更新样本
            with th.no_grad():
                x_updated = x - x.grad * 6
            x.grad.zero_()

            updated_inputs_list.append(x_updated.clone().detach().cpu())

            # 再对 model_loss backward 来更新模型参数
            model_loss.backward()
            optimizer.step()

            # 更新 Memory Bank（动量更新）
            mem_fea[batch_indices] = 0.8 * mem_fea[batch_indices] + 0.2 * features.detach()
            mem_fea[batch_indices] = mem_fea[batch_indices] / th.norm(mem_fea[batch_indices], p=2, dim=1, keepdim=True)
            mem_cls[batch_indices] = 0.8 * mem_cls[batch_indices] + 0.2 * outputs_target.detach()

            pbar.update(batch_size)

    dist.barrier()

    # 在训练结束后，保存更新后的样本用于验证
    evaluate_model(model, model_frozen,model_frozen,
                   original_inputs_list1, original_inputs_list2, updated_inputs_list,
                   labels_list, args)

def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,  # 您的硬件限制下的最大 batch_size
        D=32,  # 缩放因子
        N=50,
        scale=1,
        use_ddim=False,
        base_samples="",
        model_path="",
        save_dir="",
        save_latents=False,
        corruption="shot_noise",
        severity=5,
        classifier_config_path="",  # 模型配置路径
        pretrained_weights_path="",  # 模型权重路径
        data_dir1="",  # 原图像路径
        data_dir2="",  # 去噪图像路径
        max_samples=30000,  # 要加载的最大样本数
        seed=42,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()
