import argparse
import os
import random

from tqdm import tqdm

import numpy as np
import torch as th
import torch.distributed as dist
from torch.nn import DataParallel

from torch.optim import SGD

from mmcv import Config
from mmcls.models import build_classifier
from mmcv.runner import load_checkpoint

from mmcv import Config
from mmcls.models import build_classifier
from image_adapt.model_use import Classifier

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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 定义图像归一化参数
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)

# 数据加载函数
# def load_denoised_data(data_dir, batch_size, image_size, class_cond=False, max_samples=None):
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
#     dataset = datasets.ImageFolder(root=data_dir, transform=transform)
#     if max_samples is not None:
#         dataset.samples = dataset.samples[:max_samples]
#
#     data_loader = th.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
#
#     for images, labels in data_loader:
#         model_kwargs = {}
#         model_kwargs["ref_img"] = images
#
#         yield model_kwargs, labels
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

    # 加载两个数据集
    dataset1 = datasets.ImageFolder(root=data_dir1, transform=transform)
    dataset2 = datasets.ImageFolder(root=data_dir2, transform=transform)

    # 如果需要，截取最大样本数
    if max_samples is not None:
        dataset1.samples = dataset1.samples[:max_samples]
        dataset2.samples = dataset2.samples[:max_samples]

    # 确保两个数据集的样本数量相同
    assert len(dataset1) == len(dataset2), "两个数据集的样本数量不一致"

    # 定义一个自定义数据集，返回图像对和标签
    class PairedDataset(th.utils.data.Dataset):
        def __init__(self, dataset1, dataset2):
            self.dataset1 = dataset1
            self.dataset2 = dataset2

        def __len__(self):
            return len(self.dataset1)

        def __getitem__(self, idx):
            img1, label1 = self.dataset1[idx]
            img2, label2 = self.dataset2[idx]
            # 可选：检查标签是否一致
            assert label1 == label2, "样本标签不匹配"
            return (img1, img2), label1

    # 创建成对的数据集
    paired_dataset = PairedDataset(dataset1, dataset2)

    # 创建数据加载器
    data_loader = th.utils.data.DataLoader(paired_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 迭代数据加载器，返回模型输入和标签
    for (images1, images2), labels in data_loader:
        model_kwargs = {}
        model_kwargs["img1"] = images1
        model_kwargs["img2"] = images2

        yield model_kwargs, labels

def evaluate_model(
                   source_net_frozen, source_classifier_frozen,
                   original_inputs_list1,
                   original_inputs_list2,
                   labels_list, args):
    logger.log("Evaluating models on original and updated data...")

    # 将模型设置为评估模式

    source_net_frozen.eval()
    source_classifier_frozen.eval()

    # 获取模型所在的设备
    device = next(source_net_frozen.parameters()).device

    all_preds_frozen = []
    all_labels = []

    with th.no_grad():
        for original_x, updated_x, labels in zip(original_inputs_list1, original_inputs_list2, labels_list):
            original_x = original_x.to(device)
            updated_x = updated_x.to(device)
            labels = labels.to(device)

            features_trainable = source_net_frozen.backbone(updated_x)[-1]
            features_trainable = source_net_frozen.neck(features_trainable)
            logits_trainable = source_classifier_frozen(features_trainable)
            preds_trainable = logits_trainable.argmax(dim=1)

            features_frozen = source_net_frozen.backbone(original_x)[-1]
            features_frozen = source_net_frozen.neck(features_frozen)
            logits_frozen = source_classifier_frozen(features_frozen)
            preds_frozen = logits_frozen.argmax(dim=1)

            # 对两个模型的 logits 进行加和集成
            y_ensemble = logits_trainable + logits_frozen
            final_pred = y_ensemble.argmax(dim=1)

            # 将当前批次的预测和标签添加到列表中
            all_preds_frozen.append(preds_frozen.cpu())
            all_labels.append(labels.cpu())

    # # 将所有预测和标签拼接起来
    all_preds_frozen = th.cat(all_preds_frozen)
    all_labels = th.cat(all_labels)

    # 计算准确率
    accuracy_frozen = (all_preds_frozen == all_labels).float().mean().item()

    # 输出结果
    print(f"Frozen Model Accuracy: {accuracy_frozen * 100:.8f}%")


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

    # 输出当前破坏类型
    logger.log(f"当前域： (corruption): {args.corruption}")

    logger.log("加载数据...")
    data = load_paired_data(
        args.data_dir1,
        args.data_dir2,
        args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        max_samples=args.max_samples  # 添加最大样本数参数
    )

    # 获取数据集的总样本数
    dataset1 = datasets.ImageFolder(root=args.data_dir1)
    total_samples = len(dataset1)  # 获取实际的样本总数
    if args.max_samples is not None:
        total_samples = min(total_samples, args.max_samples)

    # 加载模型配置和权重
    cfg = Config.fromfile(args.classifier_config_path)
    source_net = build_classifier(cfg.model)
    source_net.to('cuda')
    source_classifier = Classifier(num_classes=1000).to('cuda')

    # 创建只用于推理的模型实例
    source_net_frozen = build_classifier(cfg.model)
    source_net_frozen.to('cuda')
    source_classifier_frozen = Classifier(num_classes=1000).to('cuda')

    source_net_frozen2 = build_classifier(cfg.model)
    source_net_frozen2.to('cuda')
    source_classifier_frozen2 = Classifier(num_classes=1000).to('cuda')

    # 加载预训练权重
    checkpoint = th.load(args.pretrained_weights_path, map_location='cuda')
    model_state_dict = checkpoint['state_dict']

    # 分离 backbone、neck 和 head 的权重
    backbone_state_dict = {}
    neck_state_dict = {}
    head_state_dict = {}
    for k, v in model_state_dict.items():
        if k.startswith('backbone.'):
            backbone_state_dict[k.replace('backbone.', '')] = v
        elif k.startswith('neck.'):
            neck_state_dict[k.replace('neck.', '')] = v
        elif k.startswith('head.'):
            head_state_dict[k.replace('head.', '')] = v

    # 加载 backbone 和 neck 的权重到两个模型中
    for model2 in [source_net, source_net_frozen, source_net_frozen2 ]:
        model2.backbone.load_state_dict(backbone_state_dict, strict=True)
        model2.neck.load_state_dict(neck_state_dict, strict=True)

    # 处理 head 的权重名称，将其映射到 source_classifier 的参数名称
    classifier_state_dict = {}
    for k, v in head_state_dict.items():
        if k == 'fc.weight':
            classifier_state_dict['fc.weight'] = v
        elif k == 'fc.bias':
            classifier_state_dict['fc.bias'] = v

    # 加载 head（分类器）的权重到两个分类器中
    for classifier in [source_classifier, source_classifier_frozen, source_classifier_frozen2 ]:
        classifier.load_state_dict(classifier_state_dict, strict=True)

    # 冻结不用于训练的模型的参数
    for param in source_net_frozen.parameters():
        param.requires_grad = False
    for param in source_classifier_frozen.parameters():
        param.requires_grad = False

    for param in source_net_frozen2.parameters():
        param.requires_grad = False
    for param in source_classifier_frozen2.parameters():
        param.requires_grad = False


    logger.log("源模型导入成功")

    # 保存用于评估的输入和标签
    original_inputs_list1 = []
    original_inputs_list2 = []
    labels_list = []

    logger.log("开始处理数据...")
    with tqdm(total=total_samples, desc="Processing data", unit="sample") as pbar:
        for idx, (model_kwargs, labels) in enumerate(data):
            x1 = model_kwargs["img1"]
            x2 = model_kwargs["img2"]
            labels = labels.detach()

            # 保存原始输入到列表
            original_inputs_list1.append(x1.clone().detach())
            original_inputs_list2.append(x2.clone().detach())
            labels_list.append(labels.clone().detach())

            pbar.update(x1.size(0))  # 更新进度条

    evaluate_model(
        source_net, source_classifier, # 传入集成的模型实例
        original_inputs_list1,
        original_inputs_list2,
        labels_list,
        args
    )

def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=4,
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
        classifier_config_path="",  # resnet模型配置路径
        pretrained_weights_path="",  # resnet权重路径
        data_dir1="/media/shared_space/yrzhen/data/ImageNet-C/brightness/5",  # 原图像路径
        data_dir2="/media/shared_space/wuyanzu/ImageNet-C-Syn/brightness/5",  # 去噪图像路径
        max_samples=50000,  # 要加载的最大样本数
        seed=42,
        ensemble='sum',  # 默认的集成方式
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()
