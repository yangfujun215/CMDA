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

logger.log("纯source_net、source_classifier。计算除天气外其他域。不是SDA模型")

# 定义特征提取器类，用于从中间层提取特征
# LPIPS 损失实现
# class LPIPSLoss(nn.Module):
#     def __init__(self, model):
#         super(LPIPSLoss, self).__init__()
#         self.model = model
#
#     def forward(self, x1, x2):
#         # 提取特征
#         with th.no_grad():
#           features_x1 = self.model.backbone(x1)[-1]
#           features_x2 = self.model.backbone(x2)[-1]
#
#         # 归一化特征
#         features_x1 = F.normalize(features_x1, p=2, dim=1)
#         features_x2 = F.normalize(features_x2, p=2, dim=1)
#
#         # 计算 L2 距离
#         lpips_loss = th.mean((features_x1 - features_x2) ** 2)
#
#         return lpips_loss

# 定义图像归一化参数

# class LPIPSLoss(nn.Module):
#     def __init__(self, model, weights=None):
#         super(LPIPSLoss, self).__init__()
#         self.model = model
#         self.weights = weights  # 每一层的权重，默认为None
#
#         # 如果没有提供权重，则使用默认的权重（一般是学习到的或预定义的）
#         if self.weights is None:
#             self.weights = [1.0] * 4  # 假设backbone有多个层，使用默认权重
#
#     def forward(self, x1, x2):
#         # 提取所有层的特征
#         # with th.no_grad():
#         features_x1 = self.model.backbone(x1)
#         features_x2 = self.model.backbone(x2)
#
#         # 初始化损失
#         lpips_loss = 0.0
#         num_layers = len(features_x1)
#
#         # 遍历每一层的特征，计算每层的LPIPS损失
#         for i in range(num_layers):
#             # 归一化特征
#             feat_x1 = F.normalize(features_x1[i], p=2, dim=1)
#             feat_x2 = F.normalize(features_x2[i], p=2, dim=1)
#
#             # 计算 L2 距离
#             layer_loss = th.mean((feat_x1 - feat_x2) ** 2)
#
#             # 计算加权损失
#             lpips_loss += self.weights[i] * layer_loss
#
#         # 返回加权损失
#         return lpips_loss

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)

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

    # 自定义数据集，返回图像对和标签
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
            return (img1, img2), label1, idx    # 返回样本和索引

    # 创建成对的数据集
    paired_dataset = PairedDataset(dataset1, dataset2)

    # 创建数据加载器
    data_loader = th.utils.data.DataLoader(paired_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    return data_loader

# 计算熵函数
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

    return sum_logits, sum_logits2, sum_logits3,sum_logits4, entropy_sum_logits, confidence_sum_logits, confidence_sum_logits2

def evaluate_model(source_net, source_classifier,
                   source_net_frozen2, source_classifier_frozen2,
                   original_inputs_list1, original_inputs_list2, updated_inputs_list,
                   labels_list, args):
    logger.log("Evaluating models on original and updated data...")

    # 将模型设置为评估模式
    source_net.eval()
    source_classifier.eval()
    source_net_frozen2.eval()
    source_classifier_frozen2.eval()

    # 获取模型所在的设备
    device = next(source_net.parameters()).device

    all_preds_trainable = []
    all_preds_frozen = []
    all_preds_frozen2 = []
    all_preds_ensemble_sum = []
    all_preds_ensemble_sum2 = []
    all_preds_ensemble_sum3 = []
    all_preds_ensemble_sum4 = []
    all_preds_ensemble_entropy_sum = []
    all_preds_ensemble_confidence_sum = []
    all_preds_ensemble_confidence_sum2 = []
    all_labels = []

    with th.no_grad():
        for original_x1, original_x2, updated_x2, labels in zip(original_inputs_list1, original_inputs_list2, updated_inputs_list, labels_list):
            original_x1 = original_x1.to(device)
            original_x2 = original_x2.to(device)
            updated_x2 = updated_x2.to(device)
            labels = labels.to(device)

            # 更新后的中间域模型，处理中间域数据
            features_trainable = source_net.backbone(updated_x2)[-1]
            # features_trainable = features_trainable / th.norm(features_trainable, p=2, dim=1, keepdim=True)
            logits_trainable = source_classifier(features_trainable)
            class_probs_x1 = F.softmax(logits_trainable, dim=1)
            preds_trainable = class_probs_x1.argmax(dim=1)

            # 原模型处理虚拟域数据
            features_frozen = source_net.backbone(original_x2)[-1]
            # features_frozen = features_frozen / th.norm(features_frozen, p=2, dim=1, keepdim=True)
            logits_frozen = source_classifier(features_frozen)
            class_probs_x2 = F.softmax(logits_frozen, dim=1)
            preds_frozen = class_probs_x2.argmax(dim=1)

            # 原模型处理目标域数据
            features_frozen2 = source_net_frozen2.backbone(original_x1)[-1]
            # features_frozen2 = features_frozen2 / th.norm(features_frozen2, p=2, dim=1, keepdim=True)
            logits_frozen2 = source_classifier_frozen2(features_frozen2)
            class_probs_x3 = F.softmax(logits_frozen2, dim=1)
            preds_frozen2 = class_probs_x3.argmax(dim=1)

            # 获取所有模式的集成结果
            y_ensemble_sum, y_ensemble_sum2, y_ensemble_sum3,y_ensemble_sum4, y_ensemble_entropy_sum, y_ensemble_confidence_sum, y_ensemble_confidence_sum2 = ensemble_logits(class_probs_x1, class_probs_x2, class_probs_x3)
            final_pred_sum = y_ensemble_sum.argmax(dim=1)
            final_pred_sum2 = y_ensemble_sum2.argmax(dim=1)
            final_pred_sum3 = y_ensemble_sum3.argmax(dim=1)
            final_pred_sum4 = y_ensemble_sum4.argmax(dim=1)
            final_pred_entropy_sum = y_ensemble_entropy_sum.argmax(dim=1)
            final_pred_confidence_sum = y_ensemble_confidence_sum.argmax(dim=1)
            final_pred_confidence_sum2 = y_ensemble_confidence_sum2.argmax(dim=1)

            # 将当前批次的预测和标签添加到列表中
            all_preds_trainable.append(preds_trainable.cpu())
            all_preds_frozen.append(preds_frozen.cpu())
            all_preds_frozen2.append(preds_frozen2.cpu())
            all_preds_ensemble_sum.append(final_pred_sum.cpu())
            all_preds_ensemble_sum2.append(final_pred_sum2.cpu())
            all_preds_ensemble_sum3.append(final_pred_sum3.cpu())
            all_preds_ensemble_sum4.append(final_pred_sum4.cpu())
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

    # # 设置数据路径
    args.data_dir1 = f"/media/shared_space/yrzhen/data/ImageNet-C/{args.corruption}/{args.severity}"
    args.data_dir2 = f"/media/shared_space/wuyanzu/ImageNet-C-Syn/{args.corruption}/{args.severity}"
    logger.log(f"当前域： (corruption): {args.corruption}, severity: {args.severity}")

    # # imagenet-w路径
    # args.data_dir1 = f"/media/shared_space/wuyanzu/imagenet-w/{args.corruption}"
    # args.data_dir2 = f"/media/shared_space/wuyanzu/ImageNet-W-Syn/{args.corruption}"
    # logger.log(f"当前域： (corruption): {args.corruption}, severity: {args.severity}")

    logger.log("加载数据...")
    data_loader = load_paired_data(
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
    source_classifier = Classifier_covnextT(num_classes=1000).to('cuda')

    # 创建只用于推理的模型实例
    source_net_frozen = build_classifier(cfg.model)
    source_net_frozen.to('cuda')
    source_classifier_frozen = Classifier_covnextT(num_classes=1000).to('cuda')

    source_net_frozen2 = build_classifier(cfg.model)
    source_net_frozen2.to('cuda')
    source_classifier_frozen2 = Classifier_covnextT(num_classes=1000).to('cuda')

    # 加载预训练权重
    checkpoint = th.load(args.pretrained_weights_path, map_location='cuda')
    model_state_dict = checkpoint['state_dict']

    # 分离 backbone 和 head 的权重
    backbone_state_dict = {}
    head_state_dict = {}
    for k, v in model_state_dict.items():
        if k.startswith('backbone.'):
            backbone_state_dict[k.replace('backbone.', '')] = v
        elif k.startswith('head.'):
            head_state_dict[k.replace('head.', '')] = v

    # 加载 backbone 的权重到三个模型中
    for model2 in [source_net, source_net_frozen, source_net_frozen2]:
        model2.backbone.load_state_dict(backbone_state_dict, strict=True)

    # 处理 head 的权重名称，将其映射到 source_classifier 的参数名称
    classifier_state_dict = {}
    for k, v in head_state_dict.items():
        if k == 'fc.weight' or k == 'classifier.weight':
            classifier_state_dict['fc.weight'] = v
        elif k == 'fc.bias' or k == 'classifier.bias':
            classifier_state_dict['fc.bias'] = v

    # 加载 head（分类器）的权重到三个分类器中
    for classifier in [source_classifier, source_classifier_frozen, source_classifier_frozen2]:
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

    learning_rate = 1e-5  # 设置学习率

    # 使用 AdamW 优化器，分层学习率
    optimizer_net = AdamW(source_net.parameters(), lr=learning_rate , weight_decay=1e-5)
    optimizer_classifier = AdamW(source_classifier.parameters(), lr=learning_rate, weight_decay=1e-5)

    # 使用官方LPIPS度量
    lpips_loss_fn = lpips.LPIPS(net='vgg').to('cuda')  # 可以根据需要改成'alex','squeeze','vgg'

    # weights = [0.1, 0.3, 0.4, 1]  # 四层
    # lpips_loss_fn = LPIPSLoss(source_net, weights=weights).to('cuda')

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
    mem_fea = None
    mem_cls = None

    with tqdm(total=total_samples, desc="Initializing memory bank", unit="sample") as pbar:
        for idx, batch_data in enumerate(data_loader):
            (images1, images2), labels, indices = batch_data
            x2 = images1.to('cuda')
            indices = indices.to('cuda')
            batch_size = x2.size(0)
            with th.no_grad():
                features_x2 = source_net.backbone(x2)[-1]
                features_x2 = features_x2 / th.norm(features_x2, p=2, dim=1, keepdim=True)
                logits_x2 = source_classifier(features_x2)
                class_probs_x2 = F.softmax(logits_x2, dim=1)
            if mem_fea is None:
                feature_dim = features_x2.size(1)
                num_classes = class_probs_x2.size(1)
                mem_fea = th.zeros(total_samples, feature_dim).cuda()
                mem_cls = th.zeros(total_samples, num_classes).cuda()
            mem_fea[indices] = features_x2.detach()
            mem_cls[indices] = class_probs_x2.detach()
            pbar.update(batch_size)

    # 重新创建数据加载器，以便从头开始遍历数据
    data_loader = load_paired_data(
        args.data_dir1,
        args.data_dir2,
        args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        max_samples=args.max_samples
    )

    logger.log("开始处理数据...")

    # 初始化优化器和计数器
    optimizer_net.zero_grad()
    optimizer_classifier.zero_grad()

    with tqdm(total=total_samples, desc="Processing data", unit="sample") as pbar:
        for idx, batch_data in enumerate(data_loader):
            (images1, images2), labels, batch_indices = batch_data
            x1 = images1.to('cuda')  # 目标域数据，用于后续计算正确率
            x2 = images2.to('cuda')  # 虚拟域数据
            x_orig = x1.clone().detach()  # 目标域数据
            x = x2.clone().detach()  # 利用虚拟域数据进行后续模型更新，然后更新模型到中间域
            x.requires_grad = True  # 确保x为可训练叶子节点

            labels = labels.to('cuda').detach()
            batch_indices = batch_indices.to('cuda')

            batch_size = x.size(0)

            # 保存原始输入到列表
            original_inputs_list1.append(x_orig.clone().detach().cpu())
            original_inputs_list2.append(x2.clone().detach().cpu())
            labels_list.append(labels.clone().detach().cpu())

            # 模型设置为训练模式
            source_net.train()
            source_classifier.train()

            # 对于数据，计算特征和输出
            features_current = source_net.backbone(x)[-1]
            logits_current = source_classifier(features_current)
            softmax_out = th.nn.Softmax(dim=1)(logits_current)

            # 应用温度锐化
            temperature = 0.8
            sharpened_softmax_out = temperature_sharpening(softmax_out, temperature)

            # 计算信息熵损失
            loss_emin = th.mean(Entropy(sharpened_softmax_out))
            msoftmax_out = sharpened_softmax_out.mean(dim=0)
            loss_gemin = th.sum(-msoftmax_out * th.log(msoftmax_out + 1e-10))
            loss_im = loss_emin - loss_gemin

            # 提取特征（在no_grad中仅仅是为了减少计算量，不影响对x的梯度计算，因为这里没有对x进行操作）
            with th.no_grad():
                features = features_current / th.norm(features_current, p=2, dim=1, keepdim=True)
                outputs_target = sharpened_softmax_out

            # 计算与 Memory Bank 的距离
            dis = -th.mm(features.detach(), mem_fea.t())

            # 排除自身（将自身距离设为最大值）
            for i in range(batch_size):
                dis[i, batch_indices[i]] = dis.max()

            # 获取最近的 k 个邻居
            k = min(5, mem_fea.size(0))
            _, p1 = th.topk(dis, k=k, dim=1, largest=False)
            w = th.zeros(batch_size, mem_fea.size(0)).cuda()
            for i in range(batch_size):
                w[i, p1[i]] = 1 / k

            # 计算加权分类输出，获取伪标签
            mem_cls_weighted = th.mm(w, mem_cls)
            weight_, pred = th.max(mem_cls_weighted, dim=1)

            # 插值一致性损失
            beta_distribution = th.distributions.Beta(0.5, 0.5)
            lam = beta_distribution.sample((batch_size,)).cuda()
            lam = lam.view(-1, 1)

            index = th.randperm(batch_size).cuda()
            features_interpolated = lam * features + (1 - lam) * features[index]
            pseudo_labels_interpolated = lam * mem_cls_weighted + (1 - lam) * mem_cls_weighted[index]

            # 通过分类器获得插值特征的预测
            logits_interpolated = source_classifier(features_interpolated)

            # 使用 softmax 获得插值特征的预测概率
            predicted_probs_interpolated = F.softmax(logits_interpolated, dim=1)

            # 计算交叉熵损失
            consistency_loss = -th.mean(
                th.sum(pseudo_labels_interpolated * th.log(predicted_probs_interpolated + 1e-10), dim=1))

            # # 计算 LPIPS 损失
            # # 修改处：这里使用 x（带有梯度）代替 x2，以保证对 x 有梯度传递
            # lpips_loss = lpips_loss_fn(x1, x)

            # 使用LPIPS官方函数计算感知损失
            x1_lpips = denormalize_and_prepare_for_lpips(x1)
            x_lpips = denormalize_and_prepare_for_lpips(x)
            lpips_loss = lpips_loss_fn(x1_lpips, x_lpips).mean()

            # 定义系数
            a = 1
            b = 0.3  # 0.1 0.3 0.5 0.7 0.9
            c = 0.1 # 0.1 1 10 100 1000

            # 用于更新模型的损失
            model_loss = a * consistency_loss + b * loss_im

            # 用于更新样本的损失
            sample_loss = a * consistency_loss + c * lpips_loss

            # 在更新样本之前确保梯度图的建立
            x.requires_grad = True

            # 先使用 sample_loss 来更新样本，再使用 model_loss 来更新模型
            optimizer_net.zero_grad()
            optimizer_classifier.zero_grad()

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
            optimizer_net.step()
            optimizer_classifier.step()

            # 更新 Memory Bank（动量更新）
            mem_fea[batch_indices] = 0.1 * mem_fea[batch_indices] + 0.9 * features.detach()
            mem_fea[batch_indices] = mem_fea[batch_indices] / th.norm(mem_fea[batch_indices], p=2, dim=1, keepdim=True)
            mem_cls[batch_indices] = 0.1 * mem_cls[batch_indices] + 0.9 * outputs_target.detach()

            pbar.update(batch_size)

    # # 指定文件夹路径
    # save_path_net = '/media/shared_space/wuyanzu/DDA-main/ckpt/source_net_finetuned.pth'
    # save_path_classifier = '/media/shared_space/wuyanzu/DDA-main/ckpt/source_classifier_finetuned.pth'
    # print("保存模型权重")
    # # 保存模型
    # th.save(source_net.state_dict(), save_path_net)
    # th.save(source_classifier.state_dict(), save_path_classifier)

    dist.barrier()

    # 在训练结束后，保存更新后的样本用于验证
    evaluate_model(source_net, source_classifier,
                   source_net_frozen, source_classifier_frozen,
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
        classifier_config_path="",  # resnet模型配置路径
        pretrained_weights_path="",  # resnet权重路径
        data_dir1="",  # 原图像路径
        data_dir2="",  # 去噪图像路径
        max_samples=50000,  # 要加载的最大样本数
        seed=42,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()
