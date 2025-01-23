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
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

logger.log("纯source_net、source_classifier。计算除天气外其他域。不是SDA模型")

# 定义图像归一化参数
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
    sum_logits2 = (logits1 + logits3)/2
    sum_logits3 = (logits1 + logits2)/2
    sum_logits4 = (logits2 + logits3)/2

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
            features_frozen = source_net_frozen2.backbone(original_x2)[-1]
            # features_frozen = features_frozen / th.norm(features_frozen, p=2, dim=1, keepdim=True)
            logits_frozen = source_classifier_frozen2(features_frozen)
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

    args.data_dir1 = f"/media/shared_space/yrzhen/data/ImageNet-C/{args.corruption}/{args.severity}"
    args.data_dir2 = f"/media/shared_space/wuyanzu/ImageNet-C-Syn/{args.corruption}/{args.severity}"
    logger.log(f"当前域： (corruption): {args.corruption}, severity: {args.severity}")

    # # imagenet-w路径
    # args.data_dir1 = f"/home/yfj/imagenet-w/{args.corruption}/{args.severity}"
    # args.data_dir2 = f"/home/yfj/ImageNet-W-Syn/{args.corruption}/{args.severity}"
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

    # # 冻结 source_net 的前三个阶段
    # for i, stage in enumerate(source_net.backbone.stages):
    #     if i < 3:  # 冻结前三个阶段
    #         for param in stage.parameters():
    #             param.requires_grad = False
    # logger.log("冻住前三层")

    logger.log("源模型导入成功")

    learning_rate = 1e-5  # 设置学习率

    # 使用 AdamW 优化器，分层学习率
    # optimizer_net = AdamW(source_net.parameters(), lr=learning_rate , weight_decay=1e-5)
    # optimizer_classifier = AdamW(source_classifier.parameters(), lr=learning_rate, weight_decay=1e-5)

    optimizer_net = AdamW(source_net.parameters(), lr=learning_rate, weight_decay=1e-5)
    optimizer_classifier = AdamW(source_classifier.parameters(), lr=learning_rate, weight_decay=1e-5)
    # # 使用 SGD 优化器
    # optimizer_net = SGD(source_net.parameters(), lr=learning_rate)
    # optimizer_classifier = SGD(source_classifier.parameters(), lr=learning_rate)

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

    # # 初始化 Memory Bank
    # with tqdm(total=total_samples, desc="Initializing memory bank", unit="sample") as pbar:
    #     for idx, batch_data in enumerate(data_loader):
    #         (images1, images2), labels, indices = batch_data
    #         x1 = images1.to('cuda')
    #         indices = indices.to('cuda')
    #         batch_size = x1.size(0)
    #         with th.no_grad():
    #             features_x1 = source_net_frozen2.backbone(x1)[-1]
    #             features_x1 = features_x1 / th.norm(features_x1, p=2, dim=1, keepdim=True)
    #             logits_x1 = source_classifier_frozen2(features_x1)
    #             class_probs_x1 = F.softmax(logits_x1, dim=1)
    #         if mem_fea is None:
    #             feature_dim = features_x1.size(1)
    #             num_classes = class_probs_x1.size(1)
    #             mem_fea = th.zeros(total_samples, feature_dim).cuda()
    #             mem_cls = th.zeros(total_samples, num_classes).cuda()
    #         mem_fea[indices] = features_x1.detach()
    #         mem_cls[indices] = class_probs_x1.detach()
    #         pbar.update(batch_size)

    # 初始化 Memory Bank
    with tqdm(total=total_samples, desc="Initializing memory bank", unit="sample") as pbar:
        for idx, batch_data in enumerate(data_loader):
            (images1, images2), labels, indices = batch_data
            x2 = images2.to('cuda')
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

    logger.log("Memory Bank 初始化完成，使用目标域数据进行memory bank的初始化")

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

    # 定义梯度累积步数
    accumulation_steps = args.effective_batch_size // args.batch_size

    # 初始化优化器和计数器
    optimizer_net.zero_grad()
    optimizer_classifier.zero_grad()
    step_counter = 0

    # **添加变量，跟踪已更新的样本数量**
    updated_sample_count = 0  # 已用于更新模型参数的样本数量
    max_update_samples = args.max_update_samples  # 从参数中获取最大更新样本数

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
            source_net.train()
            source_classifier.train()

            # 对于数据，计算特征和输出
            features_current = source_net.backbone(x)[-1]
            logits_current = source_classifier(features_current)
            softmax_out = th.nn.Softmax(dim=1)(logits_current)

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
                features = features_current / th.norm(features_current, p=2, dim=1, keepdim=True)
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

            # 通过分类器获得插值特征的预测
            logits_interpolated = source_classifier(features_interpolated)

            # 使用 softmax 获得插值特征的预测概率
            predicted_probs_interpolated = F.softmax(logits_interpolated, dim=1)

            # 计算交叉熵损失
            consistency_loss = -th.mean(
                th.sum(pseudo_labels_interpolated * th.log(predicted_probs_interpolated + 1e-10), dim=1))

            # 计算 norm 损失（不添加噪声，直接使用 x）
            t = th.zeros(x.size(0), dtype=th.long, device=x.device)
            out_mean_variance = diffusion.p_mean_variance(
                model_diffusion,
                x,
                t,
                clip_denoised=args.clip_denoised,
                denoised_fn=None,
                model_kwargs={"ref_img": x.clone()},
            )
            pred_xstart = out_mean_variance["pred_xstart"]

            shape = x.shape
            shape_u = (shape[0], 3, shape[2], shape[3])
            shape_d = (shape[0], 3, int(shape[2] / args.D), int(shape[3] / args.D))

            difference = resize(resize(x.clone(), scale_factors=1.0 / args.D, out_shape=shape_d),
                                scale_factors=args.D, out_shape=shape_u) - \
                         resize(resize(pred_xstart.clone(), scale_factors=1.0 / args.D, out_shape=shape_d),
                                scale_factors=args.D, out_shape=shape_u)

            norm = th.linalg.norm(difference)

            # 计算两个损失
            a = 1
            b = 0.3
            c = 0.01

            # 计算用于更新模型的损失（consistency_loss + loss_im）
            model_loss = a * consistency_loss + b * loss_im

            # 计算用于更新样本的损失（consistency_loss + c * norm）
            sample_loss = a * consistency_loss + c * norm

            # 将损失除以累积步数
            model_loss /= accumulation_steps

            # 反向传播（使用模型损失来更新模型参数）
            model_loss.backward()  # 保留计算图

            # 更新样本
            with th.no_grad():
                grad_sample = th.autograd.grad(sample_loss, x, retain_graph=True)[0]  # 计算梯度
                x_updated = x - grad_sample * 6  # 更新样本

            # ---  开始删除中间变量 ---
            del grad_sample
            del sample_loss
            # ---  结束删除中间变量 ---

            x.grad.zero_()

            # 保存更新后的输入，保存中间域数据
            updated_inputs_list.append(x_updated.clone().detach().cpu())

            # 更新计数器
            step_counter += 1

            # **仅当未达到最大更新样本数时，才更新模型参数**
            if updated_sample_count < max_update_samples:
                # 在达到累积步数时更新模型参数
                if step_counter % accumulation_steps == 0:
                    optimizer_net.step()  # 更新模型参数
                    optimizer_classifier.step()  # 更新分类器参数
                    optimizer_net.zero_grad()  # 清空网络的梯度
                    optimizer_classifier.zero_grad()  # 清空分类器的梯度
                    step_counter = 0  # 重置计数器

                # 更新已更新的样本数量
                updated_sample_count += batch_size

            # # 计算总损失
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
            # # **仅当未达到最大更新样本数时，才更新模型参数**
            # if updated_sample_count < max_update_samples:
            #     # 在达到累积步数时更新模型参数
            #     if step_counter % accumulation_steps == 0:
            #         optimizer_net.step()
            #         optimizer_classifier.step()
            #         optimizer_net.zero_grad()
            #         optimizer_classifier.zero_grad()
            #         step_counter = 0  # 重置计数器
            #
            #     # 更新已更新的样本数量
            #     updated_sample_count += batch_size

            # 更新 Memory Bank（动量更新）
            mem_fea[batch_indices] = 0.4 * mem_fea[batch_indices] + 0.6 * features.detach()
            mem_fea[batch_indices] = mem_fea[batch_indices] / th.norm(mem_fea[batch_indices], p=2, dim=1,
                                                                      keepdim=True)
            mem_cls[batch_indices] = 0.4 * mem_cls[batch_indices] + 0.6 * outputs_target.detach()

            # 更新进度条，增加处理过的样本数
            pbar.update(batch_size)

    # 处理剩余的梯度
    if step_counter != 0 and updated_sample_count < max_update_samples:
        optimizer_net.step()
        optimizer_classifier.step()
        optimizer_net.zero_grad()
        optimizer_classifier.zero_grad()

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
        effective_batch_size=64,  # 想要模拟的更大批次大小
        max_update_samples=50000,  # **新增参数：最大模型更新的样本数量**
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()
