import argparse
import os
import random

import lpips
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
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

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
#         if self.weights is None:
#             self.weights = [1.0] * 4
#
#     def forward(self, x1, x2):
#         # 使用包装后的模型直接 forward 获取特征列表
#         features_x1 = self.model(x1)  # 返回特征列表
#         features_x2 = self.model(x2)  # 返回特征列表
#
#         lpips_loss = 0.0
#         num_layers = len(features_x1)
#
#         for i in range(num_layers):
#             feat_x1 = F.normalize(features_x1[i], p=2, dim=1)
#             feat_x2 = F.normalize(features_x2[i], p=2, dim=1)
#             layer_loss = th.mean((feat_x1 - feat_x2) ** 2)
#             lpips_loss += self.weights[i] * layer_loss
#
#         return lpips_loss

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)

def load_paired_data(data_dir1, data_dir2, batch_size, image_size, class_cond=False, max_samples=None):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255),
        transforms.Normalize(
            mean=img_norm_cfg['mean'],
            std=img_norm_cfg['std']
        )
    ])

    dataset1 = datasets.ImageFolder(root=data_dir1, transform=transform)
    dataset2 = datasets.ImageFolder(root=data_dir2, transform=transform)

    if max_samples is not None:
        dataset1.samples = dataset1.samples[:max_samples]
        dataset2.samples = dataset2.samples[:max_samples]

    assert len(dataset1) == len(dataset2), "两个数据集的样本数量不一致"

    class PairedDataset(th.utils.data.Dataset):
        def __init__(self, dataset1, dataset2):
            self.dataset1 = dataset1
            self.dataset2 = dataset2

        def __len__(self):
            return len(self.dataset1)

        def __getitem__(self, idx):
            img1, label1 = self.dataset1[idx]
            img2, label2 = self.dataset2[idx]
            assert label1 == label2, "样本标签不匹配"
            return (img1, img2), label1, idx

    paired_dataset = PairedDataset(dataset1, dataset2)
    data_loader = th.utils.data.DataLoader(paired_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return data_loader

def Entropy(input_):
    epsilon = 1e-5
    entropy = -input_ * th.log(input_ + epsilon)
    entropy = th.sum(entropy, dim=1)
    return entropy

def temperature_sharpening(preds, temperature=0.5):
    preds = preds ** (1 / temperature)
    preds = preds / preds.sum(dim=1, keepdim=True)
    return preds

def ensemble_logits(logits1, logits2, logits3):
    sum_logits = logits1 + logits2 + logits3
    sum_logits2 = logits1 + logits3
    sum_logits3 = logits1 + logits2
    sum_logits4 = logits2 + logits3

    ent1 = -(logits1.softmax(1) * logits1.log_softmax(1)).sum(1, keepdim=True)
    ent2 = -(logits2.softmax(1) * logits2.log_softmax(1)).sum(1, keepdim=True)
    ent3 = -(logits3.softmax(1) * logits3.log_softmax(1)).sum(1, keepdim=True)
    entropy_sum_logits = logits1 * (ent2 + ent3) + logits2 * (ent1 + ent3) + logits3 * (ent1 + ent2)

    con1 = logits1.softmax(1).max(1, keepdim=True)[0]
    con2 = logits2.softmax(1).max(1, keepdim=True)[0]
    con3 = logits3.softmax(1).max(1, keepdim=True)[0]
    confidence_sum_logits = logits1 * con1 + logits2 * con2 + logits3 * con3
    confidence_sum_logits2 = logits1 * (con2 + con3) + logits2 * (con1 + con3) + logits3 * (con1 + con2)

    return sum_logits, sum_logits2, sum_logits3, sum_logits4, entropy_sum_logits, confidence_sum_logits, confidence_sum_logits2

def evaluate_model(source_net, source_classifier,
                   source_net_frozen2, source_classifier_frozen2,
                   original_inputs_list1, original_inputs_list2, updated_inputs_list,
                   labels_list, args):
    logger.log("Evaluating models on original and updated data...")

    source_net.eval()
    source_classifier.eval()
    source_net_frozen2.eval()
    source_classifier_frozen2.eval()

    device = th.device("cuda" if th.cuda.is_available() else "cpu")

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
            # 不提前to('cuda')，让DataParallel自行分配
            # 这里由于evaluate_model中直接调用source_net_frozen2时，如果source_net_frozen2是DataParallel封装的，我们只需要输入CPU数据即可
            original_x1 = original_x1 # CPU上
            original_x2 = original_x2
            updated_x2 = updated_x2
            labels = labels

            # DataParallel会自动分发数据到多卡
            features_trainable = source_net(updated_x2)[-1]  # wrapper_net(x)返回list，取[-1]
            logits_trainable = source_classifier(features_trainable)
            class_probs_x1 = F.softmax(logits_trainable, dim=1)
            preds_trainable = class_probs_x1.argmax(dim=1)

            features_frozen = source_net_frozen2(original_x2)[-1]
            logits_frozen = source_classifier_frozen2(features_frozen)
            class_probs_x2 = F.softmax(logits_frozen, dim=1)
            preds_frozen = class_probs_x2.argmax(dim=1)

            features_frozen2 = source_net_frozen2(original_x1)[-1]
            logits_frozen2 = source_classifier_frozen2(features_frozen2)
            class_probs_x3 = F.softmax(logits_frozen2, dim=1)
            preds_frozen2 = class_probs_x3.argmax(dim=1)

            y_ensemble_sum, y_ensemble_sum2, y_ensemble_sum3, y_ensemble_sum4, y_ensemble_entropy_sum, y_ensemble_confidence_sum, y_ensemble_confidence_sum2 = ensemble_logits(class_probs_x1, class_probs_x2, class_probs_x3)
            final_pred_sum = y_ensemble_sum.argmax(dim=1)
            final_pred_sum2 = y_ensemble_sum2.argmax(dim=1)
            final_pred_sum3 = y_ensemble_sum3.argmax(dim=1)
            final_pred_sum4 = y_ensemble_sum4.argmax(dim=1)
            final_pred_entropy_sum = y_ensemble_entropy_sum.argmax(dim=1)
            final_pred_confidence_sum = y_ensemble_confidence_sum.argmax(dim=1)
            final_pred_confidence_sum2 = y_ensemble_confidence_sum2.argmax(dim=1)

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

class BackboneWrapper(nn.Module):
    def __init__(self, source_net):
        super().__init__()
        self.source_net = source_net
    def forward(self, x):
        # 返回整个特征列表
        return self.source_net.backbone(x)

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

    # 移除分布式相关初始化
    # dist_util.setup_dist() # 移除
    # logger.configure(dir=args.save_dir) # 保留logger配置可选

    # imagenet-w路径
    args.data_dir1 = f"/media/shared_space/wuyanzu/imagenet-w/{args.corruption}"
    args.data_dir2 = f"/media/shared_space/wuyanzu/ImageNet-W-Syn/{args.corruption}"
    logger.log(f"当前域： (corruption): {args.corruption}, severity: {args.severity}")

    # args.data_dir1 = f"/media/shared_space/yrzhen/data/ImageNet-C/{args.corruption}/{args.severity}"
    # args.data_dir2 = f"/media/shared_space/wuyanzu/ImageNet-C-Syn/{args.corruption}/{args.severity}"
    # logger.log(f"当前域： (corruption): {args.corruption}, severity: {args.severity}")

    logger.log("加载数据...")
    data_loader = load_paired_data(
        args.data_dir1,
        args.data_dir2,
        args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        max_samples=args.max_samples
    )

    dataset1 = datasets.ImageFolder(root=args.data_dir1)
    total_samples = len(dataset1)
    if args.max_samples is not None:
        total_samples = min(total_samples, args.max_samples)

    cfg = Config.fromfile(args.classifier_config_path)
    source_net_raw = build_classifier(cfg.model)
    source_net_raw.to('cuda')
    source_classifier_raw = Classifier_covnextT(num_classes=1000).to('cuda')

    source_net_frozen_raw = build_classifier(cfg.model)
    source_net_frozen_raw.to('cuda')
    source_classifier_frozen_raw = Classifier_covnextT(num_classes=1000).to('cuda')

    source_net_frozen2_raw = build_classifier(cfg.model)
    source_net_frozen2_raw.to('cuda')
    source_classifier_frozen2_raw = Classifier_covnextT(num_classes=1000).to('cuda')

    checkpoint = th.load(args.pretrained_weights_path, map_location='cuda')
    model_state_dict = checkpoint['state_dict']

    backbone_state_dict = {}
    head_state_dict = {}
    for k, v in model_state_dict.items():
        if k.startswith('backbone.'):
            backbone_state_dict[k.replace('backbone.', '')] = v
        elif k.startswith('head.'):
            head_state_dict[k.replace('head.', '')] = v

    for model2 in [source_net_raw, source_net_frozen_raw, source_net_frozen2_raw]:
        model2.backbone.load_state_dict(backbone_state_dict, strict=True)

    classifier_state_dict = {}
    for k, v in head_state_dict.items():
        if k in ['fc.weight', 'classifier.weight']:
            classifier_state_dict['fc.weight'] = v
        elif k in ['fc.bias', 'classifier.bias']:
            classifier_state_dict['fc.bias'] = v

    for classifier_model in [source_classifier_raw, source_classifier_frozen_raw, source_classifier_frozen2_raw]:
        classifier_model.load_state_dict(classifier_state_dict, strict=True)

    for param in source_net_frozen_raw.parameters():
        param.requires_grad = False
    for param in source_classifier_frozen_raw.parameters():
        param.requires_grad = False

    for param in source_net_frozen2_raw.parameters():
        param.requires_grad = False
    for param in source_classifier_frozen2_raw.parameters():
        param.requires_grad = False

    logger.log("源模型导入成功")

    # learning_rate = 1e-5
    # optimizer_net = AdamW(source_net_raw.parameters(), lr=learning_rate , weight_decay=1e-5)
    # optimizer_classifier = AdamW(source_classifier_raw.parameters(), lr=learning_rate, weight_decay=1e-5)

    learning_rate = 1e-3  # 设置学习率
    # # 使用 SGD 优化器
    optimizer_net = SGD(source_net_raw.parameters(), lr=learning_rate)
    optimizer_classifier = SGD(source_classifier_raw.parameters(), lr=learning_rate)

    model_diffusion, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model_diffusion.to('cuda')
    if args.use_fp16:
        model_diffusion.convert_to_fp16()
    model_diffusion.eval()

    # 使用BackboneWrapper对每个source_net进行包装
    source_net = BackboneWrapper(source_net_raw)
    source_net_frozen = BackboneWrapper(source_net_frozen_raw)
    source_net_frozen2 = BackboneWrapper(source_net_frozen2_raw)

    # DataParallel封装
    model_diffusion = th.nn.DataParallel(model_diffusion)
    source_net = th.nn.DataParallel(source_net)
    source_classifier = th.nn.DataParallel(source_classifier_raw)
    source_net_frozen = th.nn.DataParallel(source_net_frozen)
    source_classifier_frozen = th.nn.DataParallel(source_classifier_frozen_raw)
    source_net_frozen2 = th.nn.DataParallel(source_net_frozen2)
    source_classifier_frozen2 = th.nn.DataParallel(source_classifier_frozen2_raw)

    # weights = [0.1, 0.3, 0.4, 1]
    # lpips_loss_fn = LPIPSLoss(source_net, weights=weights).to('cuda')

    # 使用官方LPIPS度量
    lpips_loss_fn = lpips.LPIPS(net='vgg').to('cuda')  # 可以根据需要改成'alex','squeeze','vgg'

    original_inputs_list1 = []
    original_inputs_list2 = []
    updated_inputs_list = []
    labels_list = []

    logger.log("开始初始化 Memory Bank...")

    mem_fea = None
    mem_cls = None

    # 注意：不要提前将images2.to('cuda')，让DataParallel分发
    with tqdm(total=total_samples, desc="Initializing memory bank", unit="sample") as pbar:
        for idx, batch_data in enumerate(data_loader):
            (images1, images2), labels, indices = batch_data
            # 不要提前 x2.to('cuda')
            x2 = images2
            indices = indices
            batch_size = x2.size(0)
            with th.no_grad():
                # 直接source_net(x2)，让DataParallel分发
                features_x2_all = source_net(x2) # 返回特征列表
                features_x2_ = features_x2_all[-1]
                features_x2_ = features_x2_ / th.norm(features_x2_, p=2, dim=1, keepdim=True)
                logits_x2 = source_classifier(features_x2_)
                class_probs_x2 = F.softmax(logits_x2, dim=1)

            if mem_fea is None:
                feature_dim = features_x2_.size(1)
                num_classes = class_probs_x2.size(1)
                mem_fea = th.zeros(total_samples, feature_dim).cuda()
                mem_cls = th.zeros(total_samples, num_classes).cuda()

            # 将indices放到CPU上分配，然后复制相应的数据
            indices = indices.to(mem_fea.device)
            mem_fea[indices] = features_x2_.detach()
            mem_cls[indices] = class_probs_x2.detach()
            pbar.update(batch_size)

    # 重新加载数据
    data_loader = load_paired_data(
        args.data_dir1,
        args.data_dir2,
        args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        max_samples=args.max_samples
    )

    logger.log("开始处理数据...")
    optimizer_net.zero_grad()
    optimizer_classifier.zero_grad()

    with tqdm(total=total_samples, desc="Processing data", unit="sample") as pbar:
        for idx, batch_data in enumerate(data_loader):
            (images1, images2), labels, batch_indices = batch_data
            # 不提前.to('cuda')，直接使用source_net分发
            x1 = images1
            x2 = images2
            x_orig = x1.clone().detach()
            x = x2.clone().detach()
            x.requires_grad = True

            labels = labels
            batch_indices = batch_indices

            batch_size = x.size(0)

            original_inputs_list1.append(x_orig.clone().detach())
            original_inputs_list2.append(x2.clone().detach())
            labels_list.append(labels.clone().detach())

            source_net.train()
            source_classifier.train()

            features_current_all = source_net(x) # list of features
            features_current = features_current_all[-1]
            logits_current = source_classifier(features_current)
            softmax_out = F.softmax(logits_current, dim=1)

            temperature = 0.8
            sharpened_softmax_out = temperature_sharpening(softmax_out, temperature)

            loss_emin = th.mean(Entropy(sharpened_softmax_out))
            msoftmax_out = sharpened_softmax_out.mean(dim=0)
            loss_gemin = th.sum(-msoftmax_out * th.log(msoftmax_out + 1e-10))
            loss_im = loss_emin - loss_gemin

            with th.no_grad():
                features = features_current / th.norm(features_current, p=2, dim=1, keepdim=True)
                outputs_target = sharpened_softmax_out

            # mem_fea, mem_cls 在 GPU 上
            # 将 batch_indices 放到相同的device上
            batch_indices = batch_indices.to(mem_fea.device)

            dis = -th.mm(features.detach(), mem_fea.t())
            for i in range(batch_size):
                dis[i, batch_indices[i]] = dis.max()

            k = min(5, mem_fea.size(0))
            _, p1 = th.topk(dis, k=k, dim=1, largest=False)
            w = th.zeros(batch_size, mem_fea.size(0)).to(mem_fea.device)
            for i in range(batch_size):
                w[i, p1[i]] = 1 / k

            mem_cls_weighted = th.mm(w, mem_cls)
            weight_, pred = th.max(mem_cls_weighted, dim=1)

            beta_distribution = th.distributions.Beta(0.5, 0.5)
            lam = beta_distribution.sample((batch_size,)).to(mem_fea.device)
            lam = lam.view(-1, 1)

            index = th.randperm(batch_size).to(mem_fea.device)
            features_interpolated = lam * features + (1 - lam) * features[index]
            pseudo_labels_interpolated = lam * mem_cls_weighted + (1 - lam) * mem_cls_weighted[index]

            logits_interpolated = source_classifier(features_interpolated)
            predicted_probs_interpolated = F.softmax(logits_interpolated, dim=1)
            consistency_loss = -th.mean(th.sum(pseudo_labels_interpolated * th.log(predicted_probs_interpolated + 1e-10), dim=1))

            # 使用LPIPS官方函数计算感知损失
            x1_lpips = denormalize_and_prepare_for_lpips(x1)
            x_lpips = denormalize_and_prepare_for_lpips(x)
            lpips_loss = lpips_loss_fn(x1_lpips, x_lpips).mean()

            a = 1
            b = 0.3
            c = 100

            model_loss = a * consistency_loss + b * loss_im
            sample_loss = a * consistency_loss + c * lpips_loss

            optimizer_net.zero_grad()
            optimizer_classifier.zero_grad()

            if x.grad is not None:
                x.grad.zero_()

            sample_loss.backward(retain_graph=True)

            with th.no_grad():
                x_updated = x - x.grad * 6
            x.grad.zero_()

            updated_inputs_list.append(x_updated.clone().detach())

            model_loss.backward()
            optimizer_net.step()
            optimizer_classifier.step()

            # 更新 Memory Bank
            mem_fea[batch_indices] = 0.8 * mem_fea[batch_indices] + 0.2 * features.detach()
            mem_fea[batch_indices] = mem_fea[batch_indices] / th.norm(mem_fea[batch_indices], p=2, dim=1, keepdim=True)
            mem_cls[batch_indices] = 0.8 * mem_cls[batch_indices] + 0.2 * outputs_target.detach()

            pbar.update(batch_size)

    # 不需要dist.barrier()，因为没有分布式训练

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
