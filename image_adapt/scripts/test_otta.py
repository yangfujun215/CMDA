import argparse
import os
import random
from torch.optim import AdamW  # 引入 AdamW 优化器

from tqdm import tqdm
from PIL import Image
import json
import numpy as np
import torch as th
import torch.distributed as dist
from torch.nn import DataParallel
import sys
from torch.optim import SGD

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

# 新增导入
from timm import create_model
import copy

import warnings
warnings.filterwarnings("ignore", module='mmcv')
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

logger.log("纯model、用来计算天气正确率。")

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

    # 设置默认的 corruptions 列表
    if not args.corruptions:
        args.corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
                            'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'contrast',
                            'elastic_transform', 'pixelate', 'jpeg_compression']

    # 构建模型
    model = create_model(
        'vit_base_patch16_224',  # 模型类型
        pretrained=False,         # 加载预训练权重
        num_classes=1000         # 输出类别数
    )

    # 加载本地预训练权重
    pretrained_weights_path = '/home/yfj/DDA-main-4090/ckpt/vit.bin'  # 替换为本地权重路径
    state_dict = th.load(pretrained_weights_path, map_location='cuda')
    model.load_state_dict(state_dict)

    model.to('cuda')

    # 导入 deepcopy
    import copy

    # 创建冻结的模型副本
    model_frozen = copy.deepcopy(model)
    for param in model_frozen.parameters():
        param.requires_grad = False
    model_frozen.to('cuda')

    logger.log("模型导入成功")

    learning_rate = 1e-3  # 设置学习率

    # 使用 AdamW 优化器
    # optimizer = AdamW(model.parameters(), lr=learning_rate , weight_decay=1e-5)
    optimizer = SGD(model.parameters(), lr=learning_rate)

    # 初始化模型和扩散过程
    model_diffusion, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model_diffusion.to(dist_util.dev())
    if args.use_fp16:
        model_diffusion.convert_to_fp16()
    model_diffusion = th.nn.DataParallel(model_diffusion)
    model_diffusion.eval()

    # 初始化用于每个 corruption 的准确率存储
    per_corruption_accuracies = {}

    device = next(model.parameters()).device  # 获取模型所在的设备

    # 遍历所有的 corruptions
    for corruption in args.corruptions:
        # 设置数据路径
        args.data_dir1 = f"/home/yfj/imagenet-c/{corruption}/{args.severity}"
        args.data_dir2 = f"/home/yfj/ImageNet-C-Syn/{corruption}/{args.severity}"
        logger.log(f"当前域： (corruption): {corruption}, severity: {args.severity}")

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

        logger.log("开始初始化 Memory Bank...")

        # 初始化 Memory Bank
        mem_fea = None
        mem_cls = None
        feature_dim = None
        num_classes = None

        # 使用冻结的模型初始化 Memory Bank
        with tqdm(total=total_samples, desc=f"Initializing memory bank for {corruption}", unit="sample") as pbar:
            for idx, batch_data in enumerate(data_loader):
                (images1, images2), labels, indices = batch_data
                x2 = images2.to(device)
                indices = indices.to(device)
                batch_size = x2.size(0)
                with th.no_grad():
                    logits_x2 = model_frozen(x2)
                    features_x2 = model_frozen.forward_features(x2)
                    # 只取 [CLS] token 的特征
                    features_x2 = features_x2[:, 0, :]  # 形状变为 [batch_size, feature_dim]
                    features_x2 = features_x2 / th.norm(features_x2, p=2, dim=1, keepdim=True)
                    class_probs_x2 = F.softmax(logits_x2, dim=1)
                if mem_fea is None:
                    feature_dim = features_x2.size(1)
                    num_classes = class_probs_x2.size(1)
                    mem_fea = th.zeros(total_samples, feature_dim).to(device)
                    mem_cls = th.zeros(total_samples, num_classes).to(device)
                mem_fea[indices] = features_x2.detach()
                mem_cls[indices] = class_probs_x2.detach()
                pbar.update(batch_size)

        logger.log(f"Memory Bank 初始化完成 for corruption {corruption}")

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
        optimizer.zero_grad()
        step_counter = 0

        # 添加变量，跟踪已更新的样本数量
        updated_sample_count = 0  # 已用于更新模型参数的样本数量
        max_update_samples = args.max_update_samples  # 从参数中获取最大更新样本数

        # 初始化变量以存储总的正确预测数量和样本数量
        total_correct_trainable = 0
        total_correct_frozen = 0
        total_correct_frozen2 = 0
        total_correct_ensemble_sum = 0
        total_correct_ensemble_sum2 = 0
        total_correct_ensemble_sum3 = 0
        total_correct_ensemble_sum4 = 0
        total_correct_ensemble_entropy_sum = 0
        total_correct_ensemble_confidence_sum = 0
        total_correct_ensemble_confidence_sum2 = 0
        total_samples_processed = 0

        with tqdm(total=total_samples, desc=f"Processing data for {corruption}", unit="sample") as pbar:
            for idx, batch_data in enumerate(data_loader):
                (images1, images2), labels, batch_indices = batch_data
                x1 = images1.to(device)  # 目标域数据，用于后续计算正确率
                x2 = images2.to(device)  # 虚拟域数据，用于进行中间域不确定性得分的比较，后续计算正确率
                x_orig = x1.clone().detach()  # 目标域数据
                x = x2.clone().detach()  # 利用虚拟域数据进行后续模型更新，然后更新模型到中间域
                x.requires_grad = True  # 对 x2 进行更新得到中间域数据
                labels = labels.to(device).detach()
                batch_indices = batch_indices.to(device)  # 将索引移动到 GPU

                batch_size = x.size(0)

                # 模型设置为训练模式
                model.train()

                # 直接使用模型进行前向传播
                logits_current = model(x)
                softmax_out = th.nn.Softmax(dim=1)(logits_current)

                # 提取特征
                features_current = model.forward_features(x)
                features_current = features_current[:, 0, :]  # 只取 [CLS] token 的特征
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
                w = th.zeros(batch_size, mem_fea.size(0)).to(device)
                for i in range(batch_size):
                    w[i, p1[i]] = 1 / k

                # 计算加权分类输出，获取伪标签
                mem_cls_weighted = th.mm(w, mem_cls)
                weight_, pred = th.max(mem_cls_weighted, dim=1)

                # 插值一致性损失
                # 从 Beta 分布中采样 λ
                beta_distribution = th.distributions.Beta(0.5, 0.5)
                lam = beta_distribution.sample((batch_size,)).to(device)
                lam = lam.view(-1, 1)

                # 对特征和伪标签进行插值
                index = th.randperm(batch_size).to(device)
                features_interpolated = lam * features + (1 - lam) * features[index]
                pseudo_labels_interpolated = lam * mem_cls_weighted + (1 - lam) * mem_cls_weighted[index]

                # 通过模型的头部获得插值特征的预测
                logits_interpolated = model.head(features_interpolated)

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

                # 计算总损失
                a = 1
                b = 0.3
                c = 0.2
                total_loss = a * consistency_loss + b * loss_im + c * norm

                # 将损失除以累积步数
                loss = total_loss / accumulation_steps

                # 反向传播
                loss.backward()

                # 更新样本 x
                with th.no_grad():
                    x_updated = x - x.grad * args.scale  # 更新后的样本

                x.grad.zero_()

                # 模型参数更新
                if (step_counter + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    step_counter = 0  # 重置计数器
                else:
                    step_counter += 1

                # 更新已更新的样本数量
                updated_sample_count += batch_size

                # **评估当前批次**
                model.eval()
                model_frozen.eval()

                with th.no_grad():
                    # 更新后的中间域模型，处理中间域数据
                    logits_trainable = model(x_updated)
                    preds_trainable = logits_trainable.argmax(dim=1)

                    # 原模型处理虚拟域数据
                    logits_frozen = model_frozen(x2)
                    preds_frozen = logits_frozen.argmax(dim=1)

                    # 原模型处理目标域数据
                    logits_frozen2 = model_frozen(x1)
                    preds_frozen2 = logits_frozen2.argmax(dim=1)

                    # 获取所有模式的集成结果
                    y_ensemble_sum, y_ensemble_sum2, y_ensemble_sum3, y_ensemble_sum4, y_ensemble_entropy_sum, y_ensemble_confidence_sum, y_ensemble_confidence_sum2 = ensemble_logits(
                        logits_trainable, logits_frozen, logits_frozen2)
                    final_pred_sum = y_ensemble_sum.argmax(dim=1)
                    final_pred_sum2 = y_ensemble_sum2.argmax(dim=1)
                    final_pred_sum3 = y_ensemble_sum3.argmax(dim=1)
                    final_pred_sum4 = y_ensemble_sum4.argmax(dim=1)
                    final_pred_entropy_sum = y_ensemble_entropy_sum.argmax(dim=1)
                    final_pred_confidence_sum = y_ensemble_confidence_sum.argmax(dim=1)
                    final_pred_confidence_sum2 = y_ensemble_confidence_sum2.argmax(dim=1)

                    # 计算当前批次的正确预测数量
                    total_correct_trainable += (preds_trainable == labels).sum().item()
                    total_correct_frozen += (preds_frozen == labels).sum().item()
                    total_correct_frozen2 += (preds_frozen2 == labels).sum().item()
                    total_correct_ensemble_sum += (final_pred_sum == labels).sum().item()
                    total_correct_ensemble_sum2 += (final_pred_sum2 == labels).sum().item()
                    total_correct_ensemble_sum3 += (final_pred_sum3 == labels).sum().item()
                    total_correct_ensemble_sum4 += (final_pred_sum4 == labels).sum().item()
                    total_correct_ensemble_entropy_sum += (final_pred_entropy_sum == labels).sum().item()
                    total_correct_ensemble_confidence_sum += (final_pred_confidence_sum == labels).sum().item()
                    total_correct_ensemble_confidence_sum2 += (final_pred_confidence_sum2 == labels).sum().item()

                    total_samples_processed += labels.size(0)

                # 更新 Memory Bank（动量更新）
                mem_fea[batch_indices] = 0.4 * mem_fea[batch_indices] + 0.6 * features.detach()
                mem_fea[batch_indices] = mem_fea[batch_indices] / th.norm(mem_fea[batch_indices], p=2, dim=1,
                                                                          keepdim=True)
                mem_cls[batch_indices] = 0.4 * mem_cls[batch_indices] + 0.6 * outputs_target.detach()

                # 更新进度条，增加处理过的样本数
                pbar.update(batch_size)

        # 计算最终的平均准确率
        accuracy_trainable = total_correct_trainable / total_samples_processed
        accuracy_frozen = total_correct_frozen / total_samples_processed
        accuracy_frozen2 = total_correct_frozen2 / total_samples_processed
        accuracy_ensemble_sum = total_correct_ensemble_sum / total_samples_processed
        accuracy_ensemble_sum2 = total_correct_ensemble_sum2 / total_samples_processed
        accuracy_ensemble_sum3 = total_correct_ensemble_sum3 / total_samples_processed
        accuracy_ensemble_sum4 = total_correct_ensemble_sum4 / total_samples_processed
        accuracy_ensemble_entropy_sum = total_correct_ensemble_entropy_sum / total_samples_processed
        accuracy_ensemble_confidence_sum = total_correct_ensemble_confidence_sum / total_samples_processed
        accuracy_ensemble_confidence_sum2 = total_correct_ensemble_confidence_sum2 / total_samples_processed

        # 输出结果
        print(f"\nCorruption: {corruption}")
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

        # 存储当前 corruption 的准确率
        per_corruption_accuracies[corruption] = {
            'accuracy_trainable': accuracy_trainable,
            'accuracy_frozen': accuracy_frozen,
            'accuracy_frozen2': accuracy_frozen2,
            'accuracy_ensemble_sum': accuracy_ensemble_sum,
            'accuracy_ensemble_sum2': accuracy_ensemble_sum2,
            'accuracy_ensemble_sum3': accuracy_ensemble_sum3,
            'accuracy_ensemble_sum4': accuracy_ensemble_sum4,
            'accuracy_ensemble_entropy_sum': accuracy_ensemble_entropy_sum,
            'accuracy_ensemble_confidence_sum': accuracy_ensemble_confidence_sum,
            'accuracy_ensemble_confidence_sum2': accuracy_ensemble_confidence_sum2,
        }

    # 输出每个 corruption 的准确率
    print("\n各个 corruption 的准确率:")
    for corruption, accuracies in per_corruption_accuracies.items():
        print(f"\nCorruption: {corruption}")
        print(f"Trainable Model (Updated x2) Accuracy 中间域在中间域模型正确率: {accuracies['accuracy_trainable'] * 100:.8f}%")
        print(f"Frozen Model (Original x2) Accuracy 虚拟域在原模型正确率: {accuracies['accuracy_frozen'] * 100:.8f}%")
        print(f"Frozen Model (Original x1) Accuracy 目标域在原模型正确率: {accuracies['accuracy_frozen2'] * 100:.8f}%")
        print(f"中间域+虚拟域+目标域正确率: {accuracies['accuracy_ensemble_sum'] * 100:.8f}%")
        print(f"中间域+目标域 正确率: {accuracies['accuracy_ensemble_sum2'] * 100:.8f}%")
        print(f"中间域+虚拟域 正确率: {accuracies['accuracy_ensemble_sum3'] * 100:.8f}%")
        print(f"DDA 正确率: {accuracies['accuracy_ensemble_sum4'] * 100:.8f}%")
        print(f"Combined Model Accuracy (entropy_sum) 正确率: {accuracies['accuracy_ensemble_entropy_sum'] * 100:.8f}%")
        print(f"Combined Model Accuracy (confidence_sum) 正确率: {accuracies['accuracy_ensemble_confidence_sum'] * 100:.8f}%")
        print(f"Combined Model Accuracy (confidence_sum2) 正确率: {accuracies['accuracy_ensemble_confidence_sum2'] * 100:.8f}%")

    dist.barrier()

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
        data_dir1="",  # 原图像路径
        data_dir2="",  # 去噪图像路径
        max_samples=500,  # 要加载的最大样本数
        seed=42,
        effective_batch_size=64,  # 想要模拟的更大批次大小
        max_update_samples=50000,  # 最大模型更新的样本数量
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    parser.add_argument('--corruptions', type=str, nargs='+', default=[], help='List of corruptions for CoTTA')
    return parser

if __name__ == "__main__":
    main()
