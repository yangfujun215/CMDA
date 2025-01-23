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
    mean=[123.675 / 255.0, 116.28 / 255.0, 103.53 / 255.0],
    std=[58.395 / 255.0, 57.12 / 255.0, 57.375 / 255.0],
    to_rgb=True
)

# 数据加载函数
def load_denoised_data(data_dir, batch_size, image_size, class_cond=False, max_samples=None):
    # 定义图像变换
    transform = transforms.Compose([
        transforms.Resize(256),  # 保持宽高比，将宽度调整到 256
        transforms.CenterCrop(224),
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=img_norm_cfg['mean'],
            std=img_norm_cfg['std']
        )
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    if max_samples is not None:
        dataset.samples = dataset.samples[:max_samples]

    data_loader = th.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    for images, labels in data_loader:
        model_kwargs = {}
        model_kwargs["ref_img"] = images

        yield model_kwargs, labels

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

def evaluate_model(source_net_trainable, source_classifier_trainable,
                   source_net_frozen, source_classifier_frozen,
                   original_inputs_list, updated_inputs_list,
                   labels_list, args):
    logger.log("Evaluating models on original and updated data...")

    # 将模型设置为评估模式
    source_net_trainable.eval()
    source_classifier_trainable.eval()
    source_net_frozen.eval()
    source_classifier_frozen.eval()

    # 获取模型所在的设备
    device = next(source_net_trainable.parameters()).device

    all_preds_trainable = []
    all_preds_frozen = []
    all_preds_ensemble = []
    all_labels = []

    with th.no_grad():
        for original_x, updated_x, labels in zip(original_inputs_list, updated_inputs_list, labels_list):
            original_x = original_x.to(device)
            updated_x = updated_x.to(device)
            labels = labels.to(device)

            features_trainable = source_net_trainable.backbone(updated_x)[-1]
            features_trainable = source_net_trainable.neck(features_trainable)
            logits_trainable = source_classifier_trainable(features_trainable)
            preds_trainable = logits_trainable.argmax(dim=1)

            features_frozen = source_net_frozen.backbone(original_x)[-1]
            features_frozen = source_net_frozen.neck(features_frozen)
            logits_frozen = source_classifier_frozen(features_frozen)
            preds_frozen = logits_frozen.argmax(dim=1)

            # 对两个模型的 logits 进行加和集成
            y_ensemble = logits_trainable + logits_frozen
            final_pred = y_ensemble.argmax(dim=1)

            # 将当前批次的预测和标签添加到列表中
            all_preds_trainable.append(preds_trainable.cpu())
            all_preds_frozen.append(preds_frozen.cpu())
            all_preds_ensemble.append(final_pred.cpu())
            all_labels.append(labels.cpu())

    # # 将所有预测和标签拼接起来
    all_preds_trainable = th.cat(all_preds_trainable)
    all_preds_frozen = th.cat(all_preds_frozen)
    all_preds_ensemble = th.cat(all_preds_ensemble)
    all_labels = th.cat(all_labels)

    # 计算准确率
    accuracy_trainable = (all_preds_trainable == all_labels).float().mean().item()
    accuracy_frozen = (all_preds_frozen == all_labels).float().mean().item()
    accuracy_ensemble = (all_preds_ensemble == all_labels).float().mean().item()

    # 输出结果
    print(f"Trainable Model Accuracy: {accuracy_trainable * 100:.8f}%")
    print(f"Frozen Model Accuracy: {accuracy_frozen * 100:.8f}%")
    print(f"Combined Model Accuracy (Logits Sum): {accuracy_ensemble * 100:.8f}%")

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
    logger.configure(dir=args.save_dir)

    # 输出当前破坏类型
    logger.log(f"当前域： (corruption): {args.corruption}")

    logger.log("加载数据...")
    data = load_denoised_data(
        args.denoised_data_dir,
        args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        max_samples=args.max_samples  # 添加最大样本数参数
    )

    # 获取数据集的总样本数
    dataset = datasets.ImageFolder(root=args.denoised_data_dir)
    total_samples = len(dataset)  # 获取实际的样本总数
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

    # # 使用 DataParallel 将模型放到多个 GPU 上
    # source_net = DataParallel(source_net)
    # source_classifier = DataParallel(source_classifier)

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
        model2.backbone.load_state_dict(backbone_state_dict, strict=False)
        model2.neck.load_state_dict(neck_state_dict, strict=False)

    # 处理 head 的权重名称，将其映射到 source_classifier 的参数名称
    classifier_state_dict = {}
    for k, v in head_state_dict.items():
        if k == 'fc.weight':
            classifier_state_dict['fc.weight'] = v
        elif k == 'fc.bias':
            classifier_state_dict['fc.bias'] = v

    # 加载 head（分类器）的权重到两个分类器中
    for classifier in [source_classifier, source_classifier_frozen, source_classifier_frozen2 ]:
        classifier.load_state_dict(classifier_state_dict, strict=False)

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
    logger.log("源分类器导入成功")

    learning_rate_source_net = 1e-5  # 设置学习率

    # 使用 SGD 优化器
    optimizer_net = SGD(source_net.parameters(), lr=learning_rate_source_net)
    optimizer_classifier = SGD(source_classifier.parameters(), lr=learning_rate_source_net)

    # 初始化模型和扩散过程
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    #model = DataParallel(model)  # 将扩散模型也放到多个 GPU 上
    model.eval()

    # 保存用于评估的输入和标签
    original_inputs_list = []
    labels_list = []
    updated_inputs_list = []

    source_net.train()
    source_classifier.train()

    # 初始化 Memory Bank
    mem_fea = None
    mem_cls = None
    mem_initialized = False  # 标志 Memory Bank 是否已初始化

    logger.log("开始处理数据...")
    with tqdm(total=total_samples, desc="Processing data", unit="sample") as pbar:
        for idx, (model_kwargs, labels) in enumerate(data):
            x = model_kwargs["ref_img"].to('cuda')
            x_orig = x.clone().detach()  # 创建原始输入的副本
            x.requires_grad = True  # 使 x 可训练，以便更新数据
            labels = labels.to('cuda').detach()

            batch_size = x.size(0)

            # 保存原始输入到列表
            original_inputs_list.append(x_orig.clone().detach().cpu())
            labels_list.append(labels.clone().detach().cpu())

            # 对于数据，计算特征和输出
            features_current = source_net.backbone(x)[-1]
            features = source_net.neck(features_current)
            logits_current = source_classifier(features)
            softmax_out = th.nn.Softmax(dim=1)(logits_current)

            # 应用温度锐化
            temperature = 0.5  # 可以根据需求调整温度参数
            sharpened_softmax_out = temperature_sharpening(softmax_out, temperature)

            # 计算信息熵损失
            loss_emin = th.mean(Entropy(sharpened_softmax_out))
            msoftmax_out = sharpened_softmax_out.mean(dim=0)
            loss_gemin = th.sum(-msoftmax_out * th.log(msoftmax_out + 1e-10))
            loss_emin -= loss_gemin
            loss_im = loss_emin

            # 初始化 Memory Bank
            if not mem_initialized:
                total_samples_mem = total_samples
                feature_dim = features.size(1)
                num_classes = sharpened_softmax_out.size(1)
                mem_fea = th.rand(total_samples_mem, feature_dim).cuda()
                mem_fea = mem_fea / th.norm(mem_fea, p=2, dim=1, keepdim=True)
                mem_cls = th.ones(total_samples_mem, num_classes).cuda() / num_classes
                mem_initialized = True

            # 获取当前批次在数据集中的全局索引
            start_idx = idx * args.batch_size
            end_idx = start_idx + batch_size
            if end_idx > total_samples:
                end_idx = total_samples
            batch_indices = list(range(start_idx, end_idx))

            with th.no_grad():
                features_target = features / th.norm(features, p=2, dim=1, keepdim=True)
                outputs_target = sharpened_softmax_out

            # 更新 Memory Bank（动量更新）
            mem_fea[batch_indices] = 0.1 * mem_fea[batch_indices] + 0.9 * features_target.detach()
            mem_fea[batch_indices] = mem_fea[batch_indices] / th.norm(mem_fea[batch_indices], p=2, dim=1, keepdim=True)
            mem_cls[batch_indices] = 0.1 * mem_cls[batch_indices] + 0.9 * outputs_target.detach()

            # 计算与 Memory Bank 的距离
            dis = -th.mm(features.detach(), mem_fea.t())

            # 排除自身（将自身距离设为最大值）
            for i in range(batch_size):
                dis[i, batch_indices[i]] = dis.max()  # 将自身距离设为最大值

            # 获取最近的 k 个邻居
            k = min(5, mem_fea.size(0))
            _, p1 = th.topk(dis, k=k, dim=1, largest=False)
            w = th.zeros(batch_size, mem_fea.size(0)).cuda()
            for i in range(batch_size):
                w[i, p1[i]] = 1 / k

            # 计算加权分类输出，获取伪标签
            mem_cls_weighted = th.mm(w, mem_cls)
            weight_, pred = th.max(mem_cls_weighted, dim=1)

            # # 计算分类损失
            # loss_ = th.nn.CrossEntropyLoss(reduction='none')(logits_current, pred)
            # classifier_loss = th.sum(weight_ * loss_) / (th.sum(weight_).item() + 1e-8)

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

            # 计算一致性损失
            consistency_loss = th.nn.KLDivLoss(reduction='batchmean')(
                F.log_softmax(logits_interpolated, dim=1),
                pseudo_labels_interpolated
            )

            # 计算总损失
            # loss = 0.9*classifier_loss + 0.3*loss_im
            loss = 0.9*consistency_loss + 0.3 * loss_im

            # 计算 norm 损失（不添加噪声，直接使用 x）
            t = th.zeros(x.size(0), dtype=th.long, device=x.device)
            out_mean_variance = diffusion.p_mean_variance(
                model,
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

            # 将 norm 添加到总损失中
            total_loss = loss + 0.2*norm

            total_loss = loss

            # 计算更新前的不确定性得分，使用 torch.no_grad() 和 x.clone().detach()
            with th.no_grad():
                features_source = source_net_frozen.backbone(x.clone().detach())[-1]
                features_source = source_net_frozen.neck(features_source)
                logits_source = source_classifier_frozen(features_source)
                probabilities_source = th.nn.Softmax(dim=1)(logits_source)
                uncertainty_source = -th.sum(probabilities_source * th.log(probabilities_source + 1e-10), dim=1).mean()

            # 清零梯度
            optimizer_net.zero_grad()
            optimizer_classifier.zero_grad()
            if x.grad is not None:
                x.grad.zero_()

            # 反向传播
            total_loss.backward()

            # 更新样本 x
            with th.no_grad():
                x_updated = x - x.grad * args.scale  # 更新后的样本
            # 更新模型参数
            optimizer_net.step()
            optimizer_classifier.step()

            # 计算更新后的样本的不确定性得分，使用 torch.no_grad() 和 x_updated.clone().detach()
            with th.no_grad():
                features_current = source_net_frozen.backbone(x_updated.clone().detach())[-1]
                features_current = source_net_frozen.neck(features_current)
                logits_current = source_classifier_frozen(features_current)
                probabilities_current = th.nn.Softmax(dim=1)(logits_current)
                uncertainty_current = -th.sum(probabilities_current * th.log(probabilities_current + 1e-10),
                                              dim=1).mean()

            # 比较不确定性得分，选择保存的样本
            if uncertainty_source < uncertainty_current:
                used_sample = x_orig.clone().detach()
            else:
                used_sample = x_updated.clone().detach()


            # 保存更新后的输入
            updated_inputs_list.append(used_sample.clone().detach().cpu())

            # 更新进度条，增加处理过的样本数
            pbar.update(batch_size)

    dist.barrier()

    # 在训练结束后，保存更新后的样本用于验证
    # 这里可以根据需求保存 updated_inputs_list 中的图像

    evaluate_model(source_net, source_classifier,
                   source_net_frozen2, source_classifier_frozen2,
                   original_inputs_list, updated_inputs_list,  # 使用更新后的输入列表
                   labels_list, args)

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
        denoised_data_dir="/media/shared_space/wuyanzu/ImageNet-C-Syn/brightness/5",  # 去噪后数据集的路径
        max_samples=50000,  # 要加载的最大样本数
        seed=42,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()
