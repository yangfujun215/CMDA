import argparse
import os
import random

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
from torch.optim import SGD

from mmcv import Config
from mmcls.models import build_classifier
from mmcv.runner import wrap_fp16_model,load_checkpoint
from image_adapt.model_use import Classifier

from image_adapt.guided_diffusion import dist_util, logger
from image_adapt.guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from image_adapt.guided_diffusion.image_datasets import load_data
from torchvision import utils
import math
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torchvision import transforms
from torchvision import models
import torch.nn.functional as F
from image_adapt.resize_right import resize
from torch.optim import Adam

import warnings
warnings.filterwarnings("ignore", module='mmcv')
import os
CUDA_VISIBLE_DEVICES=2,3
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# 定义与第一段代码相同的图像归一化配置
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)  # 使用多GPU
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False

# added
def load_reference(data_dir, batch_size, image_size, class_cond=False, corruption="shot_noise", severity=5,):
    data = load_data(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=image_size,
        class_cond=class_cond,  # 用于控制模型条件类型
        load_labels=True,  # 用于控制是否加载标签
        deterministic=True,
        random_flip=False,
        corruption=corruption,
        severity=severity,
    )
    for large_batch, model_kwargs, filename in data:
        model_kwargs["ref_img"] = large_batch
        labels = model_kwargs.get("y")  # 加载标签
        if labels is None:
            raise ValueError(
                "Labels not found in model_kwargs. Please ensure that class_cond=True and labels are correctly loaded.")
        # 确保在模型前向传播时不传递 'y'
        if 'y' in model_kwargs:
            del model_kwargs['y']
        yield model_kwargs, filename, labels

def Entropy(input_):
    epsilon = 1e-5
    entropy = -input_ * th.log(input_ + epsilon)
    entropy = th.sum(entropy, dim=1)
    return entropy

def evaluate_model(source_net, source_classifier,
                   source_net_frozen, source_classifier_frozen,
                   updated_inputs_list,
                   labels_list, args):
    logger.log("Evaluating models on original and updated data...")

    # 将模型设置为评估模式
    source_net.eval()
    source_classifier.eval()
    source_net_frozen.eval()
    source_classifier_frozen.eval()

    # 获取模型所在的设备
    device = next(source_net.parameters()).device

    all_preds_trainable = []
    all_preds_frozen = []
    all_preds_ensemble = []
    all_labels = []

    with th.no_grad():
        for updated_x, labels in zip( updated_inputs_list, labels_list):
            # 使用更新后的模型进行前向传播
            # 更新后的模型
            # original_x = original_x.to(device)
            updated_x = updated_x.to(device)
            labels = labels.to(device)

            features_trainable = source_net.backbone(updated_x)[-1]
            features_trainable = source_net.neck(features_trainable)
            logits_trainable = source_classifier(features_trainable)
            preds_trainable = logits_trainable.argmax(dim=1)

            # 冻结的模型
            features_frozen = source_net_frozen.backbone(updated_x)[-1]
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

    # 将所有预测和标签拼接起来
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

    logger.log("creating model...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cuda")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()

    model = th.nn.DataParallel(model)
    model.eval()  # 冻住模型，不改变模型参数

    logger.log("creating resizers...")
    assert math.log(args.D, 2).is_integer()

    logger.log("loading data...")
    data = load_reference(
        args.base_samples,
        args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        corruption=args.corruption,
        severity=args.severity,
    )

    assert args.num_samples >= args.batch_size * dist_util.get_world_size(), "The number of the generated samples will be larger than the specified number."

    feature_dim = 2048
    num_classes = 1000

    # 初始化记忆库为空列表，后续动态扩展
    mem_fea_list = []
    mem_cls_list = []

    # 获取特征提取器、分类器及其权重
    # 加载配置
    cfg = Config.fromfile(args.classifier_config_path)
    source_net = build_classifier(cfg.model)
    source_net.to('cuda')
    source_classifier = Classifier(num_classes=1000).to('cuda')

    # 创建只用于推理的模型实例
    source_net_frozen = build_classifier(cfg.model)
    source_net_frozen.to('cuda')
    source_classifier_frozen = Classifier(num_classes=1000).to('cuda')
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
    for model2 in [source_net, source_net_frozen]:
        missing_keys, unexpected_keys = model2.backbone.load_state_dict(backbone_state_dict, strict=False)
        if missing_keys:
            print(f"加载 backbone 权重时缺少的参数：{missing_keys}")
        if unexpected_keys:
            print(f"加载 backbone 权重时多余的参数：{unexpected_keys}")

        missing_keys, unexpected_keys = model2.neck.load_state_dict(neck_state_dict, strict=False)
        if missing_keys:
            print(f"加载 neck 权重时缺少的参数：{missing_keys}")
        if unexpected_keys:
            print(f"加载 neck 权重时多余的参数：{unexpected_keys}")

    # 处理 head 的权重名称，将其映射到 source_classifier 的参数名称
    classifier_state_dict = {}
    for k, v in head_state_dict.items():
        if k == 'fc.weight':
            classifier_state_dict['fc.weight'] = v
        elif k == 'fc.bias':
            classifier_state_dict['fc.bias'] = v

    # 加载 head（分类器）的权重到两个分类器中
    for classifier in [source_classifier, source_classifier_frozen]:
        missing_keys, unexpected_keys = classifier.load_state_dict(classifier_state_dict, strict=False)
        if missing_keys:
            print(f"加载分类器权重时缺少的参数：{missing_keys}")
        if unexpected_keys:
            print(f"加载分类器权重时多余的参数：{unexpected_keys}")

    # 冻结不用于训练的模型的参数
    for param in source_net_frozen.parameters():
        param.requires_grad = False
    for param in source_classifier_frozen.parameters():
        param.requires_grad = False

    logger.log("源模型导入成功")
    logger.log("源分类器导入成功")

    learning_rate_source_net = 5e-5  # 根据需要设置学习率

    optimizer_net = SGD(source_net.parameters(), lr=learning_rate_source_net)
    optimizer_classifier = SGD(source_classifier.parameters(), lr=learning_rate_source_net)

    source_net.train()
    source_classifier.train()
    sample_idx = 0

    # 创建用于保存训练数据的列表
    updated_inputs_list = []
    labels_list = []

    # 定义归一化变换
    normalize = transforms.Normalize(
        mean=[123.675 / 255.0, 116.28 / 255.0, 103.53 / 255.0],
        std=[58.395 / 255.0, 57.12 / 255.0, 57.375 / 255.0]
    )

    logger.log("creating samples...")
    count = 0
    while count * args.batch_size * dist_util.get_world_size() < args.num_samples:
        try:
            model_kwargs, filename, labels = next(data)  # 获取数据
        except StopIteration:
            break
        model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}

        # 确保 model_kwargs 中不包含 'y'
        if 'y' in model_kwargs:
            del model_kwargs['y']

        x = model_kwargs["ref_img"].to('cuda').detach() # 确保 x 在 GPU 上
        labels = labels.to('cuda').detach()  # 确保标签在 GPU 上

        # 保存目标域数据到列表
        # original_inputs_list.append(x.clone().detach().cpu())
        # labels_list.append(labels.clone().detach().cpu())

        # x.requires_grad_(True)  # 确保 x 需要梯度
        shape = x.shape
        shape_u = (shape[0], 3, shape[2], shape[3])
        shape_d = (shape[0], 3, int(shape[2] / args.D), int(shape[3] / args.D))

        sample_generator = diffusion.p_sample_loop_progressive(
            model,
            shape,
            noise=model_kwargs["ref_img"],
            clip_denoised=args.clip_denoised,
            denoised_fn=None,
            cond_fn=None,
            model_kwargs=model_kwargs,
            progress=False,
            device=None,
            N=args.N,
            D=args.D,
            scale=args.scale,
        )

        # 启动生成器，获取第一个时间步的输出
        try:
            out, img, t = next(sample_generator)
        except StopIteration:
            continue

        source_net.train()
        source_classifier.train()

        for _ in range(args.N):
            current_sample = out["sample"].clone().detach().requires_grad_(True).to('cuda')
            source_xstart = model_kwargs["ref_img"].clone().detach().to('cuda')
            print("当前时间步：", t)

            # # 冻结模型参数
            # for param in source_net.parameters():
            #     param.requires_grad = False
            # for param in source_classifier.parameters():
            #     param.requires_grad = False

            # 样本更新（在每个时间步都执行）
            out_mean_variance = diffusion.p_mean_variance(
                model,
                current_sample,
                t,
                clip_denoised=args.clip_denoised,
                denoised_fn=None,
                model_kwargs=model_kwargs,
            )
            pred_xstart = out_mean_variance["pred_xstart"]

            difference = resize(resize(source_xstart, scale_factors=1.0 / args.D, out_shape=shape_d),
                                scale_factors=args.D, out_shape=shape_u) - \
                         resize(resize(pred_xstart, scale_factors=1.0 / args.D, out_shape=shape_d),
                                scale_factors=args.D, out_shape=shape_u)

            norm = th.linalg.norm(difference)

            loss_grad = th.autograd.grad(outputs=norm, inputs=current_sample, retain_graph=True)[0]

            with th.no_grad():
                current_sample -= loss_grad * args.scale

            try:
                logger.log("传回生成器")
                out, img, t = sample_generator.send(current_sample)
            except StopIteration:
                break

            # 保存最后一个时间步的 norm
            norm_last = norm.clone().detach()

        final_sample = current_sample.clone().detach().requires_grad_(True).to('cuda')
        current_sample_res = (final_sample + 1) / 2
        source_xstart_res = (source_xstart + 1) / 2

        current_sample_res = F.interpolate(
            current_sample_res, size=(224, 224), mode='bilinear', align_corners=False
        )
        source_xstart_res = F.interpolate(
            source_xstart_res, size=(224, 224), mode='bilinear', align_corners=False
        )

        current_sample_res = normalize(current_sample_res)
        source_xstart_res = normalize(source_xstart_res)


        # with th.no_grad():
        features_current = source_net.backbone(current_sample_res)[-1]  # 获取最后一层的特征
        features_current = source_net.neck(features_current)  # 输出形状为 [batch_size, 2048]
        logits_current = source_classifier(features_current)
        probabilities_current = th.nn.Softmax(dim=1)(logits_current)
        #
        # 计算 current_sample 的不确定性得分（使用熵）
        uncertainty_current = -th.sum(probabilities_current * th.log(probabilities_current + 1e-10), dim=1)
        uncertainty_current = uncertainty_current.mean()  # 标量

        features_source = source_net.backbone(source_xstart_res)[-1]
        features_source = source_net.neck(features_source)
        logits_source = source_classifier(features_source)
        probabilities_source = th.nn.Softmax(dim=1)(logits_source)

        # 计算 source_xstart 的不确定性得分（使用熵）
        uncertainty_source = -th.sum(probabilities_source * th.log(probabilities_source + 1e-10), dim=1)
        uncertainty_source = uncertainty_source.mean()  # 标量

        # 比较不确定性得分，选择使用的样本
        if uncertainty_source < uncertainty_current:
            used_sample = source_xstart_res  # 注意，这里使用的是经过 resize 的样本
            features = features_source
            logits_target = logits_source
            softmax_out = probabilities_source
            need_update = False  # 不需要更新模型
        else:
            used_sample = current_sample_res
            features = features_current
            logits_target = logits_current
            softmax_out = probabilities_current
            print("使用 current_sample 更新")
            need_update = True  # 需要更新模型
        logger.log("获取成功")



        # # 只有当选择的是生成数据时，才进行后续的伪标签生成和模型更新
        # if need_update and (t == 0).all():
        # if (t == 0).all():
        if need_update:
            logger.log("更新模型参数")
            sample_idx += args.batch_size

            loss_emin = th.mean(Entropy(softmax_out))
            # 因为只有一个样本，所以 msoftmax_out 直接等于 softmax_out
            msoftmax_out = softmax_out.mean(dim=0)
            # 计算 loss_gemin
            loss_gemin = th.sum(-msoftmax_out * th.log(msoftmax_out + 1e-10))
            # 更新 loss_emin
            loss_emin -= loss_gemin
            # 设置 loss_im
            loss_im = loss_emin

            # 打伪标签、计算 loss_ce

            # 动态更新记忆库大小和内容
            # 将新特征和类别概率添加到记忆库列表中
            with th.no_grad():
                # 计算特征和输出
                features_target = features / th.norm(features, p=2, dim=1, keepdim=True)
                outputs_target = softmax_out

            # 确保 features_target 和 outputs_target 的第二维度分别为 feature_dim 和 num_classes
            assert features_target.shape[
                       1] == feature_dim, f"Feature dimension mismatch: expected {feature_dim}, got {features_target.shape[1]}"
            assert outputs_target.shape[
                       1] == num_classes, f"Number of classes mismatch: expected {num_classes}, got {outputs_target.shape[1]}"

            mem_fea_list.append(features_target.clone().detach())
            mem_cls_list.append(outputs_target.clone().detach())

            # 将列表转换为张量
            mem_fea = th.cat(mem_fea_list, dim=0)
            mem_cls = th.cat(mem_cls_list, dim=0)
            print("mem_fea形状：", mem_fea.shape)
            print("mem_cls形状：", mem_cls.shape)

            # 计算距离
            dis = -th.mm(features.detach(), mem_fea.t())

            # 获取前 5 个最近的记忆特征索引
            k = min(5, mem_fea.size(0))
            _, p1 = th.topk(dis, k=k, dim=1, largest=False)
            w = th.zeros(features.size(0), mem_fea.size(0)).cuda()
            w.scatter_(1, p1, 1 / k)

            # 计算加权最大预测
            mem_cls_weighted = th.mm(w, mem_cls)
            weight_, pred = th.max(mem_cls_weighted, dim=1)

            # 计算每个样本的交叉熵损失
            loss_ = th.nn.CrossEntropyLoss(reduction='none')(logits_target, pred)
            # 处理权重，确保它是标量
            classifier_loss = weight_ * loss_
            classifier_loss = th.sum(classifier_loss) / (th.sum(weight_).item() + 1e-8)  # 加一个小常数避免除零

            # loss = classifier_loss + 0.1 * norm + loss_im
            loss = classifier_loss  + loss_im
            print(f"Classifier Loss: {classifier_loss.item()}")  # 输出 classifier_loss 的值
            print(f"Image Loss: {loss_im.item()}")  # 输出 loss_im 的值
            print(f"Total Loss: {loss.item()}")  # 输出最终的总 loss 值

            # # 解冻模型参数
            # for param in source_net.parameters():
            #     param.requires_grad = True
            # for param in source_classifier.parameters():
            #     param.requires_grad = True

            optimizer_net.zero_grad()
            optimizer_classifier.zero_grad()

            loss.backward()
            # 更新模型参数
            optimizer_net.step()
            optimizer_classifier.step()

            # print(f"Iteration {sample_idx}, Loss: {loss.item()}")

            with th.no_grad():
                # 计算特征和输出
                features_target = features / th.norm(features, p=2, dim=1, keepdim=True)
                outputs_target = softmax_out

            # 使用批量操作更新记忆库
            mem_indices = sample_idx - args.batch_size + th.arange(args.batch_size)
            mem_fea[mem_indices] = 0.1 * mem_fea[mem_indices] + 0.9 * features_target.clone().detach()
            mem_cls[mem_indices] = 0.1 * mem_cls[mem_indices] + 0.9 * outputs_target.clone().detach()

            labels_list.append(labels.clone().detach().cpu())
            # 判断是否为最后一个时间步
            # if (t == 0).all():
            # 保存最后一个时间步的样本和标签
            final_sample = current_sample.clone().detach().cpu()

            updated_inputs_list.append(final_sample)

        count += 1
        logger.log(f"created {count * args.batch_size * dist_util.get_world_size()} samples")
        # logger.log(f"Total current_samples used: {current_sample_count}")

    # 在训练完成后，使用保存的数据进行评估
    dist.barrier()
    logger.log("sampling complete")
    evaluate_model(source_net, source_classifier,
                   source_net_frozen, source_classifier_frozen,
                    updated_inputs_list,
                   labels_list, args)

def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=4,
        D=32, # scaling factor
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
        pretrained_weights_path="", # resnet权重路径
        seed=42
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()
