import os

import torch
import mmcv
from matplotlib import pyplot as plt
from mmcls.apis import init_model
from PIL import Image
import numpy as np
import cv2
from mmcls.models import build_classifier
from mmcv import Config
# grad-cam相关引用
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from torch import nn

from image_adapt.model_use import Classifier_covnextT


# 1. 初始化模型
config_file = '/media/shared_space/wuyanzu/DDA-main/model_adapt/configs/ensemble/convnextT_ensemble_b64_imagenet.py'
checkpoint_file = '/media/shared_space/wuyanzu/DDA-main/ckpt/ConvNeXtT_source.pth'
device = 'cuda:0'  # 如果有GPU的话

cfg = Config.fromfile(config_file)
source_net = build_classifier(cfg.model)
source_net.to('cuda')
source_classifier = Classifier_covnextT(num_classes=1000).to('cuda')

# 加载预训练权重
checkpoint = torch.load(checkpoint_file, map_location='cuda')
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
for model2 in [source_net]:
    model2.backbone.load_state_dict(backbone_state_dict, strict=True)

classifier_state_dict = {}
for k, v in head_state_dict.items():
    if k == 'fc.weight' or k == 'classifier.weight':
        classifier_state_dict['fc.weight'] = v
    elif k == 'fc.bias' or k == 'classifier.bias':
        classifier_state_dict['fc.bias'] = v

for classifier in [source_classifier]:
    classifier.load_state_dict(classifier_state_dict, strict=True)

# print(source_net)

# 定义一个用于推理的包装模型
class InferenceModel(nn.Module):
    def __init__(self, backbone,  classifier):
        super(InferenceModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x):
        # 提取特征
        features = self.backbone(x)[-1]
        # 计算分类得分
        scores = self.classifier(features)
        if isinstance(scores, tuple):
            scores = scores[0]  # 如果输出是 tuple，假设分类结果在第一个元素
        return scores

# 实例化推理模型
inference_model = InferenceModel(
    backbone=source_net.backbone,
    classifier=source_classifier
)
inference_model.eval()  # 设置为评估模式


target_layer = source_net.backbone.stages[3][-1].depthwise_conv

cam = GradCAM(model=inference_model, target_layers=[target_layer])



# 2. 处理多层目录结构的函数
def generate_grad_cam_for_folders(input_root, output_root):
    for root, dirs, files in os.walk(input_root):  # 遍历所有子目录和文件
        for file in files:
            if file.lower().endswith(('jpg', 'jpeg', 'png', 'bmp')):
                img_path = os.path.join(root, file)

                # 计算输出目录路径，保持目录结构一致
                relative_path = os.path.relpath(root, input_root)  # 相对于输入目录的路径
                output_dir = os.path.join(output_root, relative_path)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                output_path = os.path.join(output_dir, f"cam_{file}")
                print(f"Processing: {img_path}")

                try:
                    # 读取和预处理图像
                    img = Image.open(img_path).convert('RGB')
                    img = np.array(img)
                    input_tensor = preprocess_image(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).to(device)

                    # 预测类别
                    with torch.no_grad():
                        features = source_net.backbone(input_tensor)[-1]
                        scores = source_classifier(features)
                        scores = scores[0] if isinstance(scores, tuple) else scores
                    pred_class = scores.argmax().item()

                    # 生成 Grad-CAM
                    targets = [ClassifierOutputTarget(pred_class)]
                    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

                    # 反标准化图像
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    img_normalized = (input_tensor.cpu().numpy()[0].transpose(1, 2, 0) * std + mean)
                    img_normalized = np.clip(img_normalized, 0, 1)

                    # 叠加热力图
                    cam_image = show_cam_on_image(img_normalized, grayscale_cam, use_rgb=True)

                    # 保存结果
                    cv2.imwrite(output_path, cam_image[..., ::-1])
                    print(f"Saved: {output_path}")
                except Exception as e:
                    print(f"Error processing {img_path}: {str(e)}")

img_dir = '/media/shared_space/wuyanzu/imagenet-c/gaussian_noise/5/n02085782/'
# img_dir = '/media/shared_space/dongxiaolin/maple1/data/imagenet/images/val/'
# img_dir = '/media/shared_space/wuyanzu/ImageNet-C-Syn/gaussian_noise/5/n02123045'
output_dir = '/media/shared_space/wuyanzu/ImageNet-C-Syn/test22'

# 5. 执行批量处理
generate_grad_cam_for_folders(img_dir, output_dir)

# 6. 完成提示
print("Grad-CAM generation completed for all images!")

