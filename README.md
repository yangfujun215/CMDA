## Collaborative Model and Data Adaptation at Test Time

The code of Collaborative Model and Data Adaptation in Test Time.
> *IEEE Transactions on Circuits and Systems for Video Technology*

<p align="center">
<img width="394" height="197" alt="图片" src="https://github.com/user-attachments/assets/16d87dd6-df0a-4912-9173-1e2f828ad9f5" />
</p>

## Abstract
 Traditional Test-Time Adaptation (TTA) methods primarily focus on updating the parameters of a pre-trained source model to better fit the target domain. In contrast, recent diffusion-driven TTA approaches leverage an unconditional diffusion model trained on the source domain to map target samples towards the source distribution, without modifying the model parameters. In this paper, we propose to combine the strengths of model adaptation and data adaptation to achieve more effective alignment between the source model and target data. Unlike existing two-stage methods that perform model and data adaptation independently, we introduce a unified Collaborative Model and Data Adaptation (CMDA) framework that integrates the two processes in a mutually beneficial manner. Specifically, model predictions on synthetic target samples serve as category discriminative signals to guide the reverse diffusion process during data adaptation. Conversely, the synthetic data generated through data adaptation are used to progressively update and refine the source model. This bidirectional collaboration between model and data adaptation occurs iteratively, progressively aligning the source model with the target data. To further enhance prediction accuracy, we designed a lightweight and learnable aggregation network that ensembles predictions from the source and adapted models on both the original and synthetic target samples. This network dynamically integrates complementary predictions, improving the robustness and confidence of the final outputs. Extensive experiments on four benchmark datasets demonstrate that CMDA achieves state-of-the-art performance under the TTA setting.

<p align="center">
<img width="799" height="332" alt="图片" src="https://github.com/user-attachments/assets/5e300b5f-9d9a-4c66-9efe-d2f8e8cf4559" />
</p>
## On the following tasks  

- **CIFAR10** -> **ImageNet-C** (Standard)
- **ImageNet** -> **ImageNet-R** (Standard)
- **ImageNet** -> **ImageNet-W** (Standard)

## Compare this with the following methods 

- [MEMO](#)
- [TENT](#)
- [DiffPure](#)
- [DDA](#)
- [GDA](#)
- [SDA](#)

## Results

* ImageNet-C
<p align="center">
<img width="445" height="231" alt="图片" src="https://github.com/user-attachments/assets/25e3e54c-063a-4aba-b1c3-511380f576da" />
</p>

* ImageNet-R/W
<p align="center">
<img width="445" height="309" alt="图片" src="https://github.com/user-attachments/assets/0a94a637-74bd-42ee-89cb-d58fc6a970ae" />
</p>

* CIFAR-10/100-C
<p align="center">
<img width="443" height="227" alt="图片" src="https://github.com/user-attachments/assets/0f486a33-fc50-4541-a8c8-2fe4ba2a13a5" />
</p>

* Visualization: Grad-CAM results with prediction
classes and confidence scores displayed above the images.
<p align="center">
<img width="627" height="379" alt="图片" src="https://github.com/user-attachments/assets/63f8bc96-4056-45ac-a9c3-a85fdad24524" />
</p>
