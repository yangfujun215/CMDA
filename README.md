## ✏️ Collaborative Model and Data Adaptation at Test Time

The code of Collaborative Model and Data Adaptation in Test Time.
> *IEEE Transactions on Circuits and Systems for Video Technology

## Abstract
 Traditional Test-Time Adaptation (TTA) methods primarily focus on updating the parameters of a pre-trained source model to better fit the target domain. In contrast, recent diffusion-driven TTA approaches leverage an unconditional diffusion model trained on the source domain to map target samples towards the source distribution, without modifying the model parameters. In this paper, we propose to combine the strengths of model adaptation and data adaptation to achieve more effective alignment between the source model and target data. Unlike existing two-stage methods that perform model and data adaptation independently, we introduce a unified Collaborative Model and Data Adaptation (CMDA) framework that integrates the two processes in a mutually beneficial manner. Specifically, model predictions on synthetic target samples serve as category discriminative signals to guide the reverse diffusion process during data adaptation. Conversely, the synthetic data generated through data adaptation are used to progressively update and refine the source model. This bidirectional collaboration between model and data adaptation occurs iteratively, progressively aligning the source model with the target data. To further enhance prediction accuracy, we designed a lightweight and learnable aggregation network that ensembles predictions from the source and adapted models on both the original and synthetic target samples. This network dynamically integrates complementary predictions, improving the robustness and confidence of the final outputs. Extensive experiments on four benchmark datasets demonstrate that CMDA achieves state-of-the-art performance under the TTA setting.



## On the following tasks 📸

- **CIFAR10** -> **ImageNet-C** (Standard)
- **ImageNet** -> **ImageNet-w** (Standard)
- **ImageNet** -> **ImageNet-R** (Standard)

## Compare this with the following methods 🌈

- [MEMO](#)
- [TENT](#)
- [DiffPure](#)
- [DDA](#)
- [GDA](#)
- [SDA](#)
