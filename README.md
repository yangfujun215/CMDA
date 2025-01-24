Meeting Halfway: Collaborative Model and Data Adaptation in Test Time

The code of Collaborative Model and Data Adaptation in Test Time

Abstract
 Traditional Test-Time Adaptation (TTA) meth
ods primarily focus on adjusting the parame
ters of a pre-trained source domain model to
 align with the target domain data distribution.
 Recently, diffusion-driven TTA approaches have
 shown promising results by leveraging an uncon
ditional diffusion model pre-trained on the source
 domain to map target domain data to the source
 domain without modifying the model parameters.
 these methods typically concentrate exclusively on
 either updating the source domain model or trans
forming the input target data, thereby limiting their
 capacity to fully enhance model performance. In
 this paper, we propose a Collaborative Model and
 Data Adaptation (CMDA) framework that seam
lessly integrates model adaptation and data adapta
tion into a unified, synergistic process. In CMDA,
 model predictions from the model adaptation stage
 provide category-discriminative guidance for data
 adaptation. Simultaneously, the synthetic data gen
erated during the data adaptation stage, which re
tains the content and category-discriminative char
acteristics of target domain data, facilitates further
 optimization of the source model. This bidirec
tional collaboration drives iterative refinement, pro
gressively aligning the source model with the tar
get domain data. Extensive experiments on three
 benchmark datasets: ImageNet-C, ImageNet-W,
 and ImageNet-Rendition, demonstrate that CMDA
 achieves state-of-the-art performance, highlighting
 its effectiveness and robustness in addressing the
 challenges of test-time adaptation.

![图片](https://github.com/user-attachments/assets/c919d71d-b790-4789-bb07-d0d1b51f9001)

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
