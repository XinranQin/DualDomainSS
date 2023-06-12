# Dual-Domain Self-Supervised Learning and Model Adaption for Deep Compressive Imaging
This repository contains the Pytorch codes for paper "Dual-Domain Self-Supervised Learning and Model Adaption for Deep Compressive Imaging" (eccv 2022) by YuhuiQuan, XinranQin, TongyaoPang, HuiJi

## Content
* [Overview](#Overview)
* [Dataset](#Dataset)
* [Requirements](#Requirements)
* [Reference](#Reference)

## Overview
Deep learning has been one promising tool for compressive imaging whose task is to reconstruct latent images from their compressive measurements. Aiming at addressing the limitations of supervised deep learning-based methods caused by their prerequisite on the ground truths of latent images, this paper proposes an unsupervised approach that trains a deep image reconstruction model using only a set of compressive measurements. The training is self-supervised in the domain of measurements and the domain of images, using a double-head noiseinjected loss with a sign-flipping-based noise generator. In addition, the proposed scheme can also be used for efficiently adapting a trained model to a test sample for further improvement, with much less overhead than existing internal learning methods. Extensive experiments show that the proposed approach provides noticeable performance gain over existing unsupervised methods and competes well against the supervised ones
![image](https://github.com/XinranQin/DualDomainSS/blob/main/images/CS.png)

## Dataset
Dataset: [Training dataset](https://drive.google.com/drive/folders/1Vl0B0TZbwZwB590V8A2ipEAK-UjyNESi?usp=sharing "悬停显示")  
 
## Requirements
RTX3090 Python==3.8.0 Pytorch==1.8.0+cu101 scipy==1.7.3  


## References

```
@inproceedings{quan2022dual,
  title={Dual-Domain Self-supervised Learning and Model Adaption for Deep Compressive Imaging},
  author={Quan, Yuhui and Qin, Xinran and Pang, Tongyao and Ji, Hui},
  booktitle={Computer Vision--ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23--27, 2022, Proceedings, Part XXX},
  pages={409--426},
  year={2022}
}
```

