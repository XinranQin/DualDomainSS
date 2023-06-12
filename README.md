# High-Quality Self-Supervised Snapshot Hyperspectral Imaging
This repository contains the Pytorch codes for paper "High-Quality Self-Supervised Snapshot Hyperspectral Imaging" (ICASSP 2022) by YuhuiQuan, XinranQin, MingqinChen, YanHuang

## Content
* [Overview](#Overview)
* [Dataset](#Dataset)
* [Requirements](#Requirements)
* [Reference](#Reference)

## Overview
Hyperspectral image (HSI) reconstruction is about recovering a 3D HSI from its 2D snapshot measurements, to which deep
models have become a promising approach. However, most existing studies train deep models on large amounts of organized data, the collection of which can be difficult in many applications. This paper leverages the image priors encoded
in untrained neural networks (NNs) to have a self-supervised learning method which is free from training datasets while adaptive to the statistics of a test sample. To induce better image priors and prevent the NN overfitting undesired solutions, we construct an unrolling-based NN equipped with fractional max pooling (FMP). Furthermore, the FMP is used with randomness to enable self-ensemble for reconstruction accuracy improvement. In the experiments, our self-supervised learning approach enjoys high-quality reconstruction and outperforms recent methods including the supervised ones.
![image](https://github.com/XinranQin/HQSCI/blob/main/Simulation/Data/image/result.png)

## Dataset
Simulation Dataset: [KAIST](https://drive.google.com/drive/folders/1I6YRHk14krGMW9Bx2V_hDCBtnwrq8LFN?usp=share_link "悬停显示")  [Mask](https://drive.google.com/file/d/121RW8hdT4BRZtBj3gb1t7GwZGoYKMtzl/view?usp=sharing "悬停显示")  
Real Dataset: [Real](https://drive.google.com/drive/folders/17vhfT93dwcg40JokNJJFa96nbTZb_RjB?usp=share_link "悬停显示") [Mask](https://drive.google.com/file/d/135Fj2IB4-6qhse3Oy0atXWt85fsTxEe-/view?usp=share_link "悬停显示") 
 
## Requirements
Pytorch==1.6.0 scipy==1.3.0  
### Run
For each sample on simulation data, run "Simulation/Self_supervised.py" to reconstruct 10 synthetic datasets. 


## References

```
@inproceedings{quan2022high,
  title={High-Quality Self-Supervised Snapshot Hyperspectral Imaging},
  author={Quan, Yuhui and Qin, Xinran and Chen, Mingqin and Huang, Yan},
  booktitle={ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1526--1530},
  year={2022},
  organization={IEEE}
}
```

