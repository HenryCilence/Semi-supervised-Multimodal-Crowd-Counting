<div align="center">
  <h2><b> A Benchmark for Semi-supervised Multi-modal Crowd Counting </b></h2>
</div>

## Overview
We present a benchmark for semi-supervised multi-modal crowd counting. [arXiv](Coming soon)

It contains a standardized protocol that specifies the labeled-unlabeled data partition across different labeled ratios (5%, 10%, 40%), and the adaptation of a diverse set of representative baselines, including existing fully supervised multi-modal methods and semi-supervised single-modal methods.

The labeled sample lists of each dataset under each labeled ratio are presented in ```label_list```. They are generated via fixed-interval sampling over the filename-sorted training set to avoid sampling bias.

## Getting Started

1. Download data. You can download the **RGBT-CC** dataset from [IADM](https://github.com/chen-judge/RGBTCrowdCounting), and download the **DroneRGBT** dataset from [MMCCN](https://github.com/VisDrone/DroneRGBT).
2. Train the model using ```train.py```.

## Notes
1. Following the format in [BL](https://github.com/zhiheng-ma/Bayesian-Crowd-Counting), the annotation .npy files contain three columns. The first two columns represent the position of each annotation point. The third column records the distances between each annotation point and its nearest neighbors. A script to generate 3-column .npy files from the original 2-column files and the generated 3-column files are available in [BM](https://github.com/HenryCilence/Broker-Modality-Crowd-Counting).
2. For multi-modal methods, if you would like to train the original models on labeled samples only, please use ```from utils.regression_trainer_multi import RegTrainer``` in ```train.py``` and set ```--unlabel-start``` larger than ```--max-epoch```. If you would like to train the adapted models using Mean Teacher, please use ```from utils.regression_trainer_mt import RegTrainer``` in ```train.py```. We recommend other default settings. For **DEFNet**, please specially set ```--batch-size``` to ```2``` and ```--downsample-ratio``` to ```4``` following their original repo.
3. Some pretrained weights for the model are available at https://pan.baidu.com/s/1Cf3y_KaVNubeXfsgG2aeJw?pwd=tf66, including the ConvNeXt-S encoder for **MC3Net** and the broker modality generator for **BM**.

## Acknowledgement

This work was supported by the National Key R&D Program of China (No. 2025YFC3811300) and the National Natural Science Foundation of China (Grant Nos. 62376070 and 62076195).

Meanwhile, we appreciate the following GitHub repos for their valuable code and effort:
- BL (https://github.com/zhiheng-ma/Bayesian-Crowd-Counting)
- IADM (https://github.com/chen-judge/RGBTCrowdCounting)
- MC3Net (https://github.com/WBangG/MC3Net)
- DEFNet (https://github.com/panyi95/DEFNet)
- CAGNet (https://github.com/WBangG/CAGNet)
- BM (https://github.com/HenryCilence/Broker-Modality-Crowd-Counting)
- DACount (https://github.com/LoraLinH/Semi-supervised-Crowd-Counting-via-Density-Agency)
- P3Net (https://github.com/LoraLinH/Semi-supervised-Counting-via-Pixel-by-pixel-Density-Distribution-Modelling)

## Contact

If you have any questions or concerns, please contact us at menghaoliang2002@163.com.

## Citation

If you find this repository useful in your research, please consider citing our work as follows:

```
(ArXiv BibTex coming soon)
```
