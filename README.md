# Mobile Computer Vision @ Facebook

This repository provides code and models for the following projects developed by Facebook for mobile:
* [FBNet: Hardware-Aware Efficient ConvNet Design via Differentiable Neural Architecture Search](https://arxiv.org/abs/1812.03443).  
* [ChamNet: Towards Efficient Network Design through Platform-Aware Model Adaptation](https://arxiv.org/abs/1812.08934)

We provide the following code and models:
* Pre-trained [ChamNet](https://github.com/facebookresearch/mobile-vision/tree/master/ChamNet) models.
* Pre-trained [FBNet](https://github.com/facebookresearch/mobile-vision/tree/master/FBNet) models.
* [CNN latency look-up table](https://github.com/facebookresearch/mobile-vision/tree/master/runtime_lut). We provided operator-level latency database using the latest caffe2 int8 inference engine (QNNPACK) on mobile phones.

## Pre-trained Models
We also provide different pre-trained ChamNet and FBNet models. The models are trained and evaluated using ImageNet 1k ([ILSVRC2012](http://www.image-net.org/challenges/LSVRC/2012/)) dataset. Validation top-1 accuracy for fp32 and int8 models are reported. Model latenies are benchmarked on a Samsung S8 CPU with NNPACK (fp32) and QNNAPCK (int8) engines using Caffe2. Note that our models are best used in int8 as they are searched using on-device int8 latency metrics.


| Model     | Resolution | Flops (M) | Params (M) | Accuracy (int8) | Latency (int8, ms) | Accuracy (fp32) | Latency (fp32, ms) |
|-----------|------------|-----------|------------|-----------------------|--------------------|-----------------------|--------------------|
| ChamNet-A | 224x224    | 552.9     | 6.2        | 74.6                  | **24.7**           | 75.4                  | 117.1              |
| ChamNet-B | 192x192    | 323.3     | 5.2        | 73.0                  | **15.5**           | 73.8                  | 74.4               |
| ChamNet-C | 224x224    | 211.6     | 3.4        | 70.6                  | **11.3**           | 71.6                  | 56.8               |
| ChamNet-D | 192x192    | 120.0     | 3.3        | 67.5                  | **7.1**            | 69.1                  | 36.3               |
| ChamNet-E | 160x160    | 53.7      | 2.3        | 62.4                  | **4.1**            | 64.2                  | 24.4               |
| ChamNet-F | 128x128    | 30.8      | 2.0        | 57.7                  | **2.7**            | 59.5                  | 15.4               |
| FBNet-A   | 224x224    | 249.4     | 4.3        | 71.7                  | **14.1**           | 73.3                  | 124.6              |
| FBNet-B   | 224x224    | 295.6     | 4.8        | 72.9                  | **17.3**           | 74.1                  | 167.7              |
| FBNet-B1  | 224x224    | 286.8     | 4.1        | 72.8                  | **17.0**           | 73.9                  | 127.5              |
| FBNet-C   | 224x224    | 383.0     | 5.5        | 73.2                  | **21.0**           | 74.9                  | 241.8              |
| FBNet-C1  | 224x224    | 374.2     | 4.9        | 73.3                  | **20.0**           | 74.5                  | 167.3              |
| FBNet-96  | 96x96      | 12.9      | 1.8        | 47.5                  | **2.3**            | 50.4                  | 22.8               |

The ChamNet and FBNet models are located at ChamNet/models and FBNet/models, respectively.

The models expect the input image to be loaded in the range of `[0, 255]` in BGR format and normalized using `mean = [0.406, 0.456, 0.485]` and `std = [0.225, 0.224, 0.229]`. The transformation should preferrably happen at preprocessing.


## CNN Latency Look-up Table

Recent studies have shown the importance of model optimization over direct metrics (e.g., latency) instead of indirect metrics (e.g., FLOPs).  However, platform-specific latency measurements require engineer efforts and can be slow and difficult to parallelize.  Thus, we provide a CNN latency look up table (LUT) to enable fast and reliable latency estimations.

Please see [CNN latency look-up table](https://github.com/facebookresearch/mobile-vision/tree/master/runtime_lut) for more details.


## License
This project is licensed under CC BY-NC, as found in the LICENSE file. If you use our code/models in your research, please cite our paper:

```
@article{wu2018fbnet,
  title={FBNet: Hardware-Aware Efficient ConvNet Design via Differentiable Neural Architecture Search},
  author={Wu, Bichen and Dai, Xiaoliang and Zhang, Peizhao and Wang, Yanghan and Sun, Fei and Wu, Yiming and Tian, Yuandong and Vajda, Peter and Jia, Yangqing and Keutzer, Kurt},
  journal={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2019}
}

@article{dai2018chamnet,
  title={ChamNet: Towards Efficient Network Design through Platform-Aware Model Adaptation},
  author={Dai, Xiaoliang and Zhang, Peizhao and Wu, Bichen and Yin, Hongxu and Sun, Fei and Wang, Yanghan and Dukhan, Marat and Hu, Yunqing and Wu, Yiming and Jia, Yangqing and others},
  journal={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2019}
}
```

## Acknowledgments
This work is developed by collaboration between Mobile Vision, Caffe2, and FAIR team at Facebook.

Thanks to Xiaoliang Dai, Marat Dukhan, Zijian He, Yunqing Hu, Yangqing Jia, Lingyi Liu, Yang Lu, Brad Stocks, Fei Sun, Yuandong Tian, Sam Tsai, Matt Uyttendaele, Peter Vajda, Yanghan Wang, Bichen Wu, Yiming Wu, Ran Xian, Fei Yang, Peizhao Zhang for their great contributions.
