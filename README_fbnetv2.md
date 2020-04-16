
# [FBNetV2: Differentiable Neural Architecture Search for Spatial and Channel Dimensions](https://arxiv.org/abs/2004.05565)

<sub>Alvin Wan, Xiaoliang Dai, Peizhao Zhang, Zijian He, Yuandong Tian, Saining Xie, Bichen Wu, Matthew Yu, Tao Xu, Kan Chen, Peter Vajda, Joseph E. Gonzalez</sub>

Run a state-of-the-art family of models found by an augmented DNAS, which searches over both input resolutions and channels. Read the [full paper](https://arxiv.org/abs/2004.05565). Our work has been accepted to CVPR 2020.

**Table of Contents**

1. [Quickstart: Running pretrained FBNetV2 models](#1-quickstart-running-pretrained-fbnetv2-models)
2. [Results on Imagenet](#2-results-on-imagenet)

## 0. Getting Started

1. Install PyTorch ([pytorch.org](http://pytorch.org))
2. Clone this repository.
3. `cd mobile-vision`

## 1. Quickstart: Running pretrained FBNetV2 models

The following loads a pretrained FBNetV2 model.

```python
from mobile_cv.model_zoo.models.fbnet import fbnetv2
model = fbnetv2("dmasking_l3", pretrained=True)
model.eval()
```

## 2. Results on Imagenet

The following FBNetV2 pre-trained models are provided. The models are trained and evaluated using the ImageNet1k (ILSVRC2012) dataset. Validation top-1 and top-5 accuracy for fp32 models are reported.

|     Model      | Resolution | Flops (M) |  Top-1 Accuracy |
| -------------- | ---------- | --------- | -------------- |
| dmasking_f1    | 128x128    | 56        | 68.3           |
| dmasking_f2    | 224x224    | 85        | 71.1           |
| dmasking_f3    | 224x224    | 126     | 73.2           |
| dmasking_f4    | 224x224    | 238     | 76.0           |
| dmasking_l1    | 224x224    | 325     | 77.2           |
| dmasking_l2    | 256x256    | 422     | 78.1           |

|     Model      | Resolution | Params (M) |  Top-1 Accuracy |
| -------------- | ---------- | --------- | -------------- |
| dmasking_p1    | 288x288    | 2.64        | 73.9           |
| dmasking_p2    | 288x288    | 2.99        | 74.8           |
| dmasking_p3    | 288x288    | 4.00     | 75.9           |

## License

This project is licensed under CC BY-NC, as found in the LICENSE file. If you use our code/models in your research, please cite our paper:

```
@article{wan2020fbnetv2,
  title={FBNetV2: Differentiable Neural Architecture Search for Spatial and Channel Dimensions},
  author={},
  journal={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2020}
}

@article{wu2018fbnet,
  title={FBNet: Hardware-Aware Efficient ConvNet Design via Differentiable Neural Architecture Search},
  author={Wu, Bichen and Dai, Xiaoliang and Zhang, Peizhao and Wang, Yanghan and Sun, Fei and Wu, Yiming and Tian, Yuandong and Vajda, Peter and Jia, Yangqing and Keutzer, Kurt},
  journal={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2019}
}
```

## Acknowledgments
This work is developed by collaboration between Mobile Vision and FAIR team at Facebook.

Thanks to Kan Chen, Xiaoliang Dai, Zijian He, Yuandong Tian, Matt Uyttendaele, Peter Vajda, Alvin Wan, Yanghan Wang, Bichen Wu, Saining Xie, Matthew Yu, Peizhao Zhang for their contributions.
