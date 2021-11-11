# CNN Latency Look-up Table

Recent studies have shown the importance of model optimization over direct metrics (e.g., latency) instead of indirect metrics (e.g., FLOPs).  However, platform-specific latency measurements require engineer efforts and can be slow and difficult to parallelize.  Thus, we provide a CNN latency look up table (LUT) to enable fast and reliable latency estimations.

The LUT has been used in the following projects:
* [ChamNet: Towards Efficient Network Design through Platform-Aware Model Adaptation](https://arxiv.org/abs/1812.08934)
* [FBNet: Hardware-Aware Efficient ConvNet Design via Differentiable Neural Architecture Search](https://arxiv.org/abs/1812.03443)

## Supported Devices
We measured all the latency for common int8 operators (Int8Conv, Int8ConvRelu etc.) using the quantized neural network package inference engine ([QNNPACK](https://code.fb.com/ml-applications/qnnpack/)), an open source library for efficient deep neural network model deployment for Caffe2. Please make sure your Caffe2 has QNNPACK based Int8 operators before running inference on it. We measure latency with the batch size of 1.

We supported the following devices:
* Samsung Galaxy S8 (SM-G950U-7.0-24)

More devices (iPhone etc.) supports will be added.

## Requirements
- Python >= 2.7
- Caffe2
- PyTorch >= 0.4.0

## Examples
The operator level latency database is available for download [here](https://dl.fbaipublicfiles.com/fbnet/lut/SM-G950U-7.0-24-op_lut_database_caffe2.json.zip). Please unzip the file to `runtime_lut/data` before using.

We provide several pre-trained models for ChamNet and FBNet.  To extract the latency from the
CNN latency look-up table (LUT):

```bash
python scripts/get_runtime.py [--model_path MODEL_PATH] [--db_dir LUT_DATABASE]
  [--input_dims INPUT_DIMENSIONS] [--input_dtype INPUT_DATA_TYPES]
  [--device DEVICE]
```
For example, to extract the latency of ChamNet-A:
```bash
python runtime_lut/scripts/get_runtime.py \
--model_path ChamNet/models/ChamNet-A/int8 \
--input_dims '{"data": [1, 3, 224, 224]}'
```

The output contains the run-time latency for each operator in the network.  We provide example outputs as follows (all the latency values have the unit of 10^-6s)
```bash
op type NCHW2NHWC with input [[1, 3, 224, 224]]
run_time: [276.0]
op type Int8Quantize with input [[1, 224, 224, 3]]
run_time: [127.0]
op type Int8ConvRelu with input [[1, 224, 224, 3], [32, 3, 3, 3], [32]]
run_time: [960.0]
op type Int8ConvRelu with input [[1, 112, 112, 32], [32, 1, 1, 32], [32]]
run_time: [674.0]
op type Int8ConvRelu with input [[1, 112, 112, 32], [32, 3, 3, 1], [32]]
run_time: [363.0]
...
op type Int8Dequantize with input [[1, 1000]]
run_time: [5.0]
All operators found in the table? True
The network run-time latency is 25330.0
```

## License
This project is licensed under CC BY-NC, as found in the LICENSE file. If you use our code/models in your research, please cite our paper:

```
@article{dai2018chamnet,
  title={ChamNet: Towards Efficient Network Design through Platform-Aware Model Adaptation},
  author={Dai, Xiaoliang and Zhang, Peizhao and Wu, Bichen and Yin, Hongxu and Sun, Fei and Wang, Yanghan and Dukhan, Marat and Hu, Yunqing and Wu, Yiming and Jia, Yangqing and others},
  journal={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2019}
}

@article{wu2018fbnet,
  title={FBNet: Hardware-Aware Efficient ConvNet Design via Differentiable Neural Architecture Search},
  author={Wu, Bichen and Dai, Xiaoliang and Zhang, Peizhao and Wang, Yanghan and Sun, Fei and Wu, Yiming and Tian, Yuandong and Vajda, Peter and Jia, Yangqing and Keutzer, Kurt},
  journal={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2019}
}
```

## Acknowledgments
This work is developed by collaboration between Mobile Vision, Caffe2, and FAIR team at Facebook.

Thanks to Xiaoliang Dai, Marat Dukhan, Zijian He, Yunqing Hu, Yangqing Jia, Lingyi Liu, Yang Lu, Brad Stocks, Fei Sun, Yuandong Tian, Sam Tsai, Matt Uyttendaele, Peter Vajda, Yanghan Wang, Bichen Wu, Yiming Wu, Ran Xian, Fei Yang, Peizhao Zhang for their great contributions.
