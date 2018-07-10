# A PyTorch implementation for PyramidNets (Deep Pyramidal Residual Networks)

This repository contains a [PyTorch](http://pytorch.org/) implementation for the paper: [Deep Pyramidal Residual Networks](https://arxiv.org/pdf/1610.02915.pdf) (CVPR 2017, Dongyoon Han*, Jiwhan Kim*, and Junmo Kim, (equally contributed by the authors*)). The code in this repository is based on the example provided in [PyTorch examples](https://github.com/pytorch/examples/tree/master/imagenet) and the nice implementation of [Densely Connected Convolutional Networks](https://github.com/andreasveit/densenet-pytorch).

Two other implementations with [LuaTorch](http://torch.ch/) and [Caffe](http://caffe.berkeleyvision.org/) are provided:
1. [A LuaTorch implementation](https://github.com/jhkim89/PyramidNet) for PyramidNets,
2. [A Caffe implementation](https://github.com/jhkim89/PyramidNet-caffe) for PyramidNets.

## Usage examples
To train additive PyramidNet-200 (alpha=300 with bottleneck) on ImageNet-1k dataset with 8 GPUs:
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --data ~/dataset/ILSVRC/Data/CLS-LOC/ --net_type pyramidnet --lr 0.05 --batch_size 128 --depth 200 -j 16 --alpha 300 --print-freq 1 --expname PyramidNet-200 --dataset imagenet --epochs 100
```
To train additive PyramidNet-110 (alpha=48 without bottleneck) on CIFAR-10 dataset with a single-GPU:
```
CUDA_VISIBLE_DEVICES=0 python train.py --net_type pyramidnet --alpha 64 --depth 110 --no-bottleneck --batch_size 32 --lr 0.025 --print-freq 1 --expname PyramidNet-110 --dataset cifar10 --epochs 300
```
To train additive PyramidNet-164 (alpha=48 with bottleneck) on CIFAR-100 dataset with 4 GPUs:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --net_type pyramidnet --alpha 48 --depth 164 --batch_size 128 --lr 0.5 --print-freq 1 --expname PyramidNet-164 --dataset cifar100 --epochs 300
```

### Notes
1. This implementation contains the training (+test) code for add-PyramidNet architecture on ImageNet-1k dataset, CIFAR-10 and CIFAR-100 datasets.
2. The traditional data augmentation for ImageNet and CIFAR datasets are used by following [fb.resnet.torch](https://github.com/facebook/fb.resnet.torch).
3. The example codes for ResNet and Pre-ResNet are also included.  
4. For efficient training on ImageNet-1k dataset, Intel MKL and NVIDIA(nccl) are prerequistes. Please check the [official PyTorch github](https://github.com/pytorch/pytorch) for the installation.

### Tracking training progress with TensorBoard
Thanks to the [implementation](https://github.com/andreasveit/densenet-pytorch), which support the [TensorBoard](https://www.tensorflow.org/get_started/summaries_and_tensorboard) to track training progress efficiently, all the experiments can be tracked with [tensorboard_logger](https://github.com/TeamHG-Memex/tensorboard_logger).

Tensorboard_logger can be installed with 
```
pip install tensorboard_logger
```

## Paper Preview
### Abstract
Deep convolutional neural networks (DCNNs) have shown remarkable performance in image classification tasks in recent years. Generally, deep neural network architectures are stacks consisting of a large number of convolution layers, and they perform downsampling along the spatial dimension via pooling to reduce memory usage. At the same time, the feature map dimension (i.e., the number of channels) is sharply increased at downsampling locations, which is essential to ensure effective performance because it increases the capability of high-level attributes. Moreover, this also applies to residual networks and is very closely related to their performance. In this research, instead of using downsampling to achieve a sharp increase at each residual unit, we gradually increase the feature map dimension at all the units to involve as many locations as possible. This is discussed in depth together with our new insights as it has proven to be an effective design to improve the generalization ability. Furthermore, we propose a novel residual unit capable of further improving the classification accuracy with our new network architecture. Experiments on benchmark CIFAR datasets have shown that our network architecture has a superior generalization ability compared to the original residual networks.

### Schematic Illustration 
We provide a simple schematic illustration to compare the several network architectures, which have (a) basic residual units, (b) bottleneck, (c) wide residual units, and (d) our pyramidal residual units, and (e) our pyramidal bottleneck residual units, as follows:

![image](https://user-images.githubusercontent.com/31481676/32218603-c9e136bc-be6e-11e7-94ee-aa31c5887fdd.png)

### Experimental Results
1. The results are readily reproduced, which show the same performances as those reproduced with [A LuaTorch implementation](https://github.com/jhkim89/PyramidNet) for PyramidNets.

2. Comparison of the state-of-the-art networks by [Top-1 Test Error Rates VS # of Parameters]:

![image](https://user-images.githubusercontent.com/31481676/32331973-9d7dad2a-c027-11e7-9828-ac00fea0e5b5.png)

2. Top-1 test error rates (%) on CIFAR datasets are shown in the following table. All the results of PyramidNets are produced with additive PyramidNets, and α denotes alpha (the widening factor). “Output Feat. Dim.” denotes the feature dimension of just before the last softmax classifier.

![image](https://user-images.githubusercontent.com/31481676/32329781-5d47ff90-c021-11e7-81ed-ffac05e8ea98.png)

### ImageNet-1k Pretrained Models 
* A pretrained model of PyramidNet-101-360 is trained from scratch using the code in this repository (single-crop (224x224) validation error rates are reported):

| Network Type | Alpha |  # of Params |  Top-1 err(%) | Top-5 err(%) | Model File|
| :-------------: | :-------------: |  :-------------: |:-------------: |:-------------: | :----------:|
| ResNet-101 (Caffe model) | - | 44.7M | 23.6 | 7.1 | [Original Model](https://github.com/KaimingHe/deep-residual-networks) |
| ResNet-101 (Luatorch model) |  - | 44.7M | 22.44 | 6.21 | [Original Model](https://github.com/facebook/fb.resnet.torch/tree/master/pretrained) |
| PyramidNet-v1-101 | 360 | 42.5M | 21.98 | 6.20 | [Download](https://drive.google.com/file/d/1d_xBxRhWvq_4yxcoy3qB4JEQ7zsNx1Om/view?usp=sharing) |
* Note that the above widely-used ResNet-101 (Caffe model) is trained with the images, where the pixel intensities are in [0,255] and are centered by the mean image, our PyramidNet-101 is trained with the images where the pixel values are standardized.
* The model is originally trained with PyTorch-0.4, and the keys of num_batches_tracked were excluded for convenience (the BatchNorm2d layer in PyTorch (>=0.4) contains the key of num_batches_tracked by track_running_stats).
  
## Updates
1. Some minor bugs are fixed (2018/02/22).
2. train.py is updated (including ImagNet-1k training code) (2018/04/06).
3. resnet.py and PyramidNet.py are updated (2018/04/06).
4. preresnet.py (Pre-ResNet architecture) is uploaded (2018/04/06).
5. A pretrained model using PyTorch is uploaded (2018/07/09).

## Citation
Please cite our paper if PyramidNets are used: 
```
@article{DPRN,
  title={Deep Pyramidal Residual Networks},
  author={Han, Dongyoon and Kim, Jiwhan and Kim, Junmo},
  journal={IEEE CVPR},
  year={2017}
}
```
If this implementation is useful, please cite or acknowledge this repository on your work.

## Contact
Dongyoon Han (dyhan@kaist.ac.kr),
Jiwhan Kim (jhkim89@kaist.ac.kr),
Junmo Kim (junmo.kim@kaist.ac.kr)
