# A PyTorch implementation for PyramidNets (Deep Pyramidal Residual Networks)

This repository contains a [PyTorch](http://pytorch.org/) implementation for the paper: [Deep Pyramidal Residual Networks](https://arxiv.org/pdf/1610.02915.pdf) (CVPR 2017, Dongyoon Han*, Jiwhan Kim*, and Junmo Kim, (equally contributed by the authors*)). The code in this repository is based on the example provided in [PyTorch examples](https://github.com/pytorch/examples/tree/master/imagenet) and the nice implementation of [Densely Connected Convolutional Networks](https://github.com/andreasveit/densenet-pytorch).

Two other implementations with [LuaTorch](http://torch.ch/) and [Caffe](http://caffe.berkeleyvision.org/) are provided:
1. [A LuaTorch implementation](https://github.com/jhkim89/PyramidNet) for PyramidNets,
2. [A Caffe implementation](https://github.com/jhkim89/PyramidNet-caffe) for PyramidNets.

## Usage examples
To train additive PyramidNet-110 (alpha=48 without bottleneck) on CIFAR-10 dataset with single-GPU:
```
CUDA_VISIBLE_DEVICES=0 python train.py --alpha 64 --depth 110 --no-bottleneck --batchsize 32 --print-freq 1 --expname PyramidNet-110 --dataset cifar10
```
To train additive PyramidNet-164 (alpha=48 with bottleneck) on CIFAR-100 dataset with 4 GPU:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --alpha 48 --depth 164 --batchsize 128 --print-freq 1 --expname PyramidNet-164 --dataset cifar100
```

### Notes
1. This implementation is for CIFAR-10 and CIFAR-100 datasets with add-PyramidNet architecture, and the code will be updated for Imagenet-1k dataset soon. 
2. The traditional data augmentation for CIFAR datasets are used by following [fb.resnet.torch](https://github.com/facebook/fb.resnet.torch).
3. To use multi-GPU, data parallelism in PyTorch should be applied [i.e., model = torch.nn.DataParallel(model).cuda()].  
4. An example code for ResNet is also included (an example code for pre-ResNet will be uploded soon).

### Tracking training progress with TensorBoard
Thanks to the [implementation](https://github.com/andreasveit/densenet-pytorch), which support the [TensorBoard](https://www.tensorflow.org/get_started/summaries_and_tensorboard) to track training progress efficiently, all the experiments can be tracked with [tensorboard_logger](https://github.com/TeamHG-Memex/tensorboard_logger).

Tensorboard_logger can be installed with 
```
pip install tensorboard_logger
```

## Paper Preview
### Abstract
Deep convolutional neural networks (DCNNs) have shown remarkable performance in image classification tasks in recent years. Generally, deep neural network architectures are stacks consisting of a large number of convolution layers, and they perform downsampling along the spatial dimension via pooling to reduce memory usage. At the same time, the feature map dimension (i.e., the number of channels) is sharply increased at downsampling locations, which is essential to ensure effective performance because it increases the capability of high-level attributes. Moreover, this also applies to residual networks and is very closely related to their performance. In this research, instead of using downsampling to achieve a sharp increase at each residual unit, we gradually increase the feature map dimension at all the units to involve as many locations as possible. This is discussed in depth together with our new insights as it has proven to be an effective design to improve the generalization ability. Furthermore, we propose a novel residual unit capable of further improving the classification accuracy with our new network architecture. Experiments on benchmark CIFAR datasets have shown that our network architecture has a superior generalization ability compared to the original residual networks.

### Network architecture details
1. Schematic illustration of comparision of several units: (a) basic residual units, (b) bottleneck, (c) wide residual units, and (d) our pyramidal residual units, and (e) our pyramidal bottleneck residual units:

![image](https://user-images.githubusercontent.com/31481676/32218603-c9e136bc-be6e-11e7-94ee-aa31c5887fdd.png)

2. Visual illustration of (a) additive PyramidNet (the feature map dimension of each unit increases linearly), (b) multiplicative PyramidNet (the feature map dimension of each unit increases geometrically), and (c) comparison of (a) and (b):

![image](https://user-images.githubusercontent.com/31481676/32218836-8f4b667a-be6f-11e7-9410-0619cfe0d0e2.png)

### Results
1. The results are readily reproduced, which show the same performances as those reproduced with [A LuaTorch implementation](https://github.com/jhkim89/PyramidNet) for PyramidNets.

2. Comparison of the state-of-the-art networks by [Top-1 Test Error Rates VS # of Parameters]:

![image](https://user-images.githubusercontent.com/31481676/32331973-9d7dad2a-c027-11e7-9828-ac00fea0e5b5.png)

2. Top-1 test error rates (%) on CIFAR datasets are shown in the following table. All the results of PyramidNets are produced with additive PyramidNets, and α denotes alpha (the widening factor). “Output Feat. Dim.” denotes the feature dimension of just before the last softmax classifier.

![image](https://user-images.githubusercontent.com/31481676/32329781-5d47ff90-c021-11e7-81ed-ffac05e8ea98.png)

## Updates

## Citation
Please cite our paper if PyramidNets are used: 
```
@article{DPRN,
  title={Deep pyramidal residual networks},
  author={Han, Dongyoon and Kim, Jiwhan and Kim, Junmo},
  journal={arXiv preprint arXiv:1610.02915},
  year={2016}
}
```
If this implementation is useful, please also cite or acknowledge this repository on your work.

## Contact
Dongyoon Han (dyhan@kaist.ac.kr),
Jiwhan Kim (jhkim89@kaist.ac.kr),
Junmo Kim (junmo.kim@kaist.ac.kr)
