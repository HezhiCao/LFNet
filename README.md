# LFNet

### LFNet: *Local Rotation Invariant Coordinate Frame for Robust Point Cloud Analysis*

Created by Hezhi Cao, Ronghui Zhan, Yanxin Ma, Chao Ma and Jun zhang from National University of Defense Technology.

### Citation

If you find our work useful in your research, please consider citing:

```
    @article{cao2021LFNet,
      title={LFNet: Local Rotation Invariant Coordinate Frame for Robust Point Cloud Analysis},
      author={Hezhi Cao, Ronghui Zhan, Yanxin Ma, Chao Ma and Jun zhang},
      journal={IEEE Signal Processing Letters},
      year={2021}
    }
```

### Introduction



### Installation

Install [TensorFlow](https://www.tensorflow.org/install/). The code is tested under TF1.13 GPU version and Python 3.6 on Ubuntu 16.04. There are also some dependencies for a few Python libraries for data processing and visualizations like `cv2`, `h5py` etc. It's highly recommended that you have access to GPUs.

#### Compile Customized TF Operators

The TF operators are included under `tf_ops`, you need to compile them (check `tf_xxx_compile.sh` under each ops subfolder) first. Update `nvcc` and `python` path if necessary. The code is tested under TF1.13.1. If you are using earlier version it's possible that you need to remove the `-D_GLIBCXX_USE_CXX11_ABI=0` flag in g++ command in order to compile correctly.

To compile the operators in TF version >=1.4, you need to modify the compile scripts slightly.

First, find Tensorflow include and library paths.

```
    TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
    TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
```

Then, add flags of `-I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework` to the `g++` commands.

### Usage

#### Shape Classification

To train a PointNet++ model to classify ModelNet40 shapes (using point clouds with XYZ coordinates):

```
    python train.py
```

To see all optional arguments for training:

```
    python train.py -h
```

If you have multiple GPUs on your machine, you can also run the multi-GPU version training (our implementation is similar to the tensorflow [cifar10 tutorial](https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10)):

```
    CUDA_VISIBLE_DEVICES=0,1 python train_multi_gpu.py --num_gpus 2
```

After training, to evaluate the classification accuracies (with optional multi-angle voting):

```
    python evaluate.py --num_votes 12 
```

*Side Note:* For the XYZ+normal experiment reported in our paper: (1) 5000 points are used and (2) a further random data dropout augmentation is used during training (see commented line after `augment_batch_data` in `train.py` and (3) the model architecture is updated such that the `nsample=128` in the first two set abstraction levels, which is suited for the larger point density in 5000-point samplings.

To use normal features for classification: You can get our sampled point clouds of ModelNet40 (XYZ and normal from mesh, 10k points per shape) [here (1.6GB)](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip). Move the uncompressed data folder to `data/modelnet40_normal_resampled`

#### Object Part Segmentation

To train a model to segment object parts for ShapeNet models:

```
    cd part_seg
    python train.py
```

Preprocessed ShapeNetPart dataset (XYZ, normal and part labels) can be found [here (674MB)](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip). Move the uncompressed data folder to `data/shapenetcore_partanno_segmentation_benchmark_v0_normal`

#### Visualization Tools

We have provided a handy point cloud visualization tool under `utils`. Run `sh compile_render_balls_so.sh` to compile it and then you can try the demo with `python show3d_balls.py` The original code is from [here](http://github.com/fanhqme/PointSetGeneration).

### License

Our code is released under MIT License (see LICENSE file for details).

### Updates

- 

### Related Projects

- [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](http://stanford.edu/~rqi/pointnet) by Qi et al. (CVPR 2017 Oral Presentation). Code and data released in [GitHub](https://github.com/charlesq34/pointnet).
- [Frustum PointNets for 3D Object Detection from RGB-D Data](https://arxiv.org/abs/1711.08488) by Qi et al. (CVPR 2018) A novel framework for 3D object detection with RGB-D data. Based on 2D boxes from a 2D object detector on RGB images, we extrude the depth maps in 2D boxes to point clouds in 3D space and then realize instance segmentation and 3D bounding box estimation using PointNet/PointNet++. The method proposed has achieved first place on KITTI 3D object detection benchmark on all categories (last checked on 11/30/2017). Code and data release TBD.