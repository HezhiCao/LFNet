# LFNet

### LFNet: *Local Rotation Invariant Coordinate Frame for Robust Point Cloud Analysis* [[pdf](https://www.researchgate.net/publication/348141380_LFNet_Local_Rotation_Invariant_Coordinate_Frame_for_Robust_Point_Cloud_Analysis)] [[CVF](https://ieeexplore.ieee.org/document/9311810)]

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

This work is based on our SPL paper.

### Installation

Install [TensorFlow](https://www.tensorflow.org/install/). The code is tested under TF1.13 GPU version and Python 3.6 on Ubuntu 16.04. 

#### Compile Customized TF Operators

The TF operators are included under `tf_ops`, you need to compile them (check `tf_xxx_compile.sh` under each ops subfolder) first. Update `nvcc` and `python` path if necessary. The code is tested under TF1.13. If you are using earlier version it's possible that you need to remove the `-D_GLIBCXX_USE_CXX11_ABI=0` flag in g++ command in order to compile correctly.

To compile the operators in TF version >=1.4, you need to modify the compile scripts slightly.

First, find Tensorflow include and library paths.

```
    TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
    TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
```

Then, add flags of `-I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework` to the `g++` commands.

### Dataset

__Shape Classification__

Download and unzip [ModelNet40](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip) (1.6G). Replace `$data_path$` in `cfgs/config_*_cls.yaml` with the dataset parent path.

__ShapeNet Part Segmentation__

Download and unzip [ShapeNet Part](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip) (674M). Replace `$data_path$` in `cfgs/config_*_partseg.yaml` with the dataset path.

### Usage

#### Shape Classification

To train a PointNet++ model to classify ModelNet40 shapes (using point clouds with XYZ coordinates):

```
    python train.py
```

After training, to evaluate the classification accuracies (with optional multi-angle voting):

```
    python evaluate.py
```

You can use our model `cls/model_iter_113_acc_0.905592_category.ckpt` as the checkpoint in `evaluate.py`, and after this voting you will get an accuracy of 91.08% if all things go right.

#### Object Part Segmentation

To train a model to segment object parts for ShapeNet models:

```
    cd shapenet_seg
    python train_seg.py
```

evaluate:

```
    cd shapenet_seg
    python evaluate_shapenet.py
```

## Acknowledgement

The code is heavily borrowed from [A-CNN](https://github.com/artemkomarichev/a-cnn).

## Contact

If you have some ideas or questions about our research to share with us, please contact caohezhi21@mail.ustc.edu.cn