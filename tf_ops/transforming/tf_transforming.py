''' The module orders neighbors in counterclockwise manner.
Author: Artem Komarichev
All Rights Reserved. 2018.
'''

import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
transforming_module=tf.load_op_library(os.path.join(BASE_DIR, 'tf_transforming_so.so'))
import numpy as np

def transform_neighbors(k, idx, input_xyz, query_xyz, query_normals,axis_x,axis_y,kernel,sita,interp,fit,radius):
    '''
    Order neighbors in counterclockwise manner.

    Input:
        input_xyz: (batch_size, ndataset, c) float32 array, input points
        query_xyz: (batch_size, npoint, c) float32 array, query points
        idx: (batch_size, npoint, k) int32 array, indecies of the k neighbor points
    Output:
        outi: (batch_size, npoint, k) int32 array, points orderred courterclockwise
        proj: (batch_size, npoint, k, 3) float32 array, projected neighbors on the local tangent plane
        angles: (batch_size, npoint, k) float32 array, values represents angles [0, 360)
    '''
    return transforming_module.transform_neighbors(input_xyz, query_xyz, query_normals, idx,kernel,axis_x,axis_y, k,sita,interp,fit,radius)
ops.NoGradient("TransformNeighbors")