""" PointNet++ Layers

Author: Charles R. Qi
Date: November 2017
"""

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/sampling'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/grouping'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/3d_interpolation'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/transforming'))
from tf_sampling import farthest_point_sample, gather_point
from tf_ops.grouping.tf_grouping import  group_point, knn_point
from tf_interpolate import three_nn, three_interpolate
from tf_transforming import transform_neighbors
import tensorflow as tf
import numpy as np
import tf_util

def sample_and_group(npoint, radius, nsample, xyz, points, tnet_spec=None, knn=False, use_xyz=True):
    '''
    Input:
        npoint: int32
        radius: float32
        nsample: int32
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        tnet_spec: dict (keys: mlp, mlp2, is_training, bn_decay), if None do not apply tnet
        knn: bool, if True use kNN instead of radius search
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Output:
        new_xyz: (batch_size, npoint, 3) TF tensor
        new_points: (batch_size, npoint, nsample, 3+channel) TF tensor
        idx: (batch_size, npoint, nsample) TF tensor, indices of local points as in ndataset points
        grouped_xyz: (batch_size, npoint, nsample, 3) TF tensor, normalized point XYZs
            (subtracted by seed point XYZ) in local regions
    '''

    indecies = farthest_point_sample(npoint, xyz)
    new_xyz = gather_point(xyz, indecies) # (batch_size, npoint, 3)
    new_normals = gather_point(normals, indecies) # (batch_size, npoint, 3)
    _,idx = knn_point(nsample, xyz, new_xyz)
    grouped_xyz = group_point(xyz, idx) # (batch_size, npoint, nsample, 3)
    grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1,1,nsample,1]) # translation normalization
    if tnet_spec is not None:
        grouped_xyz = tnet(grouped_xyz, tnet_spec)
    if points is not None:
        grouped_points = group_point(points, idx) # (batch_size, npoint, nsample, channel)
        if use_xyz:
            new_points = tf.concat([grouped_xyz, grouped_points], axis=-1) # (batch_size, npoint, nample, 3+channel)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz

    return new_xyz, new_points, new_normals, idx, grouped_xyz


def sample_and_group_all(xyz, points, use_xyz=True):
    '''
    Inputs:
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Outputs:
        new_xyz: (batch_size, 1, 3) as (0,0,0)
        new_points: (batch_size, 1, ndataset, 3+channel) TF tensor
    Note:
        Equivalent to sample_and_group with npoint=1, radius=inf, use (0,0,0) as the centroid
    '''
    batch_size = xyz.get_shape()[0].value
    nsample = xyz.get_shape()[1].value
    new_xyz = tf.constant(np.tile(np.array([0,0,0]).reshape((1,1,3)), (batch_size,1,1)),dtype=tf.float32) # (batch_size, 1, 3)
    idx = tf.constant(np.tile(np.array(range(nsample)).reshape((1,1,nsample)), (batch_size,1,1)))
    grouped_xyz = tf.reshape(xyz, (batch_size, 1, nsample, 3)) # (batch_size, npoint=1, nsample, 3)
    if points is not None:
        if use_xyz:
            new_points = tf.concat([xyz, points], axis=2) # (batch_size, 16, 259)
        else:
            new_points = points
        new_points = tf.expand_dims(new_points, 1) # (batch_size, 1, 16, 259)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points, idx, grouped_xyz

def PCA_decompose(x):
    '''
        x:(batch_size, ndataset, 3) TF tensor
    '''
    with tf.name_scope("PCA"):
        # nsmaple: number of neighboring points , n: feature channel
        nsample,n= tf.to_float(x.get_shape()[-2]),tf.to_int32(x.get_shape()[-1])
        mean = tf.reduce_mean(x,axis=-2)
        x_new = x - tf.tile(tf.expand_dims(mean, -2), [1, 1, nsample, 1])
        cov = tf.matmul(x_new,x_new,transpose_a=True)/(nsample - 1)
        e,v = tf.linalg.eigh(cov,name="eigh")
        x_pca=tf.matmul(x,v)
    return e,x_pca

def pointnet_sa_module(xyz, points, xyz_feature,npoint, radius, nsample, mlp, mlp2,mlp3, group_all, is_training, bn_decay, scope, bn=True, pooling='max', tnet_spec=None, knn=False,
                       use_xyz=False,end=False,use_xyz_feature=False):
    ''' PointNet Set Abstraction (SA) Module
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius: float32 -- search radius in local region
            nsample: int32 -- how many points in each local region
            mlp: list of int32 -- output size for MLP on each point
            mlp2: list of int32 -- output size for MLP on each region
            group_all: bool -- group all points into one PC if set true, OVERRIDE
                npoint, radius and nsample settings
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
            idx: (batch_size, npoint, nsample) int32 -- indices for local regions
    '''
    with tf.variable_scope(scope) as sc:
        batch_size = xyz.get_shape()[0].value
        if group_all:
            nsample = xyz.get_shape()[1].value
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_all(xyz, points, use_xyz)
        idx=tf.cast(idx,tf.int32)
        ########### additional xyz features
        if use_xyz_feature:
            xyz_feature = group_point(xyz_feature, idx)

            e, grouped_xyz = PCA_decompose(grouped_xyz)
            e = tf.tile(tf.expand_dims(e, 2), [1, 1, nsample, 1])
            edge_feature = tf.concat([relative_pos_encoding(tf.abs(grouped_xyz)), e], axis=-1)

            edge_feature = tf_util.conv2d(edge_feature, mlp3[0], [1, 1],
                                          padding='VALID', stride=[1, 1],
                                          bn=bn, is_training=is_training,
                                          scope='xyz_feature_%d' % (0), bn_decay=bn_decay)

            edge_feature = tf_util.conv2d(edge_feature, mlp3[1], [1, 1],
                                          padding='VALID', stride=[1, 1],
                                          bn=bn, is_training=is_training,
                                          scope='xyz_feature_%d' % (1), bn_decay=bn_decay)

            output_feature = tf.concat([xyz_feature, edge_feature], axis=-1)
            if end == False:
                xyz_feature = tf_util.conv2d(output_feature, mlp3[-1], [1, 1],
                                             padding='VALID', stride=[1, 1],
                                             bn=bn, is_training=is_training,
                                             scope='xyz_feature2', bn_decay=bn_decay)
                # we can try sum and mean
                xyz_feature = tf.reduce_max(xyz_feature, axis=[2], keep_dims=True, name='maxpool')
                xyz_feature = tf.squeeze(xyz_feature, [2])

            new_points = tf.concat([new_points, output_feature], axis=-1)

        for i, num_out_channel in enumerate(mlp):
            new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                        padding='SAME', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv%d'%(i), bn_decay=bn_decay)

        if pooling=='avg':
            new_points = tf_util.avg_pool2d(new_points, [1,nsample], stride=[1,1], padding='VALID', scope='avgpool1')
        elif pooling=='weighted_avg':
            with tf.variable_scope('weighted_avg1'):
                dists = tf.norm(grouped_xyz,axis=-1,ord=2,keep_dims=True)
                exp_dists = tf.exp(-dists * 5)
                weights = exp_dists/tf.reduce_sum(exp_dists,axis=2,keep_dims=True) # (batch_size, npoint, nsample, 1)
                new_points *= weights # (batch_size, npoint, nsample, mlp[-1])
                new_points = tf.reduce_sum(new_points, axis=2, keep_dims=True)
        elif pooling=='max':
            new_points = tf.reduce_max(new_points, axis=[2], keep_dims=True)
        elif pooling=='min':
            new_points = tf_util.max_pool2d(-1*new_points, [1,nsample], stride=[1,1], padding='VALID', scope='minpool1')
        elif pooling=='max_and_avg':
            avg_points = tf_util.max_pool2d(new_points, [1,nsample], stride=[1,1], padding='VALID', scope='maxpool1')
            max_points = tf_util.avg_pool2d(new_points, [1,nsample], stride=[1,1], padding='VALID', scope='avgpool1')
            new_points = tf.concat([avg_points, max_points], axis=-1)

        if mlp2 is None: mlp2 = []
        for i, num_out_channel in enumerate(mlp2):
            new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv_post_%d'%(i), bn_decay=bn_decay)
        new_points = tf.squeeze(new_points, [2]) # (batch_size, npoints, mlp2[-1])
        return new_xyz, new_points, idx,xyz_feature


def lfnet_module(kernel,scale,interp,fit,xyz, points, normals,axis_x,axis_y, xyz_feature,npoint, radius_list, nsample_list, mlp_list, is_training, bn_decay, scope,
                      mlp=[64,64],bn=True, use_xyz=False,weight=None,knn=0,d=1,end=False,use_xyz_feature=True,first_layer=False):
    ''' A-CNN module with rings
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            normals: (batch_size, ndataset, 3) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius_list: list of float32 -- search radiuses (inner and outer) represent ring in local region
            nsample_list: list of int32 -- how many points in each local region
            mlp: list of list of int32 -- output size for MLP on each point
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, \sum_k{mlp[k][-1]}) TF tensor
    '''
    # data_format = 'NCHW' if use_nchw else 'NHWC'
    with tf.variable_scope(scope) as sc:
        # if npoint == xyz.get_shape().as_list()[1] and knn==0:
        #     raise Exception('wrong input knn and npoint')
        if npoint != xyz.get_shape().as_list()[1]:
            indecies=farthest_point_sample(npoint, xyz)
            new_xyz = gather_point(xyz, indecies) # (batch_size, npoint, 3)
            new_normals = gather_point(normals, indecies) # (batch_size, npoint, 3)
            new_axis_x=gather_point(axis_x, indecies)
            new_axis_y=gather_point(axis_y, indecies)
        elif knn:
            new_xyz = xyz
            new_normals = normals
            new_axis_x =axis_x
            new_axis_y=axis_y
        else:
            indecies = tf.range(npoint)
            indecies = tf.tile(tf.expand_dims(indecies, 0), [xyz.get_shape().as_list()[0], 1])
            new_xyz = xyz
            new_normals = normals
            new_axis_x =axis_x
            new_axis_y=axis_y

        batch_size = xyz.get_shape()[0].value
        new_points_list = []

        for i in range(len(nsample_list)):
            radius = radius_list[i]
            print(radius)
            nsample = nsample_list[i]
            nk=kernel.get_shape().as_list()[0]
            kernel = kernel
            sita = scale
            if knn==1:
                radius=0

            _, idx = knn_point(nsample, xyz, new_xyz,d=d[i])

            grouped_xyz = group_point(xyz, idx)
            grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1, 1, nsample, 1])

            if weight is None:
                _, proj, _, kernel_out, weight, kernel_fit = transform_neighbors(nsample, idx, xyz, new_xyz, new_normals,new_axis_x,new_axis_y,
                                                                          kernel,sita, interp, fit,radius)
                proj=relative_pos_encoding(proj)
                if interp != 2:
                    # weight=tf.nn.softmax(weight,axis=-2)
                    weight = weight / tf.reduce_sum(weight, axis=-2, keep_dims=True)
                weight = tf.expand_dims(weight, 3)
            if points is not None:
                grouped_points = group_point(points, idx)
                if use_xyz:
                    grouped_points = tf.concat([grouped_points, grouped_xyz], axis=-1)
            else:
                grouped_points = proj
            ########### addition xyz features
            if use_xyz_feature:
                if xyz_feature is None:
                    xyz_feature = proj
                else:
                    xyz_feature = group_point(xyz_feature, idx)
                edge_feature = proj

                edge_feature = tf_util.conv2d(edge_feature, mlp[0], [1, 1],
                                              padding='VALID', stride=[1, 1],
                                              bn=bn, is_training=is_training,
                                              scope='xyz_feature_%d' % (0), bn_decay=bn_decay)
                edge_feature = tf_util.conv2d(edge_feature, mlp[0], [1, 1],
                                              padding='VALID', stride=[1, 1],
                                              bn=bn, is_training=is_training,
                                              scope='xyz_feature_%d' % (1), bn_decay=bn_decay)
                output_feature = tf.concat([xyz_feature, edge_feature], axis=-1)
                if end == False:
                    xyz_feature = tf_util.conv2d(output_feature, mlp[-1], [1, 1],
                                                 padding='VALID', stride=[1, 1],
                                                 bn=bn, is_training=is_training,
                                                 scope='xyz_feature2', bn_decay=bn_decay)
                    # we can try sum and mean
                    xyz_feature = tf.reduce_max(xyz_feature, axis=[2], keep_dims=True, name='maxpool')
                    xyz_feature = tf.squeeze(xyz_feature, [2])
            if use_xyz_feature:
                grouped_points = tf.concat([grouped_points, output_feature], axis=-1)
            #ASFConv,加一下for
            if first_layer:
                grouped_points=tf_util.conv2d(grouped_points, mlp_list[i][0], [1, 1],
                                    padding='VALID', stride=[1, 1], bn=bn, is_training=is_training,
                                    scope='conv%d_%d' % (i, 0), bn_decay=bn_decay)
            # Discrete Conv
            new_points = DiscreteConv(grouped_points, mlp_list , bn, i, is_training, bn_decay, weight, nk,
                                     kernel_fit)
            new_points_list.append(new_points)
        new_points = tf.concat(new_points_list, axis=-1)

        if first_layer:
            return new_xyz, new_points, new_normals,new_axis_x,new_axis_y, kernel_out, weight,kernel_fit,xyz_feature
        else:
            return new_xyz, new_points, new_normals,new_axis_x,new_axis_y,_, weight, _,xyz_feature

def relative_pos_encoding( relative_xyz):
        relative_dis = tf.sqrt(tf.reduce_sum(tf.square(relative_xyz), axis=-1, keepdims=True))
        nn_pts_local_proj_dot = tf.divide(relative_xyz, relative_dis)
        nn_pts_local_proj_dot = tf.where(tf.is_nan(nn_pts_local_proj_dot), tf.ones_like(nn_pts_local_proj_dot) * 0,
                                         nn_pts_local_proj_dot)  # check nan

        relative_feature = tf.concat([relative_dis, relative_xyz, nn_pts_local_proj_dot], axis=-1)
        return relative_feature

def DiscreteConv(grouped_points, mlp_list, bn, i, is_training, bn_decay, weight, nk, kernel_fit):
    grouped_points = tf.reduce_sum(tf.expand_dims(grouped_points, 4) * weight, axis=2)
    grouped_points = tf.transpose(grouped_points, [0, 1, 3, 2])

    grouped_points = tf_util.conv2d(grouped_points, mlp_list[i][1], [1, nk],
                                    padding='VALID', stride=[1, 1], bn=bn, is_training=is_training,
                                    scope='conv%d_%d' % (i, 1), bn_decay=bn_decay)
    new_points = tf.squeeze(grouped_points, axis=2)
    new_points = tf_util.conv1d(new_points, mlp_list[i][2], 1, padding='VALID', stride=1, bn=bn,
                                is_training=is_training,
                                scope='conv%d_%d' % (i, 2), bn_decay=bn_decay)
    return new_points

def pointnet_fp_module(xyz1, xyz2, points1, points2,xyz_feature, mlp, is_training, bn_decay, scope, bn=True):
    ''' PointNet Feature Propogation (FP) Module
        Input:
            xyz1: (batch_size, ndataset1, 3) TF tensor
            xyz2: (batch_size, ndataset2, 3) TF tensor, sparser than xyz1
            points1: (batch_size, ndataset1, nchannel1) TF tensor
            points2: (batch_size, ndataset2, nchannel2) TF tensor
            mlp: list of int32 -- output size for MLP on each point
        Return:
            new_points: (batch_size, ndataset1, mlp[-1]) TF tensor
    '''
    with tf.variable_scope(scope) as sc:
        dist, idx = three_nn(xyz1, xyz2)
        dist = tf.maximum(dist, 1e-10)
        norm = tf.reduce_sum((1.0/dist),axis=2,keep_dims=True)
        norm = tf.tile(norm,[1,1,3])
        weight = (1.0/dist) / norm
        interpolated_points = three_interpolate(points2, idx, weight)
        if xyz_feature is not None:
            interpolated_xyz_feature=three_interpolate(xyz_feature, idx, weight)

        if points1 is not None:
            if xyz_feature is not None:
                new_points1 = tf.concat(axis=2, values=[interpolated_points,interpolated_xyz_feature, points1]) # B,ndataset1,nchannel1+nchannel2
            else:
                new_points1 = tf.concat(axis=2, values=[interpolated_points, points1])
        else:
            if xyz_feature is not None:
                new_points1 = tf.concat([interpolated_points, interpolated_xyz_feature], axis=2)
            else:
                new_points1 = tf.concat([interpolated_points], axis=2)
        new_points1 = tf.expand_dims(new_points1, 2)
        for i, num_out_channel in enumerate(mlp):
            new_points1 = tf_util.conv2d(new_points1, num_out_channel, [1,1],
                                         padding='VALID', stride=[1,1],
                                         bn=bn, is_training=is_training,
                                         scope='conv_%d'%(i), bn_decay=bn_decay)
        new_points1 = tf.squeeze(new_points1, [2]) # B,ndataset1,mlp[-1]
        return new_points1
