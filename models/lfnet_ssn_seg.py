import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tensorflow as tf
import numpy as np
import tf_util
from utils.pointnet_util import pointnet_sa_module, pointnet_fp_module, lfnet_module

def placeholder_inputs(batch_size, num_point,n):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    normals_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    cls_labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    kernel = tf.placeholder(tf.float32, shape=(n, 3))
    axis_x = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    axis_y = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    return pointclouds_pl, labels_pl, normals_pl, cls_labels_pl,axis_x,axis_y,kernel

NUM_CATEGORIES = 16

def get_model(point_cloud, cls_label, normals,axis_x,axis_y, kernel, scale,interp,fit,is_training,classes=50, bn_decay=None,d=1,knn=1,nsample=16,use_xyz_feature=True):
    """ Part segmentation A-CNN, input is points BxNx3 and normals BxNx3, output Bx50 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}

    l0_xyz = point_cloud
    l0_normals = normals
    l0_axis_x = axis_x
    l0_axis_y = axis_y
    l0_points = None #用了法向量作为输入
    xyz_feature=None

    l1_xyz, l1_points, l1_normals,l1_axis_x,l1_axis_y, kernel_out, weight, kernel_fit,xyz_feature1 = lfnet_module(kernel, scale,interp,fit,l0_xyz, l0_points,
                                                                                      l0_normals,l0_axis_x,l0_axis_y, xyz_feature,1024,
                                                                                      [0.1],
                                                                                      nsample[0],
                                                                                      [ [16, 16, 32],[64,96,128]],
                                                                                      is_training, bn_decay,mlp=[16,16,32],first_layer=True,
                                                                                      scope='layer1',d=d,knn=knn,use_xyz_feature=use_xyz_feature)
    l2_xyz, l2_points, l2_normals,l2_axis_x,l2_axis_y, _, _, _ ,xyz_feature2= lfnet_module(kernel, scale,interp,fit,l1_xyz, l1_points, l1_normals,l1_axis_x,l1_axis_y,
                                                             xyz_feature1,256,[0.2], nsample[1],
                                                               [[32,32,64], [128,128,256]], is_training, bn_decay,
                                                               mlp=[32,32,64],scope='layer2',d=d,knn=knn,use_xyz_feature=use_xyz_feature)
    l3_xyz, l3_points, l3_normals, l3_axis_x, l3_axis_y, _, _, _, xyz_feature3 = lfnet_module(kernel, scale,interp, fit, l2_xyz,l2_points,l2_normals,l2_axis_x, l2_axis_y,
                                                                                                   xyz_feature2, 64,[0.4], nsample[1],
                                                                                                   [[64, 64, 128],  [128, 128, 256]],
                                                                                                   is_training, bn_decay,  mlp=[64, 64,64], scope='layer3', d=d,
                                                                                                   knn=knn, use_xyz_feature=use_xyz_feature)
    l4_xyz, l4_points, l4_normals, l4_axis_x, l4_axis_y, _, _, _, xyz_feature4 = lfnet_module(kernel, scale, interp, fit, l3_xyz,l3_points,l3_normals,l3_axis_x, l3_axis_y,
                                                                                                   xyz_feature3, 16,[0.8], nsample[1],
                                                                                                   [[128, 128, 256], [128, 128, 256]], is_training,
                                                                                                   bn_decay,mlp=[64, 64,64], scope='layer4', d=d,
                                                                                                   knn=knn, use_xyz_feature=use_xyz_feature)
    l4_points=tf_util.conv1d(l4_points, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)

    # Feature Propagation layers
    l3_points = pointnet_fp_module(l3_xyz, l4_xyz, l3_points, l4_points, xyz_feature4, [256, 128], is_training, bn_decay,scope='fa_layer1_up')
    l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, xyz_feature3, [128, 128], is_training, bn_decay, scope='fa_layer2_up')
    l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points, xyz_feature2, [128, 128], is_training, bn_decay, scope='fa_layer3_up')
    l0_points = pointnet_fp_module(l0_xyz, l1_xyz, l0_points, l1_points,xyz_feature1, [128, 128], is_training, bn_decay, scope='fa_layer4')

    l4_feature = tf.tile(tf.reduce_max(l4_points,axis=1,keep_dims=True), [1, num_point, 1])
    concat=tf.concat([l0_points,l4_feature],axis=-1)

    # FC layers
    net = tf_util.conv1d(concat, 256, 1, padding='VALID', bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.6, is_training=is_training, scope='dp2')
    net = tf_util.conv1d(net, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc3', bn_decay=bn_decay)
    net = tf_util.conv1d(net, classes, 1, padding='VALID', activation_fn=None, scope='fc4')
    return net, end_points,kernel_out,weight,kernel_fit


def get_loss(pred, label):
    """ pred: BxNxC,
        label: BxN, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    # tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)
    return classify_loss

if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,2048,6))
        net, _ = get_model(inputs, tf.constant(True))
        print(net)

