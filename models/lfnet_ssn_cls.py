import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tensorflow as tf
import numpy as np
import tf_util
from utils.pointnet_util import pointnet_sa_module, lfnet_module

def placeholder_inputs(batch_size, num_point,n,d):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    normals_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    axis_x = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    axis_y = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    kernel=tf.placeholder(tf.float32, shape=(n,d))
    return pointclouds_pl, labels_pl, normals_pl,axis_x,axis_y,kernel

def get_model(point_cloud, normals,axis_x,axis_y,kernel, scale,interp,fit,is_training, bn_decay=None,d=1,knn=1,nsample=16,use_xyz_feature=True):
    """ Classification A-CNN, input is points BxNx3 and normals BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}

    l0_xyz = point_cloud
    l0_normals = normals
    l0_axis_x=axis_x
    l0_axis_y=axis_y
    l0_points = None
    xyz_feature=None

    l1_xyz, l1_points, l1_normals,l1_axis_x,l1_axis_y, kernel_out, weight, kernel_fit,xyz_feature = lfnet_module(kernel, scale,interp,fit,l0_xyz, l0_points,
                                                                                      l0_normals,l0_axis_x,l0_axis_y, xyz_feature,512,
                                                                                      [0.23],
                                                                                      nsample[0],
                                                                                      [[32, 64, 128]],
                                                                                      is_training, bn_decay,mlp=[64,64],first_layer=True,
                                                                                      scope='layer1',d=d,knn=knn,use_xyz_feature=use_xyz_feature)
    l2_xyz, l2_points, l2_normals,l2_axis_x,l2_axis_y, _, _, _ ,xyz_feature= lfnet_module(kernel, scale,interp,fit,l1_xyz, l1_points, l1_normals,l1_axis_x,l1_axis_y, xyz_feature,
                                                                  128,[0.32], nsample[1],
                                                               [[128,128,256]], is_training, bn_decay,
                                                               mlp=[64,64],scope='layer2',d=d,knn=knn,use_xyz_feature=use_xyz_feature)
    _, l5_points, _,_ = pointnet_sa_module(l2_xyz, l2_points, xyz_feature,npoint=None, radius=None, nsample=None, mlp=[1024], mlp2=None,
                                         mlp3=[64,64], group_all=True, is_training=is_training, bn_decay=bn_decay, scope='layer3',end=True,use_xyz_feature=use_xyz_feature)

    # Fully connected layers
    net = tf.reshape(l5_points, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.4, is_training=is_training, scope='dp1')
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.4, is_training=is_training, scope='dp2')
    net = tf_util.fully_connected(net, 40, activation_fn=None, scope='fc3')

    return net, end_points,kernel_out,weight,kernel_fit


def get_loss(pred, label, end_points):
    """ pred: B*NUM_CLASSES,
        label: B, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)
    return classify_loss


if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024,3))
        net, _ = get_model(inputs, tf.constant(True))
        print(net)