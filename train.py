import argparse
import math
from datetime import datetime
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.dirname(ROOT_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider
import tf_util
import modelnet_dataset
import modelnet_dataset_rotate
import modelnet_h5_dataset
import yaml
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='cfgs/config_ssn_cls.yaml', type=str)
args = parser.parse_args()

with open(args.config) as f:
    config = yaml.load(f)
print("\n**************************")
for k, v in config['common'].items():
    setattr(args, k, v)
    print('\n[%s]:'%(k), v)
print("\n**************************\n")



BATCH_SIZE = args.batch_size
NUM_POINT = args.num_point
MAX_EPOCH = args.max_epoch
BASE_LEARNING_RATE = args.learning_rate
GPU_INDEX = args.gpu
MOMENTUM = args.momentum
OPTIMIZER = args.optimizer
DECAY_STEP = args.decay_step
DECAY_RATE = args.decay_rate


BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()
global_acc=0.90

NUM_CLASSES = 40

# Shapenet official train/test split
if args.normal:
    assert(NUM_POINT<=10000)
    DATA_PATH = args.data_path
    TRAIN_DATASET = modelnet_dataset_rotate.ModelNetDataset(root=DATA_PATH, npoints=NUM_POINT, split='train', normal_channel=args.normal, batch_size=BATCH_SIZE)
    TEST_DATASET = modelnet_dataset_rotate.ModelNetDataset(root=DATA_PATH, npoints=NUM_POINT, split='test', normal_channel=args.normal, batch_size=BATCH_SIZE)
    # TRAIN_DATASET = modelnet_h5_dataset.ModelNetH5Dataset('/home/inshallah/Documents/data/modelnet40_ply_hdf5_2048/train_files.txt', batch_size=BATCH_SIZE,
    #     npoints=NUM_POINT, shuffle=True)
    # TEST_DATASET = modelnet_h5_dataset.ModelNetH5Dataset('/home/inshallah/Documents/data/modelnet40_ply_hdf5_2048/test_files.txt', batch_size=BATCH_SIZE,
    #     npoints=NUM_POINT, shuffle=False)

def log_string(LOG_FOUT,out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train(kernel_init,LOG_DIR,LOG_FOUT, acc_out):
    MODEL = importlib.import_module(args.model)  # import network module
    MODEL_FILE = os.path.join(ROOT_DIR, 'models', args.model + '.py')
    os.system('cp %s %s' % (MODEL_FILE, LOG_DIR))
    os.system('cp train.py %s' % (LOG_DIR))  # bkp of train procedure
    os.system('cp tf_ops/transforming/tf_transforming.py %s' % (LOG_DIR))
    os.system('cp tf_ops/transforming/tf_transforming.cpp %s' % (LOG_DIR))
    os.system('cp tf_ops/transforming/tf_transforming_g.cu %s' % (LOG_DIR))
    os.system('cp models/lfnet_ssn_cls.py %s' % (LOG_DIR))
    os.system('cp utils/pointnet_util.py %s' % (LOG_DIR))
    os.system('cp modelnet_h5_dataset.py %s' % (LOG_DIR))
    LOG_FOUT.write(str(args) + '\n')
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl, normals_pl,axis_x,axis_y,kernel = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT,kernel_init.shape[0],kernel_init.shape[1])
            is_training_pl = tf.placeholder(tf.bool, shape=())

            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to helpfully increment the 'batch' parameter
            # for you every time it trains.
            batch = tf.get_variable('batch', [],
                                    initializer=tf.constant_initializer(0), trainable=False)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss
            pred, end_points, kernel_out, weight, kernel_fit = MODEL.get_model(pointclouds_pl, normals_pl,axis_x,axis_y, kernel, args.scale, args.interp, args.fit,
                                                                               is_training_pl,bn_decay=bn_decay,d=args.d,knn=args.knn,nsample=args.nsample,
                                                                               use_xyz_feature=args.use_xyz_feature)
            MODEL.get_loss(pred, labels_pl, end_points)
            losses = tf.get_collection('losses')
            total_loss = tf.add_n(losses, name='total_loss')
            tf.summary.scalar('total_loss', total_loss)
            for l in losses + [total_loss]:
                tf.summary.scalar(l.op.name, l)

            correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
            tf.summary.scalar('accuracy', accuracy)

            print("--- Get training operator")
            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(total_loss, global_step=batch)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

            # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        if args.save:
            train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
            test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)
        else:
            train_writer = None
            test_writer = None

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init)

        ops = {'pointclouds_pl': pointclouds_pl,
               'normals_pl': normals_pl,
               'axis_x': axis_x,
               'axis_y':axis_y,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': total_loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'end_points': end_points,
               'kernel': kernel,
               'kernel_out': kernel_out,
               'weight': weight,
               'kernel_fit': kernel_fit}

        best_acc = -1
        best_acc_avg=-1
        print(kernel_init)
        test_acc = np.zeros((MAX_EPOCH + 2, 1))

        for epoch in range(MAX_EPOCH):
            log_string(LOG_FOUT, '**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()

            train_one_epoch(LOG_FOUT, sess, ops, train_writer, kernel_init)
            acc, acc_avg = eval_one_epoch(LOG_FOUT, sess, ops, test_writer, kernel_init)
            test_acc[epoch] = acc
            log_string(acc_out, '%f' % (acc))

            if acc > best_acc:
                best_acc = acc
            if acc_avg > best_acc_avg:
                best_acc_avg = acc_avg
            if acc>global_acc:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model_iter_%d_acc_%0.6f_category.ckpt"%(epoch,acc)))
                log_string(LOG_FOUT, "Model saved in file: %s" % save_path)
            log_string(LOG_FOUT, "Best category accuracy: %f" % best_acc)

            # Save the variables to disk.
            if args.save_ckpt:
                if epoch % 10 == 0:
                    save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                    log_string(LOG_FOUT, "Model saved in file: %s" % save_path)

        print('______________________________________________________________________')
        print("Best category accuracy: %f" % best_acc)
        print("Best instance accuracy: %f" % best_acc_avg)

    test_acc[MAX_EPOCH] = best_acc
    test_acc[MAX_EPOCH + 1] = best_acc_avg
    log_string(acc_out, '%f' % (best_acc))
    log_string(acc_out, '%f' % (best_acc_avg))
    return test_acc


def train_one_epoch(LOG_FOUT, sess, ops, train_writer, kernel_init):
    """ ops: dict mapping from string to tf ops """
    is_training = True

    log_string(LOG_FOUT, str(datetime.now()))

    # Make sure batch data is of same size
    cur_batch_data = np.zeros((BATCH_SIZE, NUM_POINT, 3))
    cur_batch_normals = np.zeros((BATCH_SIZE, NUM_POINT, 3))
    cur_batch_axis_x= np.zeros((BATCH_SIZE, NUM_POINT, 3))
    cur_batch_axis_y = np.zeros((BATCH_SIZE, NUM_POINT, 3))
    cur_batch_label = np.zeros((BATCH_SIZE), dtype=np.int32)

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    batch_idx = 0
    while TRAIN_DATASET.has_next_batch():
        batch_data, batch_label = TRAIN_DATASET.next_batch(augment=True)

        bsize = batch_data.shape[0]
        #
        num_points = 1200
        batch_data=batch_data[:,:num_points,:]
        batch_data=batch_data[:, np.random.choice(num_points, args.num_point, False),:]
        axis_x = np.cross(batch_data[:, :, :3], batch_data[:, :, 3:])
        if args.norm_pi:
            axis_x = axis_x / np.sqrt(np.sum(axis_x ** 2, axis=-1))[:, :, np.newaxis]
        axis_y = np.cross(axis_x, batch_data[:, :, 3:])
        cur_batch_data[0:bsize, ...] = batch_data[:, :, :3]
        cur_batch_normals[0:bsize, ...] = batch_data[:, :, 3:]
        cur_batch_axis_x[0:bsize, ...] = axis_x
        cur_batch_axis_y[0:bsize, ...] = axis_y


        cur_batch_label[0:bsize] = batch_label

        feed_dict = {ops['pointclouds_pl']: cur_batch_data,
                     ops['labels_pl']: cur_batch_label,
                     ops['normals_pl']: cur_batch_normals,
                     ops['axis_x']: cur_batch_axis_x,
                     ops['axis_y']: cur_batch_axis_y,
                     ops['is_training_pl']: is_training,
                     ops['kernel']: kernel_init}
        if args.save:
            summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                                                             ops['train_op'], ops['loss'], ops['pred']],
                                                            feed_dict=feed_dict)
            train_writer.add_summary(summary, step)
        else:
            step, _, loss_val, pred_val = sess.run([ops['step'], ops['train_op'], ops['loss'], ops['pred']],
                                                   feed_dict=feed_dict)
        pred_val = np.argmax(pred_val, 1)
        correct = np.sum(pred_val[0:bsize] == batch_label[0:bsize])
        total_correct += correct
        total_seen += bsize
        loss_sum += loss_val

        # k_out,w,k_fit=sess.run([ops['kernel_out'],ops['weight'],ops['kernel_fit']],feed_dict=feed_dict)

        if (batch_idx + 1) % 50 == 0:
            log_string(LOG_FOUT, ' ---- batch: %03d ----' % (batch_idx + 1))
            log_string(LOG_FOUT, 'mean loss: %f' % (loss_sum / 50))
            log_string(LOG_FOUT, 'accuracy: %f' % (total_correct / float(total_seen)))
            total_correct = 0
            total_seen = 0
            loss_sum = 0
        batch_idx += 1

    TRAIN_DATASET.reset()


def eval_one_epoch(LOG_FOUT, sess, ops, test_writer, kernel_init):
    """ ops: dict mapping from string to tf ops """
    global EPOCH_CNT
    is_training = False

    # Make sure batch data is of same size
    cur_batch_data = np.zeros((BATCH_SIZE, NUM_POINT, 3))
    cur_batch_normals = np.zeros((BATCH_SIZE, NUM_POINT, 3))
    cur_batch_label = np.zeros((BATCH_SIZE), dtype=np.int32)
    cur_batch_axis_x = np.zeros((BATCH_SIZE, NUM_POINT, 3))
    cur_batch_axis_y = np.zeros((BATCH_SIZE, NUM_POINT, 3))

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    batch_idx = 0
    shape_ious = []
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]

    log_string(LOG_FOUT, str(datetime.now()))
    log_string(LOG_FOUT, '---- EPOCH %03d EVALUATION ----' % (EPOCH_CNT))

    while TEST_DATASET.has_next_batch():
        batch_data, batch_label = TEST_DATASET.next_batch(augment=False,rotate=args.rotate)
        bsize = batch_data.shape[0]
        #
        num_points = 1200
        batch_data=batch_data[:,:num_points,:]
        batch_data=batch_data[:, np.random.choice(num_points, args.num_point, False),:]
        axis_x=np.cross(batch_data[:,:,:3],batch_data[:,:,3:])
        if args.norm_pi:
            axis_x = axis_x / np.sqrt(np.sum(axis_x ** 2, axis=-1))[:, :, np.newaxis]
        axis_y=np.cross(axis_x,batch_data[:,:,3:])
        cur_batch_data[0:bsize, ...] = batch_data[:, :, :3]
        cur_batch_normals[0:bsize, ...] = batch_data[:, :, 3:]
        cur_batch_axis_x[0:bsize, ...] = axis_x
        cur_batch_axis_y[0:bsize, ...] = axis_y
        cur_batch_label[0:bsize] = batch_label

        feed_dict = {ops['pointclouds_pl']: cur_batch_data,
                     ops['labels_pl']: cur_batch_label,
                     ops['axis_x']: cur_batch_axis_x,
                     ops['axis_y']: cur_batch_axis_y,
                     ops['normals_pl']: cur_batch_normals,
                     ops['is_training_pl']: is_training,
                     ops['kernel']: kernel_init}
        if args.save:
            summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                                                          ops['loss'], ops['pred']], feed_dict=feed_dict)
            test_writer.add_summary(summary, step)
        else:
            step, loss_val, pred_val = sess.run([ops['step'], ops['loss'], ops['pred']], feed_dict=feed_dict)
        pred_val = np.argmax(pred_val, 1)
        correct = np.sum(pred_val[0:bsize] == batch_label[0:bsize])
        total_correct += correct
        total_seen += bsize
        loss_sum += loss_val
        batch_idx += 1
        for i in range(0, bsize):
            l = batch_label[i]
            total_seen_class[l] += 1
            total_correct_class[l] += (pred_val[i] == l)

    log_string(LOG_FOUT, 'eval mean loss: %f' % (loss_sum / float(batch_idx)))
    log_string(LOG_FOUT, 'eval accuracy: %f' % (total_correct / float(total_seen)))
    log_string(LOG_FOUT, 'eval avg class acc: %f' % (
        np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))))
    EPOCH_CNT += 1

    TEST_DATASET.reset()
    return total_correct / float(total_seen), np.mean(
        np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))


if __name__ == "__main__":
    nk = 10
    kernel = [[-7.53131478e-03, -1.11457535e-02, 1.43582161e-02],
               [4.69053978e-01, 7.71612529e-02, -8.69379288e-01],
               [-1.41868369e-01, -6.85753662e-01, 6.97777964e-01],
               [-5.25251239e-01, -5.88565834e-01, -6.15829338e-01],
               [-1.58158612e-01, 5.51346468e-01, 8.07008697e-01],
               [-5.26633482e-01, 6.69274283e-01, -5.13406609e-01],
               [5.01444853e-01, 8.60073497e-01, -8.58032089e-02],
               [8.45904744e-01, -1.97249945e-02, 5.07576565e-01],
               [-9.72054017e-01, -4.18486464e-02, 2.50755044e-01],
               [5.38774332e-01, -8.45835742e-01, -2.14561211e-01]]

    theta = np.linspace(0, 2 * np.pi, 11)
    np.random.seed(123)
    kernel_init_3D = np.array(kernel) * 2/3

    # depict the distribution of kernel points which obtained by KPConv
    if args.show:
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(kernel_init_3D[:, 0], kernel_init_3D[:, 1], kernel_init_3D[:, 2], color='red')
        plt.show()

    for i in [0]:
        print('********************************************************')
        print('kernel %d' % (i))
        EPOCH_CNT = 0
        LOG_DIR = args.log_dir
        if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)

        LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
        acc_out = open(os.path.join(LOG_DIR, 'test_acc.txt'), 'w')
        log_string(LOG_FOUT, 'pid: %s' % (str(os.getpid())))

        LOG_FOUT.write(str(kernel_init_3D[i, :, :]) + '\n')
        test_acc = train(kernel_init_3D[i, :, :], LOG_DIR, LOG_FOUT,acc_out)
        LOG_FOUT.close()
        acc_out.close()
        test_acc = np.loadtxt(LOG_DIR+'/test_acc.txt')
        plt.plot(test_acc)
        plt.show()


