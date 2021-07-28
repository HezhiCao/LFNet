import tensorflow as tf
import numpy as np
import argparse
import socket
import importlib
import time
import os
# import scipy.misc
import sys
import tqdm
from tensorflow.contrib.tensorboard.plugins import projector
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.dirname(ROOT_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider
import modelnet_dataset as modelnet_dataset
import modelnet_h5_dataset
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='cfgs/config_ssn_cls.yaml', type=str)
parser.add_argument('--repeat_num',default=1,type=int, help='repeat_num of voting')
parser.add_argument('--dump_dir',default='dump',type=str, help='normalize axis or not[default: 0]')
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
#MODEL_PATH = args.model_path
GPU_INDEX = args.gpu
MODEL = importlib.import_module(args.model) # import network module
DUMP_DIR = args.dump_dir
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)


NUM_CLASSES = 40
SHAPE_NAMES = [line.rstrip() for line in \
    open('/home/inshallah/Documents/data/modelnet40_normal_resampled/modelnet40_shape_names.txt')]

HOSTNAME = socket.gethostname()

# Shapenet official train/test split
if args.normal:
    assert(NUM_POINT<=10000)
    DATA_PATH = '/home/inshallah/Documents/data/modelnet40_normal_resampled'
    TRAIN_DATASET = modelnet_dataset.ModelNetDataset(root=DATA_PATH, npoints=NUM_POINT, split='train', normal_channel=args.normal, batch_size=BATCH_SIZE)
    TEST_DATASET = modelnet_dataset.ModelNetDataset(root=DATA_PATH, npoints=NUM_POINT, split='test', normal_channel=args.normal, batch_size=BATCH_SIZE)

    #h5_file_origin
    # DATA_PATH = '/home/inshallah/Documents/data/modelnet40_ply_hdf5_2048_origin'
    # TEST_DATASET = modelnet_h5_dataset.ModelNetH5Dataset(os.path.join(DATA_PATH,'test_files.txt'), batch_size = BATCH_SIZE, npoints = NUM_POINT, shuffle=False)
    # estimated
    # DATA_PATH ='/media/inshallah/inshallah/datasets/'
    # TEST_DATASET = modelnet_h5_dataset.ModelNetH5Dataset( os.path.join(DATA_PATH, 'modelnet40_ply_hdf5_2048_estimated/test_files.txt'), batch_size=BATCH_SIZE,npoints=NUM_POINT, shuffle=False)

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def evaluate(kernel_init):
    is_training = False

    with tf.device('/gpu:0'):
        #pointclouds_pl, labels_pl, normals_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
        pointclouds_pl, labels_pl, normals_pl,axis_x,axis_y,kernel = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT,kernel_init.shape[0],kernel_init.shape[1])
        is_training_pl = tf.placeholder(tf.bool, shape=())

        # simple model
        #pred, end_points = MODEL.get_model(pointclouds_pl, normals_pl, is_training_pl)
        pred, end_points, kernel_out, weight, kernel_fit =MODEL.get_model(pointclouds_pl, normals_pl, axis_x,axis_y,kernel,
                        args.scale, args.interp, 0, is_training_pl,
                         d=[1,2,4], knn=args.knn, nsample=[[48],[32]],use_xyz_feature=args.use_xyz_feature )
        MODEL.get_loss(pred, labels_pl, end_points)
        losses = tf.get_collection('losses')
        total_loss = tf.add_n(losses, name='total_loss')

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess = tf.Session(config=config)

    # Restore variables from disk.
    saver.restore(sess, args.model_path)
    log_string("Model restored.")

    ops = {'pointclouds_pl': pointclouds_pl,
           'normals_pl': normals_pl,
           'axis_x': axis_x,
           'axis_y': axis_y,
           'labels_pl': labels_pl,
           'is_training_pl': is_training_pl,
           'pred': pred,
           'loss': total_loss,
           'kernel': kernel}
    path_for_mnist_metadata = os.path.join('t_sne', 'meta.tsv')
    # f=open(path_for_mnist_metadata, 'w')
    for _ in range(args.repeat_num):
        eval_one_epoch(sess, ops,kernel_init,None, args.num_votes,args.rotate)

def eval_one_epoch(sess, ops, kernel_init,f,num_votes=1,rotate=0, topk=1):
    is_training = False

    # Make sure batch data is of same size
    cur_batch_data = np.zeros((BATCH_SIZE,NUM_POINT,3))
    cur_batch_normals = np.zeros((BATCH_SIZE,NUM_POINT,3))
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
    if f is not None:
        global_idx=0
        f.write("Index\tLabel\n")

    mm=0

    while TEST_DATASET.has_next_batch():
        batch_data, batch_label = TEST_DATASET.next_batch(augment=False,rotate=rotate)
        # labels = np.argmax(batch_label, 1)
        bsize = batch_data.shape[0]
        if f is not None:
            for _, label in enumerate(batch_label):
                f.write("%d\t%d\n" % (global_idx, label))
                global_idx+=1

        print('Batch: %03d, batch size: %d'%(batch_idx, bsize))

        batch_pred_sum = np.zeros((BATCH_SIZE, NUM_CLASSES)) # score for classes
        cur_batch_label[0:bsize] = batch_label

        for vote_idx in range(num_votes):
            num_points=2048
            original_data = np.copy(batch_data[:,:num_points,:])
            original_data=original_data[:,np.random.choice(original_data.shape[1], args.num_point, False),:]
            if vote_idx>0:
                jittered_data = provider.random_scale_point_cloud(original_data[:, :, 0:3])
            #     # jittered_data = provider.jitter_point_cloud(jittered_data[:,:,:3])
            else:
                # jittered_data = provider.jitter_point_cloud(original_data[:, :, :3])
                jittered_data=original_data[:,:,:3]
            # jittered_normal=provider.jitter_point_cloud(original_data[:,:,3:])
            original_data[:,:,:3] = jittered_data
            # original_data[:, :, 3:] = jittered_normal
            # original_data[:, :, :3]=pc_normalize(original_data[:,:,:3])
            # shuffled_data = provider.shuffle_points(original_data)
            shuffled_data = original_data
            axis_x = np.cross(shuffled_data[:, :, :3], shuffled_data[:, :, 3:])
            if args.norm_pi:
                axis_x = axis_x / np.sqrt(np.sum(axis_x ** 2, axis=-1))[:, :, np.newaxis]
            axis_y = np.cross(axis_x, shuffled_data[:, :, 3:])


            cur_batch_data[0:bsize,...] = shuffled_data[:,:,:3]
            cur_batch_normals[0:bsize,...] = shuffled_data[:,:,3:]
            cur_batch_axis_x[0:bsize, ...] = axis_x
            cur_batch_axis_y[0:bsize, ...] = axis_y


            feed_dict = {ops['pointclouds_pl']: cur_batch_data,
                         ops['labels_pl']: cur_batch_label,
                         ops['normals_pl']: cur_batch_normals,
                         ops['axis_x']: cur_batch_axis_x,
                         ops['axis_y']: cur_batch_axis_y,
                         ops['is_training_pl']: is_training,
                         ops['kernel']: kernel_init}
            loss_val, pred_val = sess.run([ops['loss'], ops['pred']], feed_dict=feed_dict)
            batch_pred_sum += pred_val


        pred_val = np.argmax(batch_pred_sum, 1)
        # np.savetxt('t_sne/txt1/eva%d.txt' % mm, pred_val)
        mm += 1
        # visualisation(batch_pred_sum)
        correct = np.sum(pred_val[0:bsize] == batch_label[0:bsize])
        total_correct += correct
        total_seen += bsize
        loss_sum += loss_val
        batch_idx += 1
        for i in range(bsize):
            l = batch_label[i]
            total_seen_class[l] += 1
            total_correct_class[l] += (pred_val[i] == l)

    log_string('eval mean loss: %f' % (loss_sum / float(batch_idx)))
    log_string('eval accuracy: %f'% (total_correct / float(total_seen)))
    log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))
    TEST_DATASET.reset()
    class_accuracies = np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float)
    for i, name in enumerate(SHAPE_NAMES):
        log_string('%10s:\t%0.3f' % (name, class_accuracies[i]))
def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=1)
    pc = pc - centroid[:,np.newaxis,:]
    a=np.power(pc,2)
    m = np.max(np.sqrt(np.sum(a, axis=2)),axis=1)
    pc = pc / m[:,np.newaxis,np.newaxis]
    return pc

if __name__=='__main__':
    n = 3
    nk = 10
    kernel1 = [[-7.53131478e-03, -1.11457535e-02, 1.43582161e-02],
               [4.69053978e-01, 7.71612529e-02, -8.69379288e-01],
               [-1.41868369e-01, -6.85753662e-01, 6.97777964e-01],
               [-5.25251239e-01, -5.88565834e-01, -6.15829338e-01],
               [-1.58158612e-01, 5.51346468e-01, 8.07008697e-01],
               [-5.26633482e-01, 6.69274283e-01, -5.13406609e-01],
               [5.01444853e-01, 8.60073497e-01, -8.58032089e-02],
               [8.45904744e-01, -1.97249945e-02, 5.07576565e-01],
               [-9.72054017e-01, -4.18486464e-02, 2.50755044e-01],
               [5.38774332e-01, -8.45835742e-01, -2.14561211e-01]]
    kernel_init = np.array(kernel1) * 2/3
    args.model_path='cls/model_iter_113_acc_0.905592_category.ckpt'
    args.repeat_num= 1
    args.rotate=3
    args.use_xyz_feature=1
    args.num_votes=12
    args.norm_pi=0
    LOG_FOUT = open(os.path.join(DUMP_DIR, 'cls_norm_pi%d_use_xyz%d_rotate%d_vote%d.txt')%(args.norm_pi,args.use_xyz_feature,args.rotate,args.num_votes), 'w')
    LOG_FOUT.write(str(args) + '\n')
    LOG_FOUT.write(str(kernel_init) + '\n')

    with tf.Graph().as_default():
        evaluate(kernel_init=kernel_init)
    LOG_FOUT.close()
