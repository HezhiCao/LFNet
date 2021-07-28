import argparse
import math
from datetime import datetime
import h5py
import json
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.dirname(os.path.dirname(ROOT_DIR))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import provider
import tf_util
import part_dataset_all_normal
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='../cfgs/config_ssn_seg.yaml', type=str)
parser.add_argument('--jitter_normal', type=int, default=1, help='jitter normal or not')
args = parser.parse_args()

with open(args.config) as f:
    config = yaml.load(f)
print("\n**************************")
for k, v in config['common'].items():
    setattr(args, k, v)
    print('\n[%s]:'%(k), v)
print("\n**************************\n")


EPOCH_CNT = 0

BATCH_SIZE = args.batch_size
NUM_POINT = args.num_point
GPU_INDEX = args.gpu

# MODEL_PATH = args.model_path
MODEL = importlib.import_module(args.model) # import network module
MODEL_FILE = os.path.join(ROOT_DIR, 'models', args.model+'.py')
LOG_DIR = args.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure

NUM_CLASSES = 50

# Shapenet official train/test split
DATA_PATH = args.data_path
TEST_DATASET = part_dataset_all_normal.PartNormalDataset(root=DATA_PATH, npoints=NUM_POINT, classification=False, split='test', return_cls_label=True)

def log_string(LOG_FOUT,out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def evaluate(kernel_init,LOG_FOUT,d=[1,2,4],nsample=[[48],[32]],use_xyz_feature=True,rotate=0):
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl, normals_pl,cls_labels_pl,axis_x,axis_y, kernel = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT,kernel_init.shape[0])
            is_training_pl = tf.placeholder(tf.bool, shape=())
            print(is_training_pl)

            print("--- Get model and loss")
            # pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl)
            pred, end_points, kernel_out, weight, kernel_fit = MODEL.get_model(pointclouds_pl, cls_labels_pl,
                                                                               normals_pl, axis_x,axis_y, kernel, 1,
                                                                               0,
                                                                               0, is_training_pl
                                                                               , d=d,
                                                                               knn=1, nsample=nsample,use_xyz_feature=use_xyz_feature)

            loss = MODEL.get_loss(pred, labels_pl)
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)
        # Restore variables from disk.
        saver.restore(sess, args.model_path)
        ops = {'pointclouds_pl': pointclouds_pl,
               'normals_pl': normals_pl,
               'axis_x': axis_x,
               'axis_y': axis_y,
               'cls_labels_pl': cls_labels_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'kernel': kernel,
               'kernel_out': kernel_out,
               'weight': weight,
               'kernel_fit': kernel_fit
               }

        eval_one_epoch(sess, ops,kernel_init,rotate,LOG_FOUT)

def get_batch(dataset, idxs, start_idx, end_idx,rotate=0):
    bsize = end_idx-start_idx
    batch_data = np.zeros((bsize, NUM_POINT, 6))
    batch_label = np.zeros((bsize, NUM_POINT), dtype=np.int32)
    batch_cls_label = np.zeros((bsize,), dtype=np.int32)
    for i in range(bsize):
        ps,normal,seg,cls = dataset[idxs[i+start_idx]]
        batch_data[i,:,0:3] = ps
        batch_data[i,:,3:6] = normal
        batch_label[i,:] = seg
        batch_cls_label[i] = cls
    if rotate == 2:
        # rotated points and normal
        batch_data = provider.rotate_point_cloud_with_normal(batch_data)
    elif rotate == 3:
        batch_data = provider.rotate_perturbation_point_cloud_with_normal(batch_data)
    return batch_data, batch_label, batch_cls_label

def eval_one_epoch(sess, ops,kernel_init,rotate,LOG_FOUT):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    test_idxs = np.arange(0, len(TEST_DATASET))
    # Test on all data: last batch might be smaller than BATCH_SIZE
    num_batches = (len(TEST_DATASET)+BATCH_SIZE-1)//BATCH_SIZE

    cur_batch_data = np.zeros((BATCH_SIZE,NUM_POINT,3))
    cur_batch_normals = np.zeros((BATCH_SIZE,NUM_POINT,3))
    cur_batch_label = np.zeros((BATCH_SIZE,NUM_POINT), dtype=np.int32)
    cur_batch_cls_label = np.zeros((BATCH_SIZE), dtype=np.int32)
    cur_batch_axis_x = np.zeros((BATCH_SIZE, NUM_POINT, 3))
    cur_batch_axis_y = np.zeros((BATCH_SIZE, NUM_POINT, 3))

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]

    seg_classes = TEST_DATASET.seg_classes
    shape_ious = {cat:[] for cat in seg_classes.keys()}
    seg_label_to_cat = {} # {0:Airplane, 1:Airplane, ...49:Table}
    for cat in seg_classes.keys():
        for label in seg_classes[cat]:
            seg_label_to_cat[label] = cat

    log_string(LOG_FOUT,str(datetime.now()))
    log_string(LOG_FOUT,'---- EPOCH %03d EVALUATION ----'%(EPOCH_CNT))

    batch_data = np.zeros((BATCH_SIZE, NUM_POINT, 6))
    batch_label = np.zeros((BATCH_SIZE, NUM_POINT)).astype(np.int32)
    batch_cls_label = np.zeros((BATCH_SIZE,)).astype(np.int32)
    for batch_idx in range(num_batches):
        if batch_idx %50==0:
            log_string(LOG_FOUT,'%03d/%03d'%(batch_idx, num_batches))
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(len(TEST_DATASET), (batch_idx+1) * BATCH_SIZE)
        cur_batch_size = end_idx-start_idx
        batch_data, batch_label, batch_cls_label = get_batch(TEST_DATASET, test_idxs, start_idx, end_idx,rotate=rotate)
        batch_data[:, :, :3]=pc_normalize(batch_data[:, :, :3])
        # jittered_data = provider.jitter_point_cloud(batch_data[:, :, :3])
        # batch_data[:, :, :3] = jittered_data
        axis_x = np.cross(batch_data[:, :, :3], batch_data[:, :, 3:])
        if args.norm_pi:
            axis_x = axis_x / np.sqrt(np.sum(axis_x**2, axis=-1))[:, :, np.newaxis]
        axis_y = np.cross(axis_x, batch_data[:, :, 3:])

        cur_batch_data[0:cur_batch_size,...] = batch_data[:,:,:3]
        cur_batch_normals[0:cur_batch_size,...] = batch_data[:,:,3:]
        cur_batch_axis_x[0:cur_batch_size, ...] = axis_x
        cur_batch_axis_y[0:cur_batch_size, ...] = axis_y
        cur_batch_label[0:cur_batch_size,...] = batch_label
        cur_batch_cls_label[0:cur_batch_size] = batch_cls_label

        loss_val = 0
        pred_val = np.zeros((BATCH_SIZE, NUM_POINT, NUM_CLASSES))
        for _ in range(VOTE_NUM):
            jittered_data = provider.jitter_point_cloud(cur_batch_data)
            if args.jitter_normal:
                jittered_normal = provider.jitter_point_cloud(batch_data[:, :, 3:])
                batch_data[:, :, 3:] = jittered_normal
            feed_dict = {ops['pointclouds_pl']: jittered_data,
                         ops['labels_pl']: cur_batch_label,
                         ops['cls_labels_pl']: cur_batch_cls_label,
                         ops['normals_pl']: cur_batch_normals,
                         ops['axis_x']: cur_batch_axis_x,
                         ops['axis_y']: cur_batch_axis_y,
                         ops['is_training_pl']: is_training,
                         ops['kernel']: kernel_init}
            temp_loss_val, temp_pred_val = sess.run([ops['loss'], ops['pred']], feed_dict=feed_dict)
            loss_val += temp_loss_val
            pred_val += temp_pred_val
        loss_val /= float(VOTE_NUM)
        # ---------------------------------------------------------------------

        # Select valid data
        cur_pred_val = pred_val[0:cur_batch_size]
        # Constrain pred to the groundtruth classes (selected by seg_classes[cat])
        cur_pred_val_logits = cur_pred_val
        cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
        for i in range(cur_batch_size):
            cat = seg_label_to_cat[cur_batch_label[i,0]]
            logits = cur_pred_val_logits[i,:,:]
            cur_pred_val[i,:] = np.argmax(logits[:,seg_classes[cat]], 1) + seg_classes[cat][0]
        correct = np.sum(cur_pred_val == cur_batch_label)
        total_correct += correct
        total_seen += (cur_batch_size*NUM_POINT)
        if cur_batch_size==BATCH_SIZE:
            loss_sum += loss_val
        for l in range(NUM_CLASSES):
            total_seen_class[l] += np.sum(cur_batch_label==l)
            total_correct_class[l] += (np.sum((cur_pred_val==l) & (batch_label==l)))

        for i in range(cur_batch_size):
            segp = cur_pred_val[i,:]
            segl = cur_batch_label[i,:]
            pts = batch_data[i,:,:3]
            cat = seg_label_to_cat[segl[0]]
            part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
            for l in seg_classes[cat]:
                if (np.sum(segl==l) == 0) and (np.sum(segp==l) == 0): # part is not present, no prediction as well
                    part_ious[l-seg_classes[cat][0]] = 1.0
                else:
                    part_ious[l-seg_classes[cat][0]] = np.sum((segl==l) & (segp==l)) / float(np.sum((segl==l) | (segp==l)))
            shape_ious[cat].append(np.mean(part_ious))

    all_shape_ious = []
    for cat in shape_ious.keys():
        for iou in shape_ious[cat]:
            all_shape_ious.append(iou)
        shape_ious[cat] = np.mean(shape_ious[cat])
    print(len(all_shape_ious))
    mean_shape_ious = np.mean(list(shape_ious.values()))
    log_string(LOG_FOUT,'eval mean loss: %f' % (loss_sum / float(len(TEST_DATASET)/BATCH_SIZE)))
    log_string(LOG_FOUT,'eval accuracy: %f'% (total_correct / float(total_seen)))
    log_string(LOG_FOUT,'eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))
    for cat in sorted(shape_ious.keys()):
        log_string(LOG_FOUT,'eval mIoU of %s:\t %f'%(cat, shape_ious[cat]))
    log_string(LOG_FOUT,'eval mean mIoU: %f' % (mean_shape_ious))
    log_string(LOG_FOUT,'eval mean mIoU (all shapes): %f' % (np.mean(all_shape_ious)))

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=1)
    pc = pc - centroid[:,np.newaxis,:]
    m = np.max(np.sqrt(np.sum(np.power(pc,2), axis=2)),axis=1)
    pc = pc / m[:,np.newaxis,np.newaxis]
    return pc

if __name__ == "__main__":
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
    kernel_init = np.array(kernel1) * 2 / 3

    Log =9
    args.rotate=3
    VOTE_NUM = 12
    args.norm_pi=1

    args.jitter_normal=0
    # LOG_FOUT = open(os.path.join(LOG_DIR, 'log0%d_norm_pi%d_rotate%d_vote%d_normalize1_jitter_normal%d.txt'%(Log,args.norm_pi,rotate,VOTE_NUM,args.jitter_normal)), 'w')
    LOG_FOUT = open(os.path.join(LOG_DIR, 'log6.26_6.%d_norm_pi%d_rotate%d_vote%d_normalize1_jitter_normal%d.txt'%(Log,args.norm_pi,args.rotate,VOTE_NUM,args.jitter_normal)), 'w')
    log_string(LOG_FOUT, 'pid: %s' % (str(os.getpid())))
    args.model_path = '../log_seg/log0%d/model_best_acc_inst.ckpt'%Log
    LOG_FOUT.write(str(args) + '\n')
    LOG_FOUT.write(str(kernel_init) + '\n')
    evaluate(kernel_init, LOG_FOUT, d=[1, 2, 4], nsample=[[48], [32]], rotate=rotate,use_xyz_feature=1)
    LOG_FOUT.close()


