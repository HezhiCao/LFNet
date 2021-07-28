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
DATA_DIR = os.path.dirname(os.path.dirname(ROOT_DIR))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import provider
import tf_util
import part_dataset_all_normal
import yaml
import  matplotlib.pyplot as plt
parser = argparse.ArgumentParser()
parser.add_argument('--config', default='../cfgs/config_ssn_seg.yaml', type=str)
args = parser.parse_args()
from mpl_toolkits.mplot3d import Axes3D

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

MODEL = importlib.import_module(args.model) # import network module
MODEL_FILE = os.path.join(ROOT_DIR, 'models', args.model+'.py')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

NUM_CLASSES = 50

# Shapenet official train/test split
DATA_PATH = args.data_path
TRAIN_DATASET = part_dataset_all_normal.PartNormalDataset(root=DATA_PATH, npoints=NUM_POINT, classification=False, split='trainval', return_cls_label=True)
TEST_DATASET = part_dataset_all_normal.PartNormalDataset(root=DATA_PATH, npoints=NUM_POINT, classification=False, split='test', return_cls_label=True)

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

def train(kernel_init,LOG_DIR,LOG_FOUT):
    os.system('cp %s %s' % (MODEL_FILE, LOG_DIR))  # bkp of model def
    os.system('cp train_seg.py %s' % (LOG_DIR))  # bkp of train procedure
    os.system('cp ../tf_ops/transforming/tf_transforming.py %s' % (LOG_DIR))
    os.system('cp ../tf_ops/transforming/tf_transforming.cpp %s' % (LOG_DIR))
    os.system('cp ../tf_ops/transforming/tf_transforming_g.cu %s' % (LOG_DIR))
    os.system('cp ../models/lfnet_ssn_seg.py.py %s' % (LOG_DIR))
    os.system('cp ../utils/pointnet_util.py %s' % (LOG_DIR))
    LOG_FOUT.write(str(args) + '\n')
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl, normals_pl, cls_labels_pl,axis_x,axis_y,kernel = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT,kernel_init.shape[0])
            is_training_pl = tf.placeholder(tf.bool, shape=())

            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            print("--- Get model and loss")
            # Get model and loss
            pred, end_points,kernel_out,weight,kernel_fit = MODEL.get_model(pointclouds_pl, cls_labels_pl, normals_pl,axis_x,axis_y, kernel, args.scale,args.interp,args.fit, is_training_pl,
                                                                            bn_decay=bn_decay,d=args.d,knn=args.knn,nsample = args.nsample, use_xyz_feature=args.use_xyz_feature)

            loss = MODEL.get_loss(pred, labels_pl)
            tf.summary.scalar('loss', loss)

            correct = tf.equal(tf.argmax(pred, 2), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE*NUM_POINT)
            tf.summary.scalar('accuracy', accuracy)

            print("--- Get training operator")
            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)

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
               'labels_pl': labels_pl,
               'axis_x': axis_x,
               'axis_y': axis_y,
               'cls_labels_pl': cls_labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'end_points': end_points,
               'kernel': kernel,
               'kernel_out': kernel_out,
               'weight': weight,
               'kernel_fit':kernel_fit}

        best_acc_cat = -1
        best_acc_inst = -1
        test_acc_cat = np.zeros((MAX_EPOCH + 2, 1))
        test_acc_inst = test_acc_cat
        for epoch in range(MAX_EPOCH):
            log_string(LOG_FOUT,'**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()

            train_one_epoch(LOG_FOUT,sess, ops, train_writer,kernel_init)
            _, acc_cat, acc_inst = eval_one_epoch(LOG_FOUT,sess, ops, test_writer,kernel_init)
            test_acc_cat[epoch]=acc_cat
            test_acc_inst[epoch] = acc_inst

            if acc_cat > best_acc_cat:
                best_acc_cat = acc_cat
                if args.save_ckpt:
                    save_path = saver.save(sess, os.path.join(LOG_DIR, "model_best_acc_cat.ckpt"))
                    log_string(LOG_FOUT,"Model saved in file: %s" % save_path)
            log_string(LOG_FOUT,"Best category accuracy: %f" % best_acc_cat)

            if acc_inst > best_acc_inst:
                best_acc_inst = acc_inst
                if args.save_ckpt:
                    save_path = saver.save(sess, os.path.join(LOG_DIR, "model_best_acc_inst.ckpt"))
                    log_string(LOG_FOUT,"Model saved in file: %s" % save_path)
            log_string(LOG_FOUT,"Best instance accuracy: %f" % best_acc_inst)

            # Save the variables to disk.
            if args.save_ckpt:
                if epoch % 10 == 0:
                    save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                    log_string(LOG_FOUT,"Model saved in file: %s" % save_path)
        print('______________________________________________________________________')
        print("Best category accuracy: %f" % best_acc_cat)
        print("Best instance accuracy: %f" % best_acc_inst)
        test_acc_cat[MAX_EPOCH] = best_acc_cat
        test_acc_inst[MAX_EPOCH] = best_acc_cat
        test_acc_cat[MAX_EPOCH+1] = best_acc_inst
        test_acc_inst[MAX_EPOCH+1] = best_acc_inst
        return test_acc_cat,test_acc_inst

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

def train_one_epoch(LOG_FOUT,sess, ops, train_writer,kernel_init):
    """ ops: dict mapping from string to tf ops """
    is_training = True

    log_string(LOG_FOUT, str(datetime.now()))
    # Shuffle train samples
    train_idxs = np.arange(0, len(TRAIN_DATASET))
    np.random.shuffle(train_idxs)
    num_batches = len(TRAIN_DATASET) // BATCH_SIZE

    cur_batch_data = np.zeros((BATCH_SIZE,NUM_POINT,3))
    cur_batch_normals = np.zeros((BATCH_SIZE,NUM_POINT,3))
    cur_batch_axis_x = np.zeros((BATCH_SIZE, NUM_POINT, 3))
    cur_batch_axis_y = np.zeros((BATCH_SIZE, NUM_POINT, 3))
    cur_batch_label = np.zeros((BATCH_SIZE,NUM_POINT), dtype=np.int32)

    total_correct = 0
    total_seen = 0
    loss_sum = 0

    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        batch_data, batch_label, batch_cls_label = get_batch(TRAIN_DATASET, train_idxs, start_idx, end_idx,rotate=0)

        # data augmentation step
        batch_data[:,:,0:3] = provider.jitter_point_cloud(batch_data[:,:,0:3])
        batch_data, batch_label = provider.shuffle_points_with_labels(batch_data, batch_label)
        batch_data[:, :, :3] = pc_normalize(batch_data[:, :, :3])
        axis_x = np.cross(batch_data[:, :, :3], batch_data[:, :, 3:])
        axis_x = axis_x / np.sqrt(np.sum(axis_x**2, axis=-1))[:, :, np.newaxis]
        axis_y = np.cross(axis_x, batch_data[:, :, 3:])
        bsize = batch_data.shape[0]
        cur_batch_data[0:bsize,...] = batch_data[:,:,:3]
        cur_batch_normals[0:bsize,...] = batch_data[:,:,3:]
        cur_batch_label[0:bsize,...] = batch_label
        cur_batch_axis_x[0:bsize, ...] = axis_x
        cur_batch_axis_y[0:bsize, ...] = axis_y

        feed_dict = {ops['pointclouds_pl']: cur_batch_data,
                     ops['labels_pl']: cur_batch_label,
                     ops['normals_pl']: cur_batch_normals,
                     ops['axis_x']: cur_batch_axis_x,
                     ops['axis_y']: cur_batch_axis_y,
                     ops['cls_labels_pl']: batch_cls_label,
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
        pred_val = np.argmax(pred_val, 2)
        correct = np.sum(pred_val == batch_label)
        total_correct += correct
        total_seen += (BATCH_SIZE*NUM_POINT)
        loss_sum += loss_val

        if (batch_idx+1)%50 == 0:
            log_string(LOG_FOUT,' -- %03d / %03d --' % (batch_idx+1, num_batches))
            log_string(LOG_FOUT,'mean loss: %f' % (loss_sum / 50))
            log_string(LOG_FOUT,'accuracy: %f' % (total_correct / float(total_seen)))
            total_correct = 0
            total_seen = 0
            loss_sum = 0

def eval_one_epoch(LOG_FOUT,sess, ops, test_writer,kernel_init):
    """ ops: dict mapping from string to tf ops """
    global EPOCH_CNT
    is_training = False
    test_idxs = np.arange(0, len(TEST_DATASET))
    # Test on all data: last batch might be smaller than BATCH_SIZE
    num_batches = (len(TEST_DATASET)+BATCH_SIZE-1) // BATCH_SIZE

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

    for batch_idx in range(num_batches):
        if batch_idx %50==0:
            log_string(LOG_FOUT,'%03d/%03d'%(batch_idx, num_batches))
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(len(TEST_DATASET), (batch_idx+1) * BATCH_SIZE)
        cur_batch_size = end_idx-start_idx
        batch_data, batch_label, batch_cls_label = get_batch(TEST_DATASET, test_idxs, start_idx, end_idx,rotate=args.rotate)
        batch_data[:, :, :3]=pc_normalize(batch_data[:, :, :3])
        axis_x = np.cross(batch_data[:, :, :3], batch_data[:, :, 3:])
        axis_x = axis_x / np.sqrt(np.sum(axis_x**2, axis=-1))[:, :, np.newaxis]
        axis_y = np.cross(axis_x, batch_data[:, :, 3:])

        cur_batch_data[0:cur_batch_size,...] = batch_data[:,:,:3]
        cur_batch_normals[0:cur_batch_size,...] = batch_data[:,:,3:]
        cur_batch_axis_x[0:cur_batch_size, ...] = axis_x
        cur_batch_axis_y[0:cur_batch_size, ...] = axis_y
        cur_batch_label[0:cur_batch_size,...] = batch_label
        cur_batch_cls_label[0:cur_batch_size] = batch_cls_label

        # ---------------------------------------------------------------------
        feed_dict = {ops['pointclouds_pl']: cur_batch_data,
                     ops['labels_pl']: cur_batch_label,
                     ops['cls_labels_pl']: cur_batch_cls_label,
                     ops['normals_pl']: cur_batch_normals,
                     ops['axis_x']: cur_batch_axis_x,
                     ops['axis_y']: cur_batch_axis_y,
                     ops['is_training_pl']: is_training,
                     ops['kernel']:kernel_init}
        if args.save:
            summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                                                          ops['loss'], ops['pred']], feed_dict=feed_dict)
            test_writer.add_summary(summary, step)
        else:
            step, loss_val, pred_val = sess.run([ops['step'], ops['loss'], ops['pred']], feed_dict=feed_dict)
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
    mean_shape_ious = np.mean(list(shape_ious.values()))
    log_string(LOG_FOUT,'eval mean loss: %f' % (loss_sum / float(len(TEST_DATASET)/BATCH_SIZE)))
    log_string(LOG_FOUT,'eval accuracy: %f'% (total_correct / float(total_seen)))
    log_string(LOG_FOUT,'eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))
    for cat in sorted(shape_ious.keys()):
        log_string(LOG_FOUT,'eval mIoU of %s:\t %f'%(cat, shape_ious[cat]))
    log_string(LOG_FOUT,'eval mean mIoU: %f' % (mean_shape_ious))
    log_string(LOG_FOUT,'eval mean mIoU (all shapes): %f' % (np.mean(all_shape_ious)))

    EPOCH_CNT += 1
    return total_correct/float(total_seen), mean_shape_ious, np.mean(all_shape_ious)
def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=1)
    pc = pc - centroid[:,np.newaxis,:]
    m = np.max(np.sqrt(np.sum(np.power(pc,2), axis=2)),axis=1)
    pc = pc / m[:,np.newaxis,np.newaxis]
    return pc

if __name__ == "__main__":
    n = 9
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
        LOG_DIR = args.log_dir #+ '/log%d' % (i)
        if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)

        LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
        log_string(LOG_FOUT, 'pid: %s' % (str(os.getpid())))
        LOG_FOUT.write(str(kernel_init_3D) + '\n')
        test_acc_cat,test_acc_inst = train(kernel_init_3D, LOG_DIR, LOG_FOUT)
        LOG_FOUT.close()
        acc_cat = open(os.path.join(LOG_DIR, 'test_acc_cat.txt'), 'w')
        acc_inst=open(os.path.join(LOG_DIR, 'test_acc_inst.txt'), 'w')
        for j in range(test_acc_cat.shape[0]):
            log_string(acc_cat, '%f' % (test_acc_cat[j]))
            log_string(acc_inst, '%f' % (test_acc_inst[j]))
        acc_cat.close()
