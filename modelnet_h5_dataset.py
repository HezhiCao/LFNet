'''
    ModelNet dataset. Support ModelNet40, XYZ channels. Up to 2048 points.
    Faster IO than ModelNetDataset in the first epoch.
'''

import os
import sys
import numpy as np
import h5py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import provider


# Download dataset for point cloud classification
# DATA_DIR = '/data'
# if not os.path.exists(DATA_DIR):
#     os.mkdir(DATA_DIR)
# if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
    # www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
    # zipfile = os.path.basename(www)
    # os.system('wget %s; unzip %s' % (www, zipfile))
    # os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
    # os.system('rm %s' % (zipfile))
def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=1)
    pc = pc - centroid[:,np.newaxis,:]
    m = np.max(np.sqrt(np.sum(np.power(pc,2), axis=2)),axis=1)
    pc = pc / m[:,np.newaxis,np.newaxis]
    return pc
def shuffle_data(data, labels):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], idx

def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    normal = f['normal'][:]
    label = f['label'][:]
    return (np.concatenate([data,normal],axis=-1), label)

def loadDataFile(filename):
    return load_h5(filename)


class ModelNetH5Dataset(object):
    def __init__(self, root,list_filename, batch_size = 32, npoints = 1024, shuffle=False,normalize=True):
        self.list_filename = list_filename
        self.batch_size = batch_size
        self.npoints = npoints
        self.shuffle = shuffle
        self.normalize = normalize
        self.h5_files = getDataFiles(os.path.join(root,list_filename))
        self.root=root
        self.reset()

    @staticmethod
    def augment_batch_data(batch_data, augment, rotate=0):
        if augment:
            # augment points
            jittered_data = provider.random_scale_point_cloud(batch_data[:, :, 0:3])
            jittered_data = provider.shift_point_cloud(jittered_data)
            jittered_data = provider.jitter_point_cloud(jittered_data)
            batch_data[:, :, 0:3] = jittered_data
        if rotate == 2:
            # rotated points and normal
            batch_data = provider.rotate_point_cloud_with_normal(batch_data)
        elif rotate == 3:
            batch_data = provider.rotate_perturbation_point_cloud_with_normal(batch_data)
        return provider.shuffle_points(batch_data)

    def reset(self):
        ''' reset order of h5 files '''
        self.file_idxs = np.arange(0, len(self.h5_files))
        # if self.shuffle: np.random.shuffle(self.file_idxs)
        self.current_data = None
        self.current_label = None
        self.current_file_idx = 0
        self.batch_idx = 0

    def _get_data_filename(self):
        return os.path.join(self.root,self.h5_files[self.file_idxs[self.current_file_idx]])

    def _load_data_file(self, filename):
        self.current_data,self.current_label = loadDataFile(filename)
        self.current_label = np.squeeze(self.current_label)
        self.batch_idx = 0
        if self.shuffle:
            self.current_data, self.current_label,_ = shuffle_data(self.current_data,self.current_label)
    
    def _has_next_batch_in_file(self):
        return self.batch_idx*self.batch_size < self.current_data.shape[0]

    def num_channel(self):
        return 3

    def has_next_batch(self):
        # TODO: add backend thread to load data
        if (self.current_data is None) or (not self._has_next_batch_in_file()):
            if self.current_file_idx >= len(self.h5_files):
                return False
            self._load_data_file(self._get_data_filename())
            self.batch_idx = 0
            self.current_file_idx += 1
        return self._has_next_batch_in_file()

    def next_batch(self, augment=False,rotate=0):
        ''' returned dimension may be smaller than self.batch_size '''
        start_idx = self.batch_idx * self.batch_size
        end_idx = min((self.batch_idx+1) * self.batch_size, self.current_data.shape[0])
        bsize = end_idx - start_idx

        data_batch = self.current_data[start_idx:end_idx, :, :].copy()
        label_batch = self.current_label[start_idx:end_idx].copy()
        self.batch_idx += 1
        batch_data = self.augment_batch_data(data_batch,augment,rotate)
        if self.normalize:
            batch_data[:,:,:3]=pc_normalize(batch_data[:,:,:3])
        return batch_data, label_batch

if __name__=='__main__':
    d = ModelNetH5Dataset('/data/modelnet40_ply_hdf5_2048/test_files.txt')
    print(d.shuffle)
    print(d.has_next_batch())
    ps_batch, cls_batch = d.next_batch(True)
    print(ps_batch.shape)
    print(cls_batch.shape)
