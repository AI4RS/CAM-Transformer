# dataset

import os
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split

def load_data(path):
    # DATA_DIR = 'data'
    # data_label = np.load(os.path.join(DATA_DIR, path))
    data_label = np.load(path)
    data_point = data_label[:,:,:-1].astype('float32')
    label_point = data_label[:,0,-1].astype('int64')

    # split training and validation
    data_point_tr, data_point_val, label_point_tr, label_point_val = train_test_split(data_point, label_point, test_size=0.2, random_state=42)

    return data_point_tr, label_point_tr, data_point_val, label_point_val

def load_data_test(path):
    # DATA_DIR = 'data'
    # data_label = np.load(os.path.join(DATA_DIR, path))
    data_label = np.load(path)
    data_point = data_label[:,:,:-1].astype('float32')
    label_point = data_label[:,0,-1].astype('int64')

    return data_point, label_point


class PointData(Dataset):
    def __init__(self, split, path):
        if split == 'train':
            self.data, self.label, _, _ = load_data(path=path)
        elif split == 'val':
            _, _, self.data, self.label = load_data(path=path)

    def __getitem__(self, item):
        pointcloud = self.data[item]
        label = self.label[item]
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]

class PointData_Test(Dataset):
    def __init__(self, path):
        self.data, self.label = load_data_test(path=path)

    def __getitem__(self, item):
        pointcloud = self.data[item]
        label = self.label[item]
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]

def n_dim_count(path):
    # DATA_DIR = 'data'
    # data_label = np.load(os.path.join(DATA_DIR, path))
    data_label = np.load(path)
    data_point = data_label[:,:,:-1].astype('float32')
    n_dim_count = data_point.shape[2]
    return n_dim_count

