import os
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
from collections import defaultdict
import random
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def _get_data_files(list_filename):
    with open(list_filename) as f:
        return [line.rstrip()[5:] for line in f]

def load_data(partition):
    # download()
    DATA_DIR = os.environ['TFCW_MODELNET_DATA_ROOT']
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
    
        f = h5py.File(h5_name,'r')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)

    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)

    return all_data, all_label

def rotate_pointcloud(pointcloud):
    theta = np.pi * 2 * np.random.uniform()
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    pointcloud[:, [0, 2]] = pointcloud[:, [0, 2]].dot(rotation_matrix)  # random rotation (x,z)
    return pointcloud


class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train', split=1, seed=3407):
        self.data, self.label = load_data(partition)
        self.seed = seed
        self.num_points = num_points
        self.partition = partition   
        

    def __getitem__(self, item):
        # Copy: the shuffle below must not write through into self.data (a view of the h5 array).
        pointcloud = self.data[item][:self.num_points].copy()
        label = self.label[item]
        if self.partition == 'train':
            # Reproducible training shuffle: depends only on (seed, item), not on loader order or workers.
            perm = np.random.default_rng([self.seed, int(item)]).permutation(pointcloud.shape[0])
            pointcloud = pointcloud[perm]
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    # download()
    train = ModelNet40(1024)
    test = ModelNet40(1024, 'test')

    from torch.utils.data import DataLoader
    train_loader = DataLoader(ModelNet40(partition='train', num_points=1024), num_workers=4,
                              batch_size=32, shuffle=True, drop_last=True)
    for batch_idx, (data, label) in enumerate(train_loader):
        print(f"batch_idx: {batch_idx}  | data shape: {data.shape} | ;lable shape: {label.shape}")
        print(label)
        break
    train_set = ModelNet40(partition='train', num_points=1024)
    test_set = ModelNet40(partition='test', num_points=1024)
    print(f"train_set size {train_set.__len__()}")
    print(f"test_set size {test_set.__len__()}")
