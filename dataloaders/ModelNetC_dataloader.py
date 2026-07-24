"""
ModelNet40-C DataLoader
Integrates corruption robustness evaluation into your existing framework
"""

import os
import h5py
import numpy as np
from torch.utils.data import Dataset
import torch

class ModelNet40C(Dataset):
    """
    ModelNet40-C dataset for corruption robustness evaluation
    
    Dataset structure:
    data/modelnet_c/
        ├── clean.h5
        ├── scale_1.h5
        ├── scale_2.h5
        ...
        ├── uniform_1.h5
        └── uniform_5.h5
    
    15 corruption types × 5 severity levels = 75 files
    
    NOTE: clean.h5 contains BOTH train and test sets.
    We use the standard ModelNet40 split: first 9843 samples = train, rest = test
    """
    
    # 15 corruption types as defined in the paper
    CORRUPTIONS = [
        'scale', 'jitter', 'rotate', 'dropout_global', 'dropout_local',
        'add_global', 'add_local', 
        'distortion', 'distortion_rbf', 'distortion_rbf_inv',
        'density', 'density_inc',
        'shear', 'rotation', 'cutout',
        'uniform', 'gaussian', 'background', 'impulse', 'upsampling',
        'occlusion', 'lidar'
    ]
    
    def __init__(self, data_root='data/modelnet_c', corruption='clean', severity=0, 
                 num_points=1024, partition='train'):
        """
        Args:
            data_root: path to modelnet_c folder
            corruption: corruption type ('clean' or one of CORRUPTIONS)
            severity: corruption severity level (0-4), ignored if corruption='clean'
            num_points: number of points to sample from each point cloud
            partition: 'train' or 'test' - only used for clean data
        """
        self.data_root = data_root
        self.corruption = corruption
        self.severity = severity
        self.num_points = num_points
        self.partition = partition
        
        # Load data
        if corruption == 'clean':
            h5_file = os.path.join(data_root, 'clean.h5')
        else:
            # ModelNet40-C uses 0-indexed severity (0-4, not 1-5)
            h5_file = os.path.join(data_root, f'{corruption}_{severity}.h5')
        
        if not os.path.exists(h5_file):
            raise FileNotFoundError(
                f"File not found: {h5_file}\n"
                f"Please download ModelNet40-C from: "
                f"https://drive.google.com/uc?id=1KE6MmXMtfu_mgxg4qLPdEwVD5As8B0rm"
            )
        
        with h5py.File(h5_file, 'r') as f:
            all_data = f['data'][:].astype('float32')  # [N, 2048, 3]
            all_labels = f['label'][:].astype('int64')  # [N,]
        
        # Split train/test for clean data (standard ModelNet40 split)
        # Train: 9843 samples, Test: 2468 samples
        if corruption == 'clean':
            if partition == 'train':
                self.data = all_data[:9843]
                self.labels = all_labels[:9843]
            else:  # test
                self.data = all_data[9843:]
                self.labels = all_labels[9843:]
        else:
            # Corrupted data is test-only (2468 samples)
            self.data = all_data
            self.labels = all_labels
        
        print(f"Loaded {corruption} (severity {severity if corruption != 'clean' else 'N/A'}, "
              f"partition {partition if corruption == 'clean' else 'test'}): "
              f"{len(self.data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get point cloud
        pointcloud = self.data[idx]  # [2048, 3]
        label = self.labels[idx]
        
        # Resample to num_points if needed
        if pointcloud.shape[0] != self.num_points:
            choice = np.random.choice(pointcloud.shape[0], self.num_points, replace=True)
            pointcloud = pointcloud[choice, :]
        
        # Convert to torch tensors
        pointcloud = torch.from_numpy(pointcloud).float()
        label = torch.tensor(label).long()
        
        return pointcloud, label


def compute_corruption_error(clean_acc, corrupted_acc):
    """
    Compute Corruption Error (CE) as defined in the paper
    CE = (100 - corrupted_acc) / (100 - clean_acc)
    """
    if clean_acc >= 100:
        return 0.0
    return (100.0 - corrupted_acc) / (100.0 - clean_acc)


def compute_mCE(clean_acc, corruption_accs):
    """
    Compute mean Corruption Error (mCE)
    
    Args:
        clean_acc: accuracy on clean data (%)
        corruption_accs: dict mapping corruption_name -> accuracy (%)
    
    Returns:
        mCE: mean corruption error
    """
    ces = []
    for corruption, acc in corruption_accs.items():
        ce = compute_corruption_error(clean_acc, acc)
        ces.append(ce)
    
    return np.mean(ces)