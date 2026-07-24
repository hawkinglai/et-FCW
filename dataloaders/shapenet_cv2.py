import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

class ShapeNetC(Dataset):
    def __init__(self, data_root, corruption='clean', severity=0, num_points=2048):
        self.data_root = data_root
        self.corruption = corruption
        self.severity = severity
        self.num_points = num_points
        
        # Determine file path
        if corruption == 'clean':
            h5_file = os.path.join(data_root, 'clean.h5')
        else:
            h5_file = os.path.join(data_root, f'{corruption}_{severity}.h5')
            
        if not os.path.exists(h5_file):
            raise FileNotFoundError(f"Could not find {h5_file}")
            
        with h5py.File(h5_file, 'r') as f:
            self.data = f['data'][:].astype('float32')
            self.label = f['label'][:].astype('int64')
            # Handle different naming conventions in h5 files
            if 'pid' in f:
                self.pid = f['pid'][:].astype('int64')
            elif 'part_label' in f:
                self.pid = f['part_label'][:].astype('int64')
            else:
                self.pid = np.zeros((self.data.shape[0], self.data.shape[1]), dtype=np.int64)
                
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        pointcloud = self.data[idx]
        shape_label = self.label[idx]
        part_label = self.pid[idx]
        
        # Resample if necessary
        if pointcloud.shape[0] != self.num_points:
            choice = np.random.choice(pointcloud.shape[0], self.num_points, replace=True)
            pointcloud = pointcloud[choice, :]
            part_label = part_label[choice]
            
        # run_part.py expects a 4th return value (norm_plt), so we pass dummy zeros
        fake_normal = np.zeros_like(pointcloud)
        
        return (
            torch.from_numpy(pointcloud).float(), 
            torch.tensor(shape_label).long(), 
            torch.tensor(part_label).long(), 
            torch.from_numpy(fake_normal).float()
        )

def main():
    data_root = os.environ.get('DATASET_ROOT', os.path.join(os.path.dirname(__file__), '..', 'data', 'shapenet_c'))
    
    # List of corruptions based on your previous 'ls' output
    corruptions = [
        'add_global', 
        'add_local', 
        'dropout_global', 
        'dropout_local', 
        'jitter', 
        'rotate', 
        'scale'
    ]
    
    print(f"Checking dataset lengths in: {data_root}\n")
    print("=" * 50)
    
    # 1. Check Clean Dataset
    try:
        clean_dataset = ShapeNetC(data_root=data_root, corruption='clean')
        print(f"{'clean':<15} | Sev: N/A | Length: {len(clean_dataset)} samples")
    except Exception as e:
        print(f"{'clean':<15} | Sev: N/A | Error: {e}")
        
    print("=" * 50)
    
    # 2. Check Corrupted Datasets
    for corr in corruptions:
        for sev in range(5):  # Files are indexed 0 to 4
            try:
                dataset = ShapeNetC(data_root=data_root, corruption=corr, severity=sev)
                print(f"{corr:<15} | Sev: {sev}   | Length: {len(dataset)} samples")
            except Exception as e:
                print(f"{corr:<15} | Sev: {sev}   | Error: {e}")
        print("-" * 50)

if __name__ == '__main__':
    main()