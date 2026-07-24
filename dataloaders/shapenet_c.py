"""
ShapeNet-C DataLoader for Part Segmentation Corruption Robustness
Based on ModelNet40-C structure
"""

import os
import h5py
import numpy as np
from torch.utils.data import Dataset
import torch
import json


class ShapeNetC(Dataset):
    """
    ShapeNet-C dataset for part segmentation corruption robustness evaluation
    
    Dataset structure:
    data/shapenet_c/
        ├── clean.h5
        ├── scale_0.h5 ... scale_4.h5
        ├── jitter_0.h5 ... jitter_4.h5
        ├── rotate_0.h5 ... rotate_4.h5
        ├── dropout_global_0.h5 ... dropout_global_4.h5
        ├── dropout_local_0.h5 ... dropout_local_4.h5
        ├── add_global_0.h5 ... add_global_4.h5
        └── add_local_0.h5 ... add_local_4.h5
    
    7 corruption types × 5 severity levels = 35 files
    """
    
    # Corruption types matching the file structure
    CORRUPTIONS = [
        'scale', 'jitter', 'rotate',
        'dropout_global', 'dropout_local',
        'add_global', 'add_local'
    ]
    
    # ShapeNet part categories
    seg_classes = {
        'Airplane': [0, 1, 2, 3],
        'Bag': [4, 5],
        'Cap': [6, 7],
        'Car': [8, 9, 10, 11],
        'Chair': [12, 13, 14, 15],
        'Earphone': [16, 17, 18],
        'Guitar': [19, 20, 21],
        'Knife': [22, 23],
        'Lamp': [24, 25, 26, 27],
        'Laptop': [28, 29],
        'Motorbike': [30, 31, 32, 33, 34, 35],
        'Mug': [36, 37],
        'Pistol': [38, 39, 40],
        'Rocket': [41, 42, 43],
        'Skateboard': [44, 45, 46],
        'Table': [47, 48, 49]
    }
    
    def __init__(self, data_root='dataloaders/data/shapenet_c', corruption='clean', severity=0, 
                 num_points=2048, split='test', use_original_train=False,
                 original_shapenet_path='dataloaders/data/shapenetcore_partanno_segmentation_benchmark_v0_normal'):
        """
        Args:
            data_root: path to shapenet_c folder
            corruption: corruption type ('clean' or one of CORRUPTIONS)
            severity: corruption severity level (0-4), ignored if corruption='clean'
            num_points: number of points to sample from each point cloud
            split: 'train', 'val', or 'test' (only test has corruptions)
            use_original_train: if True and split='train', load from original ShapeNet instead of clean.h5
            original_shapenet_path: path to original ShapeNet dataset
        """
        self.data_root = data_root
        self.corruption = corruption
        self.severity = severity
        self.num_points = num_points
        self.split = split
        
        # Only test set has corruptions
        if split != 'test' and corruption != 'clean':
            raise ValueError(f"Corruptions only available for test split, got split={split}")
        
        # CRITICAL: Use original ShapeNet TRAINVAL set for memory bank (matching original implementation)
        if split == 'train' and use_original_train:
            print(f"Loading ORIGINAL ShapeNet TRAINVAL set from {original_shapenet_path}")
            self._load_original_shapenet(original_shapenet_path)
            return
        
        # Load data from ShapeNet-C
        if corruption == 'clean':
            h5_file = os.path.join(data_root, 'clean.h5')
        else:
            # ShapeNet-C uses 0-indexed severity (0-4)
            h5_file = os.path.join(data_root, f'{corruption}_{severity}.h5')
        
        if not os.path.exists(h5_file):
            raise FileNotFoundError(
                f"File not found: {h5_file}\n"
                f"Please ensure ShapeNet-C data is downloaded and extracted to {data_root}"
            )
        
        # Load from HDF5
        with h5py.File(h5_file, 'r') as f:
            self.data = f['data'][:].astype('float32')      # [N, 2048, 3]
            self.labels = f['label'][:].astype('int64')     # [N,] object category
            self.seg_labels = f['pid'][:].astype('int64')   # [N, 2048] part labels
        
        # Create category to index mapping
        self.seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...}
        for cat in self.seg_classes.keys():
            for label in self.seg_classes[cat]:
                self.seg_label_to_cat[label] = cat
        
        print(f"Loaded ShapeNet-C {corruption} (severity {severity if corruption != 'clean' else 'N/A'}): "
              f"{len(self.data)} samples, {self.data.shape[1]} points")
    
    def _load_original_shapenet(self, shapenet_path):
        """Load from original ShapeNet dataset (for training set)"""
        import json
        
        # Load category mappings
        catfile = os.path.join(shapenet_path, 'synsetoffset2category.txt')
        self.cat = {}
        with open(catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        
        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))
        
        # Load train + val split (trainval) - MATCHING ORIGINAL
        train_split = os.path.join(shapenet_path, 'train_test_split', 
                                   'shuffled_train_file_list.json')
        val_split = os.path.join(shapenet_path, 'train_test_split',
                                'shuffled_val_file_list.json')
        
        with open(train_split, 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(val_split, 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        
        # Combine train + val for trainval split
        trainval_ids = train_ids | val_ids
        
        print(f"  Found {len(train_ids)} train + {len(val_ids)} val = {len(trainval_ids)} trainval shape IDs")
        
        # Load all data
        all_data = []
        all_labels = []
        all_seg_labels = []
        
        loaded_count = 0
        for cat_id in sorted(self.cat.keys()):
            cat_name = self.cat[cat_id]
            shape_dir = os.path.join(shapenet_path, cat_name)
            
            if not os.path.exists(shape_dir):
                print(f"  ⚠️  Category directory not found: {shape_dir}")
                continue
            
            # Get all .txt files in this category
            shape_files = sorted([f for f in os.listdir(shape_dir) if f.endswith('.txt')])
            
            cat_count = 0
            for shape_file in shape_files:
                shape_id = shape_file[0:-4]  # Remove .txt extension
                
                # Check if this shape is in trainval set
                if shape_id not in trainval_ids:
                    continue
                
                # Load point cloud
                filepath = os.path.join(shape_dir, shape_file)
                try:
                    data = np.loadtxt(filepath).astype(np.float32)
                except Exception as e:
                    print(f"  ⚠️  Error loading {filepath}: {e}")
                    continue
                
                # data format: [x, y, z, nx, ny, nz, label]
                # Use only xyz coordinates (first 3 columns)
                points = data[:, 0:3]
                seg_labels = data[:, -1].astype(np.int64)
                
                all_data.append(points)
                all_labels.append(self.classes[cat_id])
                all_seg_labels.append(seg_labels)
                
                cat_count += 1
                loaded_count += 1
            
            if cat_count > 0:
                print(f"  ✓ Loaded {cat_count:4d} shapes from {cat_name}")
        
        # Convert labels to array
        self.data = all_data  # List of variable-length arrays
        self.labels = np.array(all_labels, dtype=np.int64)
        self.seg_labels = all_seg_labels  # List of variable-length arrays
        self.is_original = True  # Flag to handle variable-length data
        
        print(f"✓ Loaded ORIGINAL ShapeNet TRAINVAL set: {len(self.data)} samples")
        
        if len(self.data) < 13900:
            print(f"⚠️  WARNING: Expected ~13,998 trainval samples, got {len(self.data)}")
            print(f"    Missing {13998 - len(self.data)} samples (~{100*(13998-len(self.data))/13998:.1f}%)")
            print(f"    This is acceptable for evaluation, but may slightly affect results.")
        else:
            print(f"✓ Sample count matches expected trainval size!")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get labels
        category = self.labels[idx]
        
        # Handle original ShapeNet (variable-length) vs ShapeNet-C (fixed-length)
        if hasattr(self, 'is_original') and self.is_original:
            # Original ShapeNet: variable number of points
            pointcloud = self.data[idx]  # [variable, 3]
            seg_label = self.seg_labels[idx]  # [variable]
            
            # Sample num_points
            num_available = pointcloud.shape[0]
            if num_available >= self.num_points:
                choice = np.random.choice(num_available, self.num_points, replace=False)
            else:
                choice = np.random.choice(num_available, self.num_points, replace=True)
            
            pointcloud = pointcloud[choice, :]
            seg_label = seg_label[choice]
        else:
            # ShapeNet-C: fixed 2048 points
            pointcloud = self.data[idx]          # [2048, 3]
            seg_label = self.seg_labels[idx]     # [2048]
            
            # Resample to num_points if needed
            if pointcloud.shape[0] != self.num_points:
                choice = np.random.choice(pointcloud.shape[0], self.num_points, replace=True)
                pointcloud = pointcloud[choice, :]
                seg_label = seg_label[choice]
        
        # Convert to torch tensors
        pointcloud = torch.from_numpy(pointcloud).float()
        category = torch.tensor(category).long()
        seg_label = torch.from_numpy(seg_label).long()
        
        return pointcloud, category, seg_label


class ShapeNetCDataModule:
    """
    Helper class for ShapeNet-C evaluation across multiple corruptions
    """
    
    def __init__(self, data_root='dataloaders/data/shapenet_c', num_points=2048, batch_size=8,
                 use_original_train=True,
                 original_shapenet_path='dataloaders/data/shapenetcore_partanno_segmentation_benchmark_v0_normal'):
        self.data_root = data_root
        self.num_points = num_points
        self.batch_size = batch_size
        self.corruptions = ShapeNetC.CORRUPTIONS
        self.use_original_train = use_original_train
        self.original_shapenet_path = original_shapenet_path
    
    def get_train_loader(self):
        """Get training set loader (from ORIGINAL ShapeNet TRAINVAL - train+val combined)"""
        dataset = ShapeNetC(
            data_root=self.data_root,
            corruption='clean',
            severity=0,
            num_points=self.num_points,
            split='train',
            use_original_train=self.use_original_train,
            original_shapenet_path=self.original_shapenet_path
        )
        from torch.utils.data import DataLoader
        return DataLoader(dataset, batch_size=self.batch_size, 
                         shuffle=False, num_workers=4, drop_last=False)
    
    def get_clean_loader(self):
        """Get clean test set loader (from ShapeNet-C clean.h5)"""
        dataset = ShapeNetC(
            data_root=self.data_root,
            corruption='clean',
            severity=0,
            num_points=self.num_points,
            split='test'
        )
        from torch.utils.data import DataLoader
        return DataLoader(dataset, batch_size=self.batch_size, 
                         shuffle=False, num_workers=4, drop_last=False)
    
    def get_corrupted_loader(self, corruption, severity):
        """Get corrupted test set loader for specific corruption and severity"""
        dataset = ShapeNetC(
            data_root=self.data_root,
            corruption=corruption,
            severity=severity,
            num_points=self.num_points,
            split='test'
        )
        from torch.utils.data import DataLoader
        return DataLoader(dataset, batch_size=self.batch_size,
                         shuffle=False, num_workers=4, drop_last=False)
    
    def get_all_corruptions(self, severity):
        """Get loaders for all corruptions at given severity"""
        loaders = {}
        for corruption in self.corruptions:
            loaders[corruption] = self.get_corrupted_loader(corruption, severity)
        return loaders
    
    def get_all_severities(self, corruption):
        """Get loaders for all severities of given corruption"""
        loaders = {}
        for severity in range(5):
            loaders[severity] = self.get_corrupted_loader(corruption, severity)
        return loaders


def compute_part_seg_mIoU(pred_labels, true_labels, num_classes=50):
    """
    Compute mean IoU for part segmentation
    
    Args:
        pred_labels: [N, num_points] predicted part labels
        true_labels: [N, num_points] ground truth part labels
        num_classes: number of part classes (50 for ShapeNet)
    
    Returns:
        mIoU: mean IoU across all part classes
        class_ious: IoU per class
    """
    class_ious = []
    
    for class_id in range(num_classes):
        # Find where this class appears in ground truth
        if (true_labels == class_id).sum() == 0:
            continue  # Skip classes not present
        
        # Compute IoU for this class
        intersection = ((pred_labels == class_id) & (true_labels == class_id)).sum().float()
        union = ((pred_labels == class_id) | (true_labels == class_id)).sum().float()
        
        if union > 0:
            iou = intersection / union
            class_ious.append(iou.item())
    
    mIoU = np.mean(class_ious) if class_ious else 0.0
    return mIoU, class_ious


# Example usage
if __name__ == '__main__':
    # Test dataloader
    print("Testing ShapeNet-C dataloader...")
    
    # Load clean data
    clean_dataset = ShapeNetC(
        data_root='dataloaders/data/shapenet_c',
        corruption='clean',
        severity=0,
        num_points=2048
    )
    
    print(f"\nClean dataset size: {len(clean_dataset)}")
    points, category, seg_labels = clean_dataset[0]
    print(f"Sample shape - Points: {points.shape}, Category: {category}, Seg labels: {seg_labels.shape}")
    
    # Load corrupted data
    print("\nLoading corrupted datasets...")
    for corruption in ['scale', 'jitter', 'dropout_global']:
        corrupted_dataset = ShapeNetC(
            data_root='dataloaders/data/shapenet_c',
            corruption=corruption,
            severity=4,
            num_points=2048
        )
        print(f"{corruption}_4: {len(corrupted_dataset)} samples")
    
    # Test DataModule
    print("\nTesting DataModule...")
    data_module = ShapeNetCDataModule(data_root='dataloaders/data/shapenet_c')
    clean_loader = data_module.get_clean_loader()
    print(f"Clean loader: {len(clean_loader)} batches")
    train_loader = data_module.get_train_loader()
    print(f"Train loader: {len(train_loader)} batches")
    print(f"Train length: {len(train_loader.dataset)} samples")
    
    # Test batch
    for batch in clean_loader:
        points, categories, seg_labels = batch
        print(f"Batch - Points: {points.shape}, Categories: {categories.shape}, Seg: {seg_labels.shape}")
        break
    
    print("\n✓ ShapeNet-C dataloader test passed!")