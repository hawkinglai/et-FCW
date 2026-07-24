import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
import numpy as np
from tqdm import tqdm
import argparse

from models.parts import PartNet
from dataloaders.ShapeNet import PartNormalDataset

def compute_overall_iou(pred, target, num_classes=50):
    shape_ious = []
    pred = pred.max(dim=2)[1]   # (batch_size, num_points)  the pred_class_idx of each point in each sample
    pred_np = pred.cpu().data.numpy()

    target_np = target.cpu().data.numpy()
    
    for shape_idx in range(pred.shape[0]):   # sample_idx
        part_ious = []
        for part in range(num_classes):   # class_idx! no matter which category, only consider all part_classes of all categories, check all 50 classes
            # for target, each point has a class no matter which category owns this point! also 50 classes!!!
            # only return 1 when both belongs to this class, which means correct:
            I = np.sum(np.logical_and(pred_np[shape_idx] == part, target_np[shape_idx] == part))
            # always return 1 when either is belongs to this class:
            U = np.sum(np.logical_or(pred_np[shape_idx] == part, target_np[shape_idx] == part))
            F = np.sum(target_np[shape_idx] == part)

            if F != 0:       
                iou = I / float(U)    #  iou across all points for this class
                part_ious.append(iou)   #  append the iou of this class
        shape_ious.append(np.mean(part_ious))   # each time append an average iou across all classes of this sample (sample_level!)
    return shape_ious   # [batch_size]


def get_arguments():
    import ast
    parser = argparse.ArgumentParser()
    parser.add_argument('--bz', type=int, default=128)  # ModelNet 128 # Aug 64 # orginal 16

    parser.add_argument('--points', type=str, default="[1024, 512, 256, 128]")
    parser.add_argument('--stages', type=int, default=4)
    parser.add_argument('--k', type=str, default="[105,105,110,120]")
    parser.add_argument('--metric', type=int, default=2)
    parser.add_argument('--rescale', type=float, default=0.2)
    parser.add_argument('--surface', type=str, default='knnxyz', help='Surface type')
    parser.add_argument('--de_k', type=int, default=3, help='Number of neighbors for the decoder')
    parser.add_argument('--gamma', type=int, default=230, help='Scaling factor for similarity matching')
    args = parser.parse_args()
    args.points = ast.literal_eval(args.points)
    args.k = ast.literal_eval(args.k)
    return args


@torch.no_grad()
def main():

    print('==> Loading args..')
    args = get_arguments()
    print(args)
    torch.manual_seed(3407)

    print('==> Preparing model..')
    mini_model = PartNet(
        input_points=args.points,
        num_stages=args.stages,
        k_neighbors=args.k,
        rescale=args.rescale,
        metric=args.metric,
        surface=args.surface,
        de_neighbors=args.de_k
    ).cuda()
    
    mini_model.eval()


    print('==> Preparing data..')
    train_loader = DataLoader(PartNormalDataset(npoints=args.points[0], split='trainval', normalize=False), 
                                num_workers=8, batch_size=args.bz, shuffle=False, drop_last=False)
    test_loader = DataLoader(PartNormalDataset(npoints=args.points[0], split='test', normalize=False), 
                                num_workers=8, batch_size=args.bz, shuffle=False, drop_last=False)


    print('==> Constructing Point-Memory Bank..')
    num_part, num_shape = 50, 16
    # We organize point-memory bank by 16 shape labels
    feature_memory = [[] for i in range(num_shape)]
    label_memory = [[] for i in range(num_shape)]
    start_time = time.time()
    for points, shape_label, part_label, norm_plt in tqdm(train_loader, disable=True):
        # pre-process
        points = points.float().cuda()
        norm_plt = norm_plt.float().cuda()
        shape_labels = shape_label.long().cuda().squeeze(1)
        part_labels = part_label.long().cuda()
        
        # Pass through the Non-Parametric Encoder + Decoder
        point_features = mini_model(points)
        # All 2048 point features in a shape
        point_features = point_features.permute(0, 2, 1)  # bz, 2048, c
        # Extracting part prototypes for a shape
        for b in range(point_features.shape[0]):
            feature_memory_list = []
            label_memory_list = []
            point_feature = point_features[b]
            shape_label = shape_labels[b]
            part_label = part_labels[b]

            for i in range(num_part):
                # Find the point indices for the part_label within a shape
                part_mask = (part_label == i)
                if torch.sum(part_mask) == 0:
                    continue
                # Extract point features for the part_label
                part_features = point_feature[part_mask]
                # Obtain part prototypes by average point features for the part_label
                part_features = part_features.mean(0).unsqueeze(0)
                feature_memory_list.append(part_features)
                label_memory_list.append(torch.tensor(i).unsqueeze(0))
            
            # Feature Memory: store prototypes indexed by the corresponding shape_label
            feature_memory_list = torch.cat(feature_memory_list, dim=0)
            feature_memory[int(shape_label)].append(feature_memory_list)


            # Label Memory: store labels indexed by the corresponding shape_label
            label_memory_list = torch.cat(label_memory_list, dim=0)
            label_memory_list = F.one_hot(label_memory_list, num_classes=num_part)
            label_memory[int(shape_label)].append(label_memory_list)

    # Organize the point-memory bank
    for i in range(num_shape):
        # Feature Memory
        feature_memory[i] = torch.cat(feature_memory[i], dim=0)
        feature_memory[i] /= feature_memory[i].norm(dim=-1, keepdim=True)
        feature_memory[i] = feature_memory[i].permute(1, 0)
        # Label Memory
        label_memory[i] = torch.cat(label_memory[i], dim=0).cuda().float()

    print('==> Starting Point-NN..')
    logits_list, label_list = [], []
    for points, shape_label, part_label, norm_plt in tqdm(test_loader, disable=True):
        
        # pre-process
        points = points.float().cuda()
        norm_plt = norm_plt.float().cuda()
        shape_label = shape_label.long().cuda().squeeze(1)
        part_label = part_label.long().cuda()

        # Pass through the Non-Parametric Encoder + Decoder
        point_features = mini_model(points)
        for b in range(point_features.shape[0]):
            point_feature = point_features[b]
            point_feature = point_feature.permute(1, 0)
            point_feature /= point_feature.norm(dim=-1, keepdim=True)
            # Similarity Matching
            Sim = point_feature @ feature_memory[int(shape_label[b])]

            # Label Integrate
            logits = (-args.gamma * (1 - Sim)).exp() @ label_memory[int(shape_label[b])]
            logits_list.append(logits.unsqueeze(0))
        label_list.append(part_label)
            
    logits_list = torch.cat(logits_list, dim=0)
    label_list = torch.cat(label_list, dim=0)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")
    # Compute mIoU
    iou = compute_overall_iou(logits_list, label_list)
    miou = np.mean(iou) * 100
    
    print(f"Point-NN's part segmentation mIoU: {miou:.2f}.")


if __name__ == '__main__':
    main()