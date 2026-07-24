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
    pred = pred.max(dim=2)[1]   # (batch_size, num_points)
    pred_np = pred.cpu().data.numpy()
    target_np = target.cpu().data.numpy()
    
    for shape_idx in range(pred.shape[0]):
        part_ious = []
        for part in range(num_classes):
            I = np.sum((pred_np[shape_idx] == part) & (target_np[shape_idx] == part))
            U = np.sum((pred_np[shape_idx] == part) | (target_np[shape_idx] == part))
            F = np.sum(target_np[shape_idx] == part)
            if F != 0:
                part_ious.append(I / float(U))
        shape_ious.append(np.mean(part_ious))
    return shape_ious  # list of length = num_samples

def get_arguments():
    import ast
    parser = argparse.ArgumentParser()
    parser.add_argument('--bz', type=int, default=8)  # ModelNet 128 # Aug 64 # orginal 16

    parser.add_argument('--points', type=str, default="[1024, 512, 256, 128]")
    parser.add_argument('--stages', type=int, default=4)
    parser.add_argument('--k', type=str, default="[75, 75, 75, 75]")
    parser.add_argument('--metric', type=int, default=1)
    # parser.add_argument('--metric', type=str, default='seuclidean')
    parser.add_argument('--rescale', type=float, default=0.2)
    parser.add_argument('--surface', type=str, default='knnxyz', help='Surface type') # geobp
    parser.add_argument('--de_k', type=int, default=6, help='Number of neighbors for the decoder')
    parser.add_argument('--gamma', type=int, default=100, help='Scaling factor for similarity matching')
    args = parser.parse_args()
    args.points = ast.literal_eval(args.points)
    args.k = ast.literal_eval(args.k)
    return args

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
    for points, shape_label, part_label, norm_plt in tqdm(train_loader, disable=False):
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

    torch.cuda.empty_cache()

    # -- test loop, now recording shape labels per sample --
    logits_list, label_list = [], []
    shape_labels_list = []
    for points, shape_label, part_label, norm_plt in tqdm(test_loader):
        points, norm_plt = points.cuda().float(), norm_plt.cuda().float()
        shape_label = shape_label.long().squeeze(1).cuda()
        part_label  = part_label.long().cuda()

        point_feats = mini_model(points)  # (bz, c, n)
        for b in range(point_feats.shape[0]):
            feat = point_feats[b].permute(1,0)                  # (n, c)
            feat = feat / feat.norm(dim=-1, keepdim=True)      # normalize
            sim  = feat @ feature_memory[int(shape_label[b])]
            logits = (-args.gamma * (1 - sim)).exp() @ label_memory[int(shape_label[b])]
            logits_list.append(logits.unsqueeze(0))             # (1, n, 50)
            label_list.append(part_label[b].unsqueeze(0))       # (1, n)
            shape_labels_list.append(int(shape_label[b].item()))

    logits_tensor = torch.cat(logits_list, dim=0)  # (num_samples, n, 50)
    labels_tensor = torch.cat(label_list,  dim=0)  # (num_samples, n)

    # 1) compute per-sample IoU
    sample_ious = compute_overall_iou(logits_tensor, labels_tensor)

    # 2) group by shape category
    num_shapes = 16
    ious_per_shape = {i: [] for i in range(num_shapes)}
    for idx, shape_idx in enumerate(shape_labels_list):
        ious_per_shape[shape_idx].append(sample_ious[idx])
    test = PartNormalDataset(npoints=args.points[0], split='test', normalize=False)
    
    shape_names = [f"shape_{i}" for i in range(num_shapes)]
    label_to_name = {int_label: synset_str for synset_str, int_label in test.classes.items()}
    print("\nPer-category mIoU:")
    for i in range(num_shapes):
        if len(ious_per_shape[i]) == 0:
            print(f"  {shape_names[i]:12s}: no samples")
        else:
            mean_iou = np.mean(ious_per_shape[i]) * 100
            shape_names[i] = label_to_name.get(i, shape_names[i])
            print(f"  {shape_names[i]:12s}: {mean_iou:5.2f}%")

    

    print("\nOverall mIoU across all samples: {:.2f}%".format(np.mean(compute_overall_iou(logits_tensor, labels_tensor)) * 100))

if __name__ == '__main__':
    main()
