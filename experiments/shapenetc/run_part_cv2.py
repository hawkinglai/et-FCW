import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import argparse
import os
import pandas as pd

from models.parts import PartNet
from dataloaders.ShapeNet import PartNormalDataset
from dataloaders.shapenet_cv2 import ShapeNetC

def compute_overall_iou(pred, target, num_classes=50):
    shape_ious = []
    pred = pred.max(dim=2)[1]
    pred_np = pred.cpu().data.numpy()
    target_np = target.cpu().data.numpy()
    
    for shape_idx in range(pred.shape[0]):
        part_ious = []
        for part in range(num_classes):
            I = np.sum(np.logical_and(pred_np[shape_idx] == part, target_np[shape_idx] == part))
            U = np.sum(np.logical_or(pred_np[shape_idx] == part, target_np[shape_idx] == part))
            F_count = np.sum(target_np[shape_idx] == part)

            if F_count != 0:       
                iou = I / float(U)
                part_ious.append(iou)
        if len(part_ious) > 0:
            shape_ious.append(np.mean(part_ious))
    return shape_ious

def get_arguments():
    import ast
    parser = argparse.ArgumentParser()
    parser.add_argument('--bz', type=int, default=128)
    parser.add_argument('--points', type=str, default="[1024, 512, 256, 128]") # Standard part point counts
    parser.add_argument('--stages', type=int, default=4)
    parser.add_argument('--k', type=str, default="[105,105,110,120]")
    parser.add_argument('--metric', type=int, default=2)
    parser.add_argument('--rescale', type=float, default=0.2)
    parser.add_argument('--de_k', type=int, default=3)
    parser.add_argument('--gamma', type=int, default=230)
    
    # New Arguments for Stress Test
    parser.add_argument('--data_root', type=str, default=os.environ.get('TFCW_SHAPENETC_ROOT'))
    parser.add_argument('--corruption', type=str, default='add_local', help='Type of corruption to evaluate') 
    
    args = parser.parse_args()
    if not args.data_root:
        raise SystemExit('Set TFCW_SHAPENETC_ROOT or pass --data_root (directory with clean.h5 and {jitter,add_global}_{0..4}.h5).')
    args.points = ast.literal_eval(args.points)
    args.k = ast.literal_eval(args.k)
    return args


@torch.no_grad()
def main():
    args = get_arguments()
    torch.manual_seed(3407)
    
    # We will test both surfaces to prove the redundancy theory
    surfaces = ['knnxyz', 'geobp']
    final_results = {}

    print(f"==> Starting Part Segmentation Stress Test on: {args.corruption.upper()}")

    for surface in surfaces:
        print(f"\n{'='*50}")
        print(f"Evaluating Surface: {surface.upper()}")
        print(f"{'='*50}")
        
        final_results[surface] = {}
        
        mini_model = PartNet(
            input_points=args.points, num_stages=args.stages, k_neighbors=args.k,
            rescale=args.rescale, metric=args.metric, surface=surface,
            de_neighbors=args.de_k
        ).cuda().eval()

        print('--> Building Clean Memory Bank...')
        train_loader = DataLoader(
            PartNormalDataset(npoints=args.points[0], split='trainval', normalize=False), 
            num_workers=8, batch_size=args.bz, shuffle=False
        )
        
        num_part, num_shape = 50, 16
        feature_memory = [[] for _ in range(num_shape)]
        label_memory = [[] for _ in range(num_shape)]
        
        for points, shape_label, part_label, _ in tqdm(train_loader, leave=False):
            points = points.float().cuda()
            shape_labels = shape_label.long().cuda().squeeze(1)
            part_labels = part_label.long().cuda()
            
            point_features = mini_model(points).permute(0, 2, 1)
            
            for b in range(point_features.shape[0]):
                feature_memory_list, label_memory_list = [], []
                point_feature = point_features[b]
                s_label = shape_labels[b]
                p_label = part_labels[b]

                for i in range(num_part):
                    part_mask = (p_label == i)
                    if torch.sum(part_mask) == 0: continue
                    part_features = point_feature[part_mask].mean(0).unsqueeze(0)
                    feature_memory_list.append(part_features)
                    label_memory_list.append(torch.tensor(i).unsqueeze(0))
                
                if feature_memory_list:
                    feature_memory[int(s_label)].append(torch.cat(feature_memory_list, dim=0))
                    label_memory[int(s_label)].append(F.one_hot(torch.cat(label_memory_list, dim=0), num_classes=num_part))

        for i in range(num_shape):
            if len(feature_memory[i]) > 0:
                feature_memory[i] = torch.cat(feature_memory[i], dim=0)
                feature_memory[i] /= feature_memory[i].norm(dim=-1, keepdim=True)
                feature_memory[i] = feature_memory[i].permute(1, 0)
                label_memory[i] = torch.cat(label_memory[i], dim=0).cuda().float()

        print(f"--> Evaluating Clean Baseline and {args.corruption} levels...")
        
        # Configuration list: ('corruption_name', file_severity_idx, table_severity_idx)
        # table_severity_idx 0 = Clean (Sev_0)
        eval_configs = [('clean', 0, 0)] + [(args.corruption, i, i+1) for i in range(5)]
        
        for corr_name, file_sev, paper_sev in eval_configs:
            try:
                test_dataset = ShapeNetC(data_root=args.data_root, corruption=corr_name, severity=file_sev, num_points=args.points[0])
                test_loader = DataLoader(test_dataset, num_workers=8, batch_size=args.bz, shuffle=False)
                sss_list = []
                logits_list, label_list = [], []
                for points, shape_label, part_label, _ in tqdm(test_loader, leave=False):
                    points = points.float().cuda()
                    sss_list.append(shape_label.cpu().numpy())
                    shape_label = shape_label.long().cuda().squeeze()
                    part_label = part_label.long().cuda()
                    
                    # Fix scalar shape_label issue when batch_size=1
                    if shape_label.dim() == 0: shape_label = shape_label.unsqueeze(0)

                    point_features = mini_model(points)
                    for b in range(point_features.shape[0]):
                        point_feature = point_features[b].permute(1, 0)
                        point_feature /= point_feature.norm(dim=-1, keepdim=True)
                        
                        s_idx = int(shape_label[b].item())
                        if len(feature_memory[s_idx]) == 0: continue # Skip if no prototypes

                        Sim = point_feature @ feature_memory[s_idx]
                        logits = (-args.gamma * (1 - Sim)).exp() @ label_memory[s_idx]
                        logits_list.append(logits.unsqueeze(0))
                        label_list.append(part_label[b].unsqueeze(0))
                if logits_list:
                    iou = compute_overall_iou(torch.cat(logits_list, dim=0), torch.cat(label_list, dim=0))
                    miou = np.mean(iou) * 100
                    final_results[surface][paper_sev] = miou
                    
                    label_name = "Clean (Sev_0)" if paper_sev == 0 else f"Severity {paper_sev}"
                    print(f"    {label_name}: mIoU {miou:.2f}%")
                else:
                    final_results[surface][paper_sev] = 0.0
                    
            except Exception as e:
                print(f"    Skipping {corr_name} Sev {paper_sev} due to error: {e}")
                final_results[surface][paper_sev] = 0.0

    print("\n" + "="*70)
    print(f"PART SEGMENTATION STRESS TEST: {args.corruption.upper()}")
    print("="*70)
    df = pd.DataFrame(final_results).T
    df.columns = [f'Sev_{i}' for i in df.columns]
    
    # Calculate Drop from Clean (Sev_0) to Max Corruption (Sev_5)
    if 'Sev_0' in df.columns and 'Sev_5' in df.columns:
        df['Total_Drop'] = df['Sev_0'] - df['Sev_5']
        
    print(df.round(2))
    print("="*70)

if __name__ == '__main__':
    main()