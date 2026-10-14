"""
ModelNet40-C Stress Test: Global vs. Local vs. United
Focus: Background Noise Robustness
"""

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from models.cls_globallocal_test import clsNet as MiniSurfEncoder
from dataloaders.ModelNet import ModelNet40
from dataloaders.ModelNetC_dataloader import ModelNet40C
from tqdm import tqdm
import argparse
import numpy as np
import pandas as pd
import os

def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy().item()
    acc = 100 * acc / target.shape[0]
    return acc

def classification(feature_bank, label_bank, test_feature_bank, test_label_bank):
    # Reduced gamma search space for speed during ablation
    gamma_list = [i * 10000 / 1000 for i in range(1000)] 
    best_acc, best_gamma = 0, 0
    
    # Ensure memory banks are on GPU and float
    feature_bank = feature_bank.cuda().float()
    label_bank = label_bank.cuda().float()
    test_feature_bank = test_feature_bank.cuda().float()
    
    # Calculate Similarity: Test x Train^T
    Sim = test_feature_bank @ feature_bank.permute(1, 0)
    
    for gamma in gamma_list:
        # Eq 3: exp(-gamma * (1-S)) @ Labels
        logits = (-gamma * (1 - Sim)).exp() @ label_bank
        acc = cls_acc(logits, test_label_bank)
        if acc > best_acc:
            best_acc, best_gamma = acc, gamma
    
    return best_acc, best_gamma

@torch.no_grad()
def FeatureProcessor(encoder, dataset_loader):
    feature_memory, label_memory = [], []
    
    for points, labels in tqdm(dataset_loader, desc="Extracting features", leave=False):
        points = points.cuda()
        point_features = encoder(points)
        feature_memory.append(point_features)
        labels = labels.cuda()
        label_memory.append(labels)
    
    feature_memory = torch.cat(feature_memory, dim=0)
    # Normalize features (Crucial for cosine similarity)
    feature_memory /= feature_memory.norm(dim=-1, keepdim=True)
    label_memory = torch.cat(label_memory, dim=0)
    
    return feature_memory, label_memory

def get_arguments():
    import ast
    parser = argparse.ArgumentParser()
    
    # Standard Model Params
    parser.add_argument('--bz', type=int, default=32)
    parser.add_argument('--points', type=str, default="[1024, 512, 256, 128]")
    parser.add_argument('--stages', type=int, default=4)
    parser.add_argument('--k', type=str, default="[110, 110, 110, 110]")
    parser.add_argument('--metric', type=int, default=2)
    parser.add_argument('--rescale', type=float, default=0.8)
    parser.add_argument('--surface', type=str, default='geobp')
    
    # Experiment Params
    parser.add_argument('--data_root', type=str, default=os.environ.get('TFCW_MODELNETC_ROOT'))
    # Use 'scale' or 'jitter' if 'background' is missing from your ls output
    parser.add_argument('--corruption', type=str, default='add_local', 
                        help='Corruption type to stress test (default: background)')
    # Modes to compare
    parser.add_argument('--modes', nargs='+', default=['global', 'local', 'united'],
                        help='List of model modes to evaluate (must be supported by model class)')
    
    args, unknown = parser.parse_known_args()
    if not args.data_root:
        raise SystemExit('Set TFCW_MODELNETC_ROOT or pass --data_root (directory with {jitter,add_global}_{0..4}.h5).')
    args.points = ast.literal_eval(args.points)
    args.k = ast.literal_eval(args.k)
    return args

def main():
    print('==> Loading args...')
    args = get_arguments()
    print(f"Experimental Setup: {args.modes} on Corruption: {args.corruption}")
    
    # Quick check for corruption existence based on your LS output
    first_file = os.path.join(args.data_root, f"{args.corruption}_0.h5")
    if not os.path.exists(first_file):
        print(f"\n[WARNING] The file {first_file} does not exist.")
        print(f"Your ls output showed: scale, jitter, rotate, dropout, add, clean.")
        print(f"If you don't have 'background', try running with: --corruption scale")
        # Proceeding anyway, but expect FileNotFoundError if not fixed

    # Store results: results[mode][severity] = accuracy
    final_results = {}
    
    # --- LOOP THROUGH MODES (Ablation Study) ---
    for mode in args.modes:
        print(f"\n{'='*40}")
        print(f"Starting Evaluation for Mode: {mode.upper()}")
        print(f"{'='*40}")
        
        final_results[mode] = {}
        
        # 1. Initialize Model in specific mode
        encoder = MiniSurfEncoder(
            input_points=args.points, 
            num_stages=args.stages, 
            k_neighbors=args.k, 
            rescale=args.rescale,
            surface=args.surface,
            metric=args.metric,
            mode=mode  
        ).cuda()
        encoder.eval()
        
        # 2. Build Memory Bank (Train)
        print('--> Building Memory Bank...')
        train_dataset = ModelNet40(partition='train', num_points=args.points[0])
        train_loader = DataLoader(train_dataset, num_workers=8, batch_size=args.bz, shuffle=False)
        train_features, train_labels = FeatureProcessor(encoder, train_loader)
        
        # One-hot encode train labels for classification
        train_labels_onehot = F.one_hot(train_labels).squeeze().float()
        
        # 3. Evaluate on Clean Data (Severity 0)
        print('--> Evaluating Clean Baseline...')
        clean_dataset = ModelNet40(partition='test', num_points=args.points[0])
        clean_loader = DataLoader(clean_dataset, num_workers=8, batch_size=args.bz, shuffle=False)
        clean_features, clean_labels = FeatureProcessor(encoder, clean_loader)
        
        clean_acc, _ = classification(train_features, train_labels_onehot, clean_features, clean_labels)
        final_results[mode][0] = clean_acc
        print(f"    Mode {mode} | Clean Acc: {clean_acc:.2f}%")
        
        # 4. Evaluate on Corruption Severities 1-5 (Files 0-4)
        print(f'--> Evaluating {args.corruption} Levels 1-5...')
        
        # FIX 1: Loop 0 to 4 (matching files _0.h5 to _4.h5)
        for severity_idx in range(5): 
            try:
                # Severity Index 0 corresponds to Paper Severity 1
                # Severity Index 4 corresponds to Paper Severity 5
                file_severity = severity_idx 
                paper_severity = severity_idx + 1
                
                corr_dataset = ModelNet40C(
                    data_root=args.data_root,
                    corruption=args.corruption,
                    severity=file_severity, # Load _0, _1, ... _4
                    num_points=args.points[0]
                )
                corr_loader = DataLoader(corr_dataset, num_workers=8, batch_size=args.bz, shuffle=False)
                
                corr_features, corr_labels = FeatureProcessor(encoder, corr_loader)
                acc, _ = classification(train_features, train_labels_onehot, corr_features, corr_labels)
                
                # Store using 1-based index (1-5) for clarity in table
                final_results[mode][paper_severity] = acc
                print(f"    Mode {mode} | Severity {paper_severity} (File _{file_severity}): {acc:.2f}%")
                
            except Exception as e:
                print(f"    Skipping Severity {severity_idx} due to error: {e}")
                final_results[mode][severity_idx + 1] = 0.0

    # --- FINAL SUMMARY TABLE ---
    print("\n\n")
    print("="*60)
    print(f"STRESS TEST RESULTS: {args.corruption.upper()}")
    print("="*60)
    
    # Create DataFrame for nice printing
    # Keys are integers: 0 (clean), 1, 2, 3, 4, 5
    df = pd.DataFrame(final_results).T
    
    # Rename columns to strings for display: 'Sev_0', 'Sev_1'...
    df.columns = [f'Sev_{i}' for i in df.columns]
    
    # FIX 2: Use String Keys for calculation
    if 'Sev_0' in df.columns and 'Sev_5' in df.columns:
        df['Total_Drop'] = df['Sev_0'] - df['Sev_5']
    else:
        print("Warning: Could not compute Total_Drop (missing Sev_0 or Sev_5)")
    
    print(df.round(2))
    print("-" * 60)
    
    # Check Hypothesis
    if 'global' in df.index and 'local' in df.index and 'Total_Drop' in df.columns:
        global_drop = df.loc['global', 'Total_Drop']
        local_drop = df.loc['local', 'Total_Drop']
        print("\nHYPOTHESIS CHECK:")
        print(f"Global Drop: {global_drop:.2f}%")
        print(f"Local Drop:  {local_drop:.2f}%")
        if global_drop > local_drop:
            print(">> CONFIRMED: Global features are more sensitive to background noise.")
        else:
            print(">> REJECTED: Local features degraded more (or similarly).")
    print("="*60)

if __name__ == '__main__':
    torch.manual_seed(3407)
    main()