# Third-party portions retain the licenses listed below.
# Modified by the et-FCW authors.
# Origin: https://github.com/ZrrSkywalker/Point-NN
# Copyright (c) 2023 Renrui Zhang
# License: MIT; see ../../third_party/licenses/Point-NN.txt
# The et-FCW additions are covered by the repository MIT license.

import os
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from models.ri_cls import clsNet as MiniSurfEncoder
from dataloaders.ModelNet_ri import ModelNetDataLoader
from tqdm import tqdm
import argparse
from pytorch3d.transforms import RotateAxisAngle, Rotate, random_rotations

def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy().item()
    acc = 100 * acc / target.shape[0]
    return acc


def classification(feature_bank, label_bank, test_feature_bank, test_label_bank, plot=False):
    gamma_list = [i * 10000 / 5000 for i in range(5000)]
    best_acc, best_gamma = 0, 0
    Sim = test_feature_bank.cuda().float() @ feature_bank.cuda().permute(1, 0).float()
    for gamma in tqdm(gamma_list, desc="Searching best gamma"):
        logits = (-gamma * (1 - Sim)).exp() @ label_bank
        # logits = F.softmax(logits, dim=0).argmax(dim=-1).reshape(-1, 1)
        acc = cls_acc(logits, test_label_bank)

        if acc > best_acc:
            best_acc, best_gamma = acc, gamma

    print(f"classification accuracy: {best_acc:.4f} with best gamma={best_gamma}.")
    return best_acc, best_gamma

def get_arguments():
    import ast
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mn40')
    # parser.add_argument('--dataset', type=str, default='scan')

    # parser.add_argument('--split', type=int, default=1)
    # parser.add_argument('--split', type=int, default=2)
    parser.add_argument('--split', type=int, default=3)

    parser.add_argument('--bz', type=int, default=128)  # ModelNet 128 # Aug 64 # orginal 16

    parser.add_argument('--points', type=str, default="[1024, 512, 256, 128]")
    parser.add_argument('--stages', type=int, default=4)
    # parser.add_argument('--k', type=int, default=90)
    parser.add_argument('--k', type=str, default="[90, 90, 90, 90]")
    parser.add_argument('--metric', type=int, default=2)
    parser.add_argument('--rescale', type=float, default=0.8)
    parser.add_argument('--surface', type=str, default='knnxyz', help='Surface type') # geobp
    # for ri_surface
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--use_normals', type=bool, default=True, help='use normals')
    parser.add_argument('--process_data', type=bool, default=True, help='save data offline')
    parser.add_argument('--use_uniform_sample', type=bool, default=True, help='use uniform sampiling')
    args = parser.parse_args()
    args.points = ast.literal_eval(args.points)
    args.k = ast.literal_eval(args.k)
    return args

@torch.no_grad()
def init():
    print('==> Loading args..')
    args = get_arguments()
    print(args)

    print('==> Preparing model..')
    encoder = MiniSurfEncoder(input_points=args.points, 
                             num_stages=args.stages, 
                             n_samples=args.k, 
                             rescale=args.rescale,
                             surface=args.surface,
                             metric=args.metric
                             ).cuda()
    encoder.eval()
    
    # torch.manual_seed(8648899642802701287)

    print('==> Preparing data..')
    data_path = os.environ['TFCW_MODELNET_NORMAL_ROOT']
    train_dataset = ModelNetDataLoader(root=data_path, args=args, split='train', process_data=args.process_data)
    test_dataset = ModelNetDataLoader(root=data_path, args=args, split='test', process_data=args.process_data)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bz, shuffle=False, num_workers=10, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.bz, shuffle=False, num_workers=10)

    return args, encoder, train_loader, test_loader

@torch.no_grad()
def FeatureProcessor(args, encoder, dataset_loader, rot=None):
    print('==> Constructing Memory Bank from ' + args.dataset + ' sets..')
    feature_memory, label_memory = [], []
    
    # with torch.no_grad():
    for points, labels in tqdm(dataset_loader):
        points = points.cuda()
        trot = None
        if rot == 'z':
            trot = RotateAxisAngle(angle=torch.rand(points.shape[0]) * 360, axis="Z", degrees=True, device='cuda:0')
        elif rot == 'so3':
            trot = Rotate(R=random_rotations(points.shape[0]), device='cuda:0')
        if trot is not None:
            xyz = trot.transform_points(points[:, :, :3])
            norm = trot.transform_normals(points[:, :, 3:])
        else:
            xyz = points[:, :, :3]
            norm = points[:, :, 3:]
        point_features = encoder(xyz, norm)
        feature_memory.append(point_features)

        labels = labels.cuda().long()
        label_memory.append(labels)     

    feature_memory = torch.cat(feature_memory, dim=0)
    feature_memory /= feature_memory.norm(dim=-1, keepdim=True)

    label_memory = torch.cat(label_memory, dim=0)
    return [feature_memory, label_memory]



if __name__ == '__main__':
    
    torch.manual_seed(3407)
    args, encoder, train_loader, test_loader = init()
    encoder.eval()

    # import time
    # start_time = time.time()

    list = [['z', 'z'], ['z', 'so3'], ['so3', 'so3'], ['none', 'none']]

    for l in list:
        training_memory_list = FeatureProcessor(args, encoder, train_loader, rot=l[0])
        test_memory_list = FeatureProcessor(args, encoder, test_loader, rot=l[1])
        print("Training labels shape:", training_memory_list[1].shape)
        print("Training labels type:", training_memory_list[1].dtype)
        print("Training labels sample:", training_memory_list[1][:10])
        classification(training_memory_list[0], 
                        F.one_hot(training_memory_list[1]).squeeze().float(), 
                        test_memory_list[0],
                        test_memory_list[1])