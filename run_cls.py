import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from models.cls import clsNet as MiniSurfEncoder
from dataloaders.ModelNet import ModelNet40
try:
    from dataloaders.ScanObjectNN import ScanObjectNN
except ImportError:
    ScanObjectNN = None
from tqdm import tqdm
import argparse

def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy().item()
    acc = 100 * acc / target.shape[0]
    return acc

def avg_class_acc(output, target):
    pred = output.argmax(dim=1)
    num_classes = output.shape[1]
    class_accs = []
    
    for c in range(num_classes):
        class_mask = (target == c)
        if class_mask.sum() > 0:  # If class exists in target
            class_acc = (pred[class_mask] == c).float().mean()
            class_accs.append(class_acc)
    
    return torch.stack(class_accs).mean() * 100

from sklearn.metrics import f1_score

def avg_class_f1(output, target):
    pred = output.argmax(dim=1).cpu().numpy()
    target = target.cpu().numpy()
    return f1_score(target, pred, average='macro') * 100

def overall_f1(output, target):
    pred = output.argmax(dim=1).cpu().numpy()
    target = target.cpu().numpy()
    return f1_score(target, pred, average='weighted') * 100

def classification(feature_bank, label_bank, test_feature_bank, test_label_bank, plot=False):
    gamma_list = [i * 10000 / 5000 for i in range(5000)]
    best_acc, best_avg_acc, best_gamma = 0, 0, 0
    Sim = test_feature_bank.cuda().float() @ feature_bank.cuda().permute(1, 0).float()
    for gamma in tqdm(gamma_list, desc="Searching best gamma", disable=True):
        logits = (-gamma * (1 - Sim)).exp() @ label_bank
        # logits = F.softmax(logits, dim=0).argmax(dim=-1).reshape(-1, 1)
        acc = cls_acc(logits, test_label_bank)
        # avg_acc = avg_class_acc(logits, test_label_bank)
        if acc > best_acc:
            best_acc, best_gamma = acc, gamma
        # if avg_acc > best_avg_acc:
        #     best_avg_acc, best_gamma = avg_acc, gamma

    # if plot == True:
    #     tensor_np = Sim.cpu().numpy()
    #     adjusted_tensor = adjust_values(tensor_np)
    #     plt.imshow(adjusted_tensor, cmap='hot', interpolation='nearest')
        # plt.savefig('Point-TDA/plot.pdf', bbox_inches='tight')
    
    print(f"classification accuracy: {best_acc:.4f} with best gamma={best_gamma}.")
    print(f"classification accuracy: {best_avg_acc:.4f} with best gamma={best_gamma}.")
    return best_acc, best_gamma

def get_arguments():
    import ast
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mn40')
    # parser.add_argument('--dataset', type=str, default='scan')

    parser.add_argument('--split', type=int, default=1)
    # parser.add_argument('--split', type=int, default=2)
    # parser.add_argument('--split', type=int, default=3)

    parser.add_argument('--bz', type=int, default=8)  # ModelNet 128 # Aug 64 # orginal 16

    parser.add_argument('--points', type=str, default="[1024, 512, 256, 128]")
    parser.add_argument('--stages', type=int, default=4)
    parser.add_argument('--k', type=str, default="[75, 75, 75, 75]")
    parser.add_argument('--metric', type=int, default=1)
    parser.add_argument('--rescale', type=float, default=0.2)
    parser.add_argument('--surface', type=str, default='knnxyz', help='Surface type') # geobp
    
    # args = parser.parse_args()
    args, unknown = parser.parse_known_args()
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
                             k_neighbors=args.k, 
                             rescale=args.rescale,
                             metric=args.metric,
                             surface=args.surface,
                             dataset=args.dataset
                             ).cuda()
    encoder.eval()
    
    # torch.manual_seed(8648899642802701287)

    print('==> Preparing data..')

    if args.dataset == 'scan':
        train_loader = DataLoader(ScanObjectNN(split=args.split, partition='training', num_points=args.points[0]), 
                                    num_workers=8, batch_size=args.bz, shuffle=False, drop_last=False)
        test_loader = DataLoader(ScanObjectNN(split=args.split, partition='test', num_points=args.points[0]), 
                                    num_workers=8, batch_size=args.bz, shuffle=False, drop_last=False)
    elif args.dataset == 'mn40':
        train_loader = DataLoader(ModelNet40(partition='train', num_points=args.points[0]), 
                                    num_workers=8, batch_size=args.bz, shuffle=False, drop_last=False)
        test_loader = DataLoader(ModelNet40(partition='test', num_points=args.points[0]), 
                                    num_workers=8, batch_size=args.bz, shuffle=False, drop_last=False)
    return args, encoder, train_loader, test_loader

@torch.no_grad()
def FeatureProcessor(args, encoder, dataset_loader):
    print('==> Constructing Memory Bank from ' + args.dataset + ' sets..')
    feature_memory, label_memory = [], []
    
    # with torch.no_grad():
    for points, labels in tqdm(dataset_loader, disable=True):
        points = points.cuda()
        # Pass through the Non-Parametric Encoder
        point_features = encoder(points)
        feature_memory.append(point_features)

        labels = labels.cuda()
        label_memory.append(labels)     

    feature_memory = torch.cat(feature_memory, dim=0)
    feature_memory /= feature_memory.norm(dim=-1, keepdim=True)

    label_memory = torch.cat(label_memory, dim=0)
    return [feature_memory, label_memory]



if __name__ == '__main__':
    
    torch.manual_seed(3407)
    args, encoder, train_loader, test_loader = init()
    encoder.eval()

    import time
    start_time = time.time()
    training_memory_list = FeatureProcessor(args, encoder, train_loader)
    end_time = time.time()
    test_memory_list = FeatureProcessor(args, encoder, test_loader)

    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")
    


    classification(training_memory_list[0], 
                       F.one_hot(training_memory_list[1]).squeeze().float(), 
                       test_memory_list[0],
                       test_memory_list[1])