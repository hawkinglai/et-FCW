import ast
import argparse
import time
import numpy as np
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F

from dataloaders.loader import MyTestDataset, batch_task_collate
from models.sem import FewshotHandler as SemNet

def cast_cuda(input):
    if type(input) == type([]):
        for i in range(len(input)):
            input[i] = cast_cuda(input[i])
    else:
        return input.cuda()
    return input

def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_points', type=str, default="[2048, 1024, 512, 256, 128]")
    parser.add_argument('--k_neighbors', type=str, default="[30, 30, 30, 30, 30]")
    parser.add_argument('--metric', type=int, default=2)
    parser.add_argument('--rescale', type=float, default=0.3)


    #data
    parser.add_argument('--model', type=str, default='minisurf')
    parser.add_argument('--stages', type=int, default=4)
    parser.add_argument('--de_k', type=int, default=20)
    parser.add_argument('--surface', type=str, default='knnxyz', choices=['knnxyz', 'geobp'])
    parser.add_argument('--dataset', type=str, default='s3dis', help='Dataset name: s3dis|scannet')
    parser.add_argument('--cvfold', type=int, default=0, help='Fold left-out for testing in leave-one-out setting ''Options:{0,1}')
    parser.add_argument('--data_path', type=str, default='dataloaders/data/S3DIS/scenes/blocks_bs1_s1/', help='Directory to the source data')
    parser.add_argument('--save_path', type=str, default='./log_s3dis/', help='Directory to save log and checkpoints')

    #optimization
    parser.add_argument('--batch_size', type=int, default=1, help='Number of samples/tasks in one batch')
    parser.add_argument('--n_workers', type=int, default=8, help='number of workers to load data')

    #few-shot episode setting
    parser.add_argument('--n_way', type=int, default=2, help='Number of classes for each episode: 2|3')
    parser.add_argument('--k_shot', type=int, default=1, help='Number of samples/shots for each class: 1|5')
    parser.add_argument('--n_queries', type=int, default=1, help='Number of queries for each class')
    parser.add_argument('--n_episode_test', type=int, default=100,
                        help='Number of episode per configuration during testing')

    # Point cloud processing
    parser.add_argument('--pc_attribs', default='xyzrgbXYZ',
                        help='Point attributes fed to PointNets, if empty then all possible. '
                             'xyz = coordinates, rgb = color, XYZ = normalized xyz')           
    parser.add_argument('--way_pcratio', default='[0.05, 0.05]', 
                        help='The least ratio of points for each target class in the point cloud.')
    parser.add_argument('--way_pcnum', default='[300, 300]', 
                        help='The least number of points for each target class in the point cloud.')
    parser.add_argument('--gamma', type=int, default=100)  # Best as 230

    args = parser.parse_args()
    print('Dataset:  ', args.dataset)
    print('CV Fold:  ', args.cvfold)
    print('Num Way:  ', args.n_way)
    print('Num Shot: ', args.k_shot)
    print('PC Ratio: ', args.way_pcratio, args.way_pcnum)

    args.way_pcratio = ast.literal_eval(args.way_pcratio)
    args.way_pcnum = ast.literal_eval(args.way_pcnum)
    args.points = ast.literal_eval(args.input_points)
    args.k = ast.literal_eval(args.k_neighbors)

    
    return args


class model_processor(object):
    def __init__(self, args):
        self.model = SemNet(args)
        self.model.cuda()

    def test(self, data):
        # Args:
        #     support_x: support point clouds with shape (n_way, k_shot, in_channels, num_points)
        #     support_y: support masks (foreground) with shape (n_way, k_shot, num_points), each point \in {0,1}.
        #     query_x: query point clouds with shape (n_queries, in_channels, num_points)
        #     query_y: query labels with shape (n_queries, num_points), each point \in {0,..., n_way}
        self.model.eval()
        [support_x, support_y, query_x, query_y] = data
        
        with torch.no_grad():
            logits, _ = self.model(support_x, support_y, query_x, query_y)
            
            pred = F.softmax(logits, dim=-1).argmax(dim=-1)
            correct = torch.eq(pred, query_y).sum().item()
            accuracy = correct / (query_y.shape[0] * query_y.shape[1])

        return pred, accuracy


def evaluate_metric(pred_labels_list, gt_labels_list, label2class_list, test_classes):
    """
    :param pred_labels_list: a list of np array, each entry with shape (n_queries*n_way, num_points).
    :param gt_labels_list: a list of np array, each entry with shape (n_queries*n_way, num_points).
    :param test_classes: a list of np array, each entry with shape (n_way,)
    :return: iou: scaler
    """
    assert len(pred_labels_list) == len(gt_labels_list) == len(label2class_list)
    
    print('*****Test Classes: {0}*****'.format(test_classes))

    NUM_CLASS = len(test_classes) + 1  # add 1 to consider background class
    gt_classes = [0 for _ in range(NUM_CLASS)]
    positive_classes = [0 for _ in range(NUM_CLASS)]
    true_positive_classes = [0 for _ in range(NUM_CLASS)]

    for i, batch_gt_labels in enumerate(gt_labels_list):
        batch_pred_labels = pred_labels_list[i]  # (n_queries*n_way, num_points)
        label2class = label2class_list[i]  # (n_way,)
        for j in range(batch_pred_labels.shape[0]):
            for k in range(batch_pred_labels.shape[1]):
                gt = int(batch_gt_labels[j, k])
                pred = int(batch_pred_labels[j, k])

                if gt == 0:  # 0 indicate background class
                    gt_index = 0
                else:
                    gt_class = label2class[gt - 1]  # the ground truth class in the dataset
                    gt_index = test_classes.index(gt_class) + 1
                gt_classes[gt_index] += 1

                if pred == 0:
                    pred_index = 0
                else:
                    pred_class = label2class[pred - 1]
                    pred_index = test_classes.index(pred_class) + 1
                positive_classes[pred_index] += 1

                true_positive_classes[gt_index] += int(gt == pred)

    iou_list = []
    for c in range(NUM_CLASS):
        iou = true_positive_classes[c] / float(gt_classes[c] + positive_classes[c] - true_positive_classes[c] + 0.001)
        iou_list.append(iou)

    mean_IoU = np.array(iou_list[1:]).mean()

    return mean_IoU


def test_few_shot(test_loader, learner, test_classes):
    
    predicted_label_total = []
    gt_label_total = []
    label2class_total = []
    all_acc = []
    start_time = time.time()
    for batch_idx, (data, sampled_classes) in enumerate(test_loader):
        
        query_label = data[-1]
        
        data = cast_cuda(data)
        
        query_pred, acc = learner.test(data)
            
        if (batch_idx + 1) % 500 == 0:
            print('[Eval] Iter: %d | Acc: %.4f | %s' % (batch_idx + 1, acc, str(datetime.now())))

        predicted_label_total.append(query_pred.cpu().detach().numpy())
        gt_label_total.append(query_label.numpy())
        label2class_total.append(sampled_classes)
        all_acc.append(acc)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")
       
    mean_IoU = evaluate_metric(predicted_label_total, gt_label_total, label2class_total, test_classes)
    return mean_IoU


def training_free(args):
    print(args)
    learner = model_processor(args)
        
    # Init dataset, dataloader
    TEST_DATASET = MyTestDataset(args.model, args.data_path, args.dataset, cvfold=args.cvfold,
                                 num_episode_per_comb=args.n_episode_test,
                                 n_way=args.n_way, k_shot=args.k_shot, n_queries=args.n_queries,
                                 num_point=args.points, pc_attribs=args.pc_attribs, 
                                 way_ratio=args.way_pcratio, way_num=args.way_pcnum, mode='test')
    TEST_CLASSES = list(TEST_DATASET.classes)
    TEST_LOADER = DataLoader(TEST_DATASET, batch_size=1, shuffle=False, collate_fn=batch_task_collate)

    mean_IoU = test_few_shot(TEST_LOADER, learner, TEST_CLASSES)
    print('\n=====[TEST] Mean IoU: %f =====\n' % (mean_IoU))


if __name__ == '__main__':
    args = get_arguments()
    training_free(args)
