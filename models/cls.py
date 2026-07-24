import torch
import torch.nn as nn
from .surfaces import pcs_geobp, pcs_knnxyz, tFCW_encoding


def pooling(x=None):
    x = x.max(-1)[0] + x.mean(-1)
    return x

class clsNet(nn.Module):
    def __init__(self, input_points, num_stages, k_neighbors, rescale, metric, surface, dataset='mn40'):
        super().__init__()
        self.input_points = input_points
        self.num_stages = num_stages
        self.k_neighbors = k_neighbors
        self.rescale = rescale
        self.metric = metric
        self.pc_sampling_list = nn.ModuleList()
        self.dataset = dataset
        self.tfcw = tFCW_encoding(metric=self.metric, rescale=self.rescale)
        self.sigmoid = nn.Sigmoid()

        for i in range(len(self.input_points)):
            sampled_points = self.input_points[i] // 2
            processor = pcs_geobp if surface == 'geobp' else pcs_knnxyz
            self.pc_sampling_list.append(processor(sampled_points, self.k_neighbors[i]))

    def forward(self, xyz):
        feature_list = [] # global feature, local feature
        for i in range(self.num_stages):
            # pc, knn
            xyz, knn_xyz = self.pc_sampling_list[i](xyz)
            b, n, k, d = knn_xyz.shape

            # global tfcw
            global_xyz = knn_xyz.permute(0, 3, 1, 2)
            global_xyz = pooling(global_xyz)
            global_xyz = self.tfcw(global_xyz) # b, n, d, d
            global_xyz = torch.flatten(global_xyz, start_dim=1)
            feature_list.append(global_xyz)

            # local tfcw
            local_xyz = knn_xyz.permute(0, 1, 3, 2)
            local_xyz = local_xyz.reshape(-1, d, k)
            local_xyz = self.tfcw(local_xyz) # b*n, d, d

            if self.dataset == 'mn40':
                attention_xyz = local_xyz.reshape(b, n, d, d)
                graph_size = attention_xyz.shape[-1] * attention_xyz.shape[-2] - 1
                attention_graph = (attention_xyz - attention_xyz.mean(dim=[-2, -1], keepdim=True)).pow(2)
                attention_graph = attention_graph / (4 * (attention_graph.sum(dim=[-2, -1], keepdim=True) / graph_size + 1e-4)) + 0.5
                attention_xyz = attention_xyz * self.sigmoid(attention_graph)
                local_xyz = torch.flatten(attention_xyz, start_dim=2).permute(0, 2, 1)
            else:
                local_xyz = torch.flatten(local_xyz, start_dim=1).reshape(b, n, -1).permute(0, 2, 1)
            local_xyz = pooling(local_xyz)
            feature_list.append(local_xyz)


        output = torch.cat(feature_list, dim=-1)
        output /= output.norm(dim=-1, keepdim=True)
        return output


if __name__ == '__main__':
    # surface = 'geobp'  # or 'knnxyz'
    surface = 'knnxyz'
    data = torch.randn(2, 1024, 3).cuda()
    model = clsNet(input_points=[1024, 512, 256, 128], num_stages=4, k_neighbors=[90, 45, 22, 11], rescale=0.1, metric=2, surface=surface).cuda()
    model = model.eval()
    output = model(data)

    perm = torch.randperm(1024)
    data2 = data[:, perm, :]
    output2 = model(data2)
    max_diff = (output - output2).abs().max().item()
    print("Max abs diff after sorting:", max_diff)
