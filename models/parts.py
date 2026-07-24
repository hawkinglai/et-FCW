import torch
import torch.nn as nn
from .surfaces import pcs_geobp, pcs_knnxyz, tFCW_encoding

def pooling(x=None):
    x = x.max(-1)[0] + x.mean(-1)
    return x

class Encoder(nn.Module):
    def __init__(self, input_points, num_stages, k_neighbors, rescale, metric, surface):
        super().__init__()
        self.input_points = input_points
        self.num_stages = num_stages
        self.k_neighbors = k_neighbors
        self.rescale = rescale
        self.metric = metric
        self.pc_sampling_list = nn.ModuleList()

        self.tfcw = tFCW_encoding(metric=self.metric, rescale=self.rescale)
        processor = pcs_geobp if surface == 'geobp' else pcs_knnxyz
        self.pc_sampling_list.append(processor(self.input_points[0], self.k_neighbors[0]))
        self.sigmoid = nn.Sigmoid()
        for i in range(len(self.input_points)):
            sampled_points = self.input_points[i] // 2
            self.pc_sampling_list.append(processor(sampled_points, self.k_neighbors[i]))

    def forward(self, xyz):
        feature_list = [] # global feature, local feature
        xyz_list = [xyz]
        for i in range(self.num_stages + 1):
            # pc, knn
            xyz, knn_xyz = self.pc_sampling_list[i](xyz)
            if i != 0:
                xyz_list.append(xyz)

            b, n, k, d = knn_xyz.shape       

            # global feature
            global_xyz = knn_xyz.permute(0, 3, 1, 2)
            global_xyz = pooling(global_xyz)

            knn_xyz = knn_xyz.permute(0, 1, 3, 2)
            knn_xyz = knn_xyz.reshape(-1, d, k)
            # local tfcw
            local_xyz = self.tfcw(knn_xyz)
            # attention_xyz = local_xyz.reshape(b, n, d, d)
            # graph_size = attention_xyz.shape[-1] * attention_xyz.shape[-2] -1
            # attention_graph = (attention_xyz - attention_xyz.mean(dim=[-2,-1], keepdim=True)).pow(2)
            # attention_graph = attention_graph / (4 * (attention_graph.sum(dim=[-2,-1], keepdim=True) / graph_size + 1e-4)) + 0.5
            # attention_xyz = attention_xyz * self.sigmoid(attention_graph)
            # attention_xyz = torch.flatten(attention_xyz, start_dim=2).permute(0, 2, 1)
            local_xyz = local_xyz.reshape(b, n, -1).permute(0, 2, 1)
            local_xyz = torch.cat([local_xyz, global_xyz], dim=-2)
            feature_list.append(local_xyz)
        return xyz_list, feature_list
    
class Decoder(nn.Module):
    def __init__(self, num_stages, de_neighbors):
        super().__init__()
        self.num_stages = num_stages
        self.de_neighbors = de_neighbors

    def propagate(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, N, 3]
            xyz2: sampled input points position data, [B, S, 3]
            points1: input points data, [B, D', N]
            points2: input points data, [B, D'', S]
        Return:
            new_points: upsampled points data, [B, D''', N]
        """

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = self.square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :self.de_neighbors], idx[:, :, :self.de_neighbors]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            weight = weight.view(B, N, self.de_neighbors, 1)

            self.index_points(xyz1, idx)
            interpolated_points = torch.sum(self.index_points(points2, idx) * weight, dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)

        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        return new_points
    
    def square_distance(self, src, dst):
        """
        Calculate Euclid distance between each two points.
        src^T * dst = xn * xm + yn * ym + zn * zm；
        sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
        sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
        dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
            = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
        Input:
            src: source points, [B, N, C]
            dst: target points, [B, M, C]
        Output:
            dist: per-point square distance, [B, N, M]
        """
        B, N, _ = src.shape
        _, M, _ = dst.shape
        dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
        dist += torch.sum(src ** 2, -1).view(B, N, 1)
        dist += torch.sum(dst ** 2, -1).view(B, 1, M)
        return dist
    
    def index_points(self, points, idx):
        """
        index_point with rest point version.
        Input:
            points: input points data, [B, N, C]
            idx: sample index data, [B, S]
        Return:
            new_points:, indexed points data, [B, S, C]
            rest_points:, remaining points data, [B, N-S, C]
        """
        device = points.device
        B = points.shape[0]
        view_shape = list(idx.shape)
        view_shape[1:] = [1] * (len(view_shape) - 1)
        repeat_shape = list(idx.shape)
        repeat_shape[0] = 1
        batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
        new_points = points[batch_indices, idx, :]
        return new_points
    
    def forward(self, xyz_list, x_list):
        xyz_list.reverse()
        x_list.reverse()

        x = x_list[0]
        for i in range(self.num_stages):
            # Propagate point features to neighbors
            x = self.propagate(xyz_list[i+1], xyz_list[i], x_list[i+1], x)
        return x
    
class PartNet(nn.Module):
    def __init__(self, input_points, num_stages, k_neighbors, rescale, metric, surface, de_neighbors):
        super().__init__()
        self.encoder = Encoder(input_points, num_stages, k_neighbors, rescale, metric, surface)
        self.decoder = Decoder(num_stages, de_neighbors)

    def forward(self, xyz):
        xyz_list, feature_list = self.encoder(xyz)

        for i in range(len(xyz_list)):
            feature_list[i] = torch.cat([feature_list[i], xyz_list[i].permute(0, 2, 1)], dim=-2)
        output = self.decoder(xyz_list, feature_list)
        return output
    
if __name__ == '__main__':
    def get_arguments():
        import ast
        import argparse
        parser = argparse.ArgumentParser()
        # parser.add_argument('--dataset', type=str, default='mn40')
        parser.add_argument('--dataset', type=str, default='scan')

        parser.add_argument('--split', type=int, default=1)
        # parser.add_argument('--split', type=int, default=2)
        # parser.add_argument('--split', type=int, default=3)

        parser.add_argument('--bz', type=int, default=8)  # ModelNet 128 # Aug 64 # orginal 16

        parser.add_argument('--points', type=str, default="[1024, 512, 256, 128]")
        parser.add_argument('--stages', type=int, default=4)
        parser.add_argument('--k', type=str, default="[75, 75, 75, 75]")
        parser.add_argument('--metric', type=int, default=1)
        # parser.add_argument('--metric', type=str, default='seuclidean')
        parser.add_argument('--rescale', type=float, default=0.2)
        parser.add_argument('--surface', type=str, default='knnxyz', help='Surface type') # geobp
        
        args = parser.parse_args()
        args.points = ast.literal_eval(args.points)
        args.k = ast.literal_eval(args.k)
        return args
    args = get_arguments()
    data = torch.rand(32, 1024, 3).cuda()
    model = PartNet(
        input_points=args.points,
        num_stages=args.stages,
        k_neighbors=args.k,
        rescale=args.rescale,
        metric=args.metric,
        surface=args.surface,
        de_neighbors=6  # Assuming a fixed value for de_neighbors
    ).cuda()
    print(args)
    print(model(data).shape)
