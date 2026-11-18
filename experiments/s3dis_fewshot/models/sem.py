import torch
import torch.nn as nn
from .semsurfaces import pcs_geobp, pcs_knnxyz, tFCW_encoding
# from semsurfaces import pcs_geobp, pcs_knnxyz, tFCW_encoding
import torch.nn.functional as F

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
        self.sigmoid = nn.Sigmoid()

        self.tfcw = tFCW_encoding(metric=self.metric, rescale=self.rescale)
        processor = pcs_geobp if surface == 'geobp' else pcs_knnxyz
        self.pc_sampling_list.append(processor(self.input_points[0], self.k_neighbors[0]))
        for i in range(len(self.input_points)):
            sampled_points = self.input_points[i] // 2
            self.pc_sampling_list.append(processor(sampled_points, self.k_neighbors[i]))

    def forward(self, xyz, rgb):
        feature_list = []  # global feature, local feature
        xyz_list = [xyz]
        for i in range(self.num_stages+1):
            # pc, knn, color, knn_color
            xyz, knn_xyz, rgb, knn_rgb = self.pc_sampling_list[i](xyz, rgb)
            if i != 0:
                xyz_list.append(xyz)

            b, n, k, d = knn_xyz.shape

            global_xyz = knn_xyz.permute(0, 3, 1, 2) # b, d, n, k
            global_xyz = pooling(global_xyz)
            knn_xyz = knn_xyz.permute(0, 1, 3, 2)
            knn_xyz = knn_xyz.reshape(-1, d, k) 

            global_rgb = knn_rgb.permute(0, 3, 1, 2) # b, d, n, k
            global_rgb = pooling(global_rgb)
            knn_rgb = knn_rgb.permute(0, 1, 3, 2)
            knn_rgb = knn_rgb.reshape(-1, d, k) 

            # minisurf 
            mini_surf_temp_xyz = self.tfcw(knn_xyz).reshape(b, n, -1) # (-1, d, d) -> # (b, n, d*d)

            mini_surf_temp_rgb = self.tfcw(knn_rgb).reshape(b, n, -1) # (-1, d, d) -> # (b, n, d*d)
            attention_rgb = mini_surf_temp_rgb.reshape(b, n, d, d)
            graph_size = attention_rgb.shape[-1] * attention_rgb.shape[-2] - 1
            attention_graph = (attention_rgb - attention_rgb.mean(dim=[-2, -1], keepdim=True)).pow(2)
            attention_graph = attention_graph / (4 * (attention_graph.sum(dim=[-2, -1], keepdim=True) / graph_size + 1e-4)) + 0.5
            attention_rgb = attention_rgb * self.sigmoid(attention_graph)
            attention_rgb = torch.flatten(attention_rgb, start_dim=2)
            mini_surf_temp = torch.cat([mini_surf_temp_xyz, 
                                        mini_surf_temp_rgb, 
                                        attention_rgb,
                                        global_xyz.permute(0, 2, 1),
                                        global_rgb.permute(0, 2, 1)
                                        ], dim=-1) # b, n, 2*d*d+d+d

            mini_surf_temp = mini_surf_temp.permute(0, 2, 1)
            
            feature_list.append(mini_surf_temp)

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

class SemNet(nn.Module):
        def __init__(self, num_stages, input_points, k_neighbors, rescale, metric, de_neighbors, surface):
            super().__init__()

            self.encoder = Encoder(input_points=input_points, num_stages=num_stages, 
                                               k_neighbors=k_neighbors, rescale=rescale, 
                                               metric=metric, surface = surface)
            
            self.decoder = Decoder(num_stages, de_neighbors)

        def forward(self, x):
            # make sure your x is [b, n, 9]
            # xyz: [b, n, 3], rgb: [b, n, 3], norm: [b, n, 3]
            xyz, rgb, pos = x[..., 0:3], x[..., 3:6], x[..., 6:] 
            xyz_list, minisurf_list = self.encoder(xyz, rgb)
            # xyz token
            # for i in range(len(xyz_list)):
            #     minisurf_list[i] = torch.cat([minisurf_list[i], xyz_list[i].permute(0, 2, 1)], dim=-2)
            x1 = self.decoder(xyz_list, minisurf_list)

            pos_list, MINISURF_list = self.encoder(pos, rgb)
            # pos token
            # for i in range(len(pos_list)):
            #     MINISURF_list[i] = torch.cat([MINISURF_list[i], pos_list[i].permute(0, 2, 1)], dim=-2)
            x2 = self.decoder(pos_list, MINISURF_list)
            return x1, x2 

class FewshotHandler(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_way = args.n_way
        self.k_shot = args.k_shot
        self.dataset = args.dataset
        self.gamma = args.gamma

        if args.dataset == 's3dis':
            self.encoder = SemNet(
                num_stages=args.stages, input_points=args.points, 
                k_neighbors=args.k, rescale=args.rescale, 
                metric=args.metric, de_neighbors=args.de_k,
                surface=args.surface)
        self.encoder.eval()

    def forward(self, support_x, support_y, query_x, query_y):
        
        N_way, K_shot, C, PN = support_x.shape
        
        support_x = support_x.float().cuda().permute(0, 1, 3, 2).view(-1, PN, C)
        support_y = support_y.float().cuda().view(N_way, K_shot, PN)
        query_x = query_x.float().cuda().permute(0, 2, 1).view(-1, PN, C)
        query_y = query_y.cuda().view(-1, PN)
        # Pass through the Non-Parametric Encoder + Decoder
        with torch.no_grad():
            support_features, support_XYZ_features = self.encoder(support_x)
            support_features = support_features.permute(0, 2, 1)  # bz, 2048, c
            support_features = support_features / torch.norm(support_features, dim=-1, keepdim=True)
            
            support_XYZ_features = support_XYZ_features.permute(0, 2, 1)  # bz, 2048, c
            
            feature_memory_list, XYZ_memory_list = [], []
            label_memory_list = []
            support_features, support_XYZ_features = support_features.view(N_way, K_shot, PN, -1), support_XYZ_features.view(N_way, K_shot, PN, -1)
            for i in range(self.n_way):
                mask_fg = (support_y[i] == 1)

                fg_features = support_features[i, mask_fg]
                fg_features = fg_features.mean(0).unsqueeze(0)
                feature_memory_list.append(fg_features)
                label_memory_list.append(torch.tensor(i+1).unsqueeze(0))
                
                fg_XYZ_features = support_XYZ_features[i, mask_fg]
                fg_XYZ_features = fg_XYZ_features.mean(0).unsqueeze(0)
                XYZ_memory_list.append(fg_XYZ_features)

            # Find the point indices for the part_label within a shape
            mask_bg = (support_y == 0)
            bg_features = support_features[mask_bg]
            bg_features = bg_features.mean(0).unsqueeze(0)
            feature_memory_list.append(bg_features)
            label_memory_list.append(torch.tensor(0).unsqueeze(0))
        
            bg_XYZ_features = support_XYZ_features[mask_bg]
            bg_XYZ_features = bg_XYZ_features.mean(0).unsqueeze(0)
            XYZ_memory_list.append(bg_XYZ_features)

            feature_memory = torch.cat(feature_memory_list, dim=0)
            XYZ_memory = torch.cat(XYZ_memory_list, dim=0)
            
            label_memory = torch.cat(label_memory_list, dim=0).cuda()
            label_memory = F.one_hot(label_memory, num_classes=self.n_way+1)

            feature_memory = feature_memory / torch.norm(feature_memory, dim=-1, keepdim=True)
            feature_memory = feature_memory.permute(1, 0)
            XYZ_memory = XYZ_memory / torch.norm(XYZ_memory, dim=-1, keepdim=True)
            XYZ_memory = XYZ_memory.permute(1, 0)
            
            query_features, query_XYZ_features = self.encoder(query_x)
            query_features = query_features.permute(0, 2, 1)  # bz, 2048, c
            query_features /= query_features.norm(dim=-1, keepdim=True)
            
            query_XYZ_features = query_XYZ_features.permute(0, 2, 1)  # bz, 2048, c
            query_XYZ_features /= query_XYZ_features.norm(dim=-1, keepdim=True)
        
            Sim = query_features @ feature_memory
            Sim_XYZ = query_XYZ_features @ XYZ_memory
            
            if self.dataset == 's3dis':
                logits = (-self.gamma * (1 - Sim)).exp() @ label_memory.float()
                logits_XYZ = (-self.gamma * (1 - Sim_XYZ)).exp() @ label_memory.float()
            elif self.dataset == 'scannet':
                logits = (-self.gamma * (1 - Sim)).exp() @ label_memory.float()
                logits_XYZ = (-self.gamma * (1 - Sim_XYZ)).exp() @ label_memory.float()
            logits = logits + logits_XYZ
            
            logits = F.softmax(logits, dim=-1)
                        
        return logits, 0

if __name__ == '__main__':
    import argparse
    import ast
    def get_arguments():
        # for point-nn
        parser = argparse.ArgumentParser()
        #few-shot episode setting
        parser.add_argument('--n_way', type=int, default=2, help='Number of classes for each episode: 2|3')
        parser.add_argument('--k_shot', type=int, default=1, help='Number of samples/shots for each class: 1|5')
        parser.add_argument('--n_queries', type=int, default=1, help='Number of queries for each class')
        parser.add_argument('--n_episode_test', type=int, default=100,
                            help='Number of episode per configuration during testing')
        parser.add_argument('--dataset', type=str, default='s3dis')  # 71.27, 73.95

        parser.add_argument('--bz', type=int, default=64)  # Freeze as 1

        parser.add_argument('--points', type=str, default="[1024, 512, 256, 128]")
        parser.add_argument('--stages', type=int, default=4)
        parser.add_argument('--k', type=str, default="[75, 75, 75, 75]")
        parser.add_argument('--de_k', type=int, default=3)  # propagate neighbors in decoder

        parser.add_argument('--metric', type=int, default=2)
        parser.add_argument('--rescale', type=float, default=0.3)
        parser.add_argument('--gamma', type=int, default=230)  # Best as 230
        parser.add_argument('--surface', type=str, default='knnxyz', help='Surface type')

        args = parser.parse_args()
        args.points = ast.literal_eval(args.points)
        args.k = ast.literal_eval(args.k)
        return args
    args = get_arguments()
    sx = torch.rand(2, 1, 9, 1024).cuda()
    sy = torch.rand(2, 1, 1024).cuda()
    qx = torch.rand(2, 9, 1024).cuda()
    qy = torch.rand(2, 1024).cuda()
    model = FewshotHandler(args).cuda()

    x1, _ = model(sx, sy, qx, qy)

    print(x1.shape)

