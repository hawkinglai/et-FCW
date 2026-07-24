import torch
import torch.nn as nn
from pointnet2_ops import pointnet2_utils
import torch.nn.functional as F

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids

def index_points(points, idx):
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

def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx

def square_distance(src, dst):
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

def geometric_backpropagation(x, k=3, idx=None):
    """
    Geometric Backpropagation
    Geometric Back-projection Network for Point Cloud Classification (IEEE Transactions on Multimedia, TMM 2021)
    """
    # x: B,3,N
    batch_size = x.size(0)
    num_points = x.size(2)
    org_x = x
    x = x.view(batch_size, -1, num_points)
    def knn(x, k):
        inner = -2*torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x**2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)
    
        idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
        return idx
    
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
    idx_base = idx_base.type(torch.cuda.LongTensor)
    idx = idx.type(torch.cuda.LongTensor)
    idx = idx + idx_base
    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    neighbors = x.view(batch_size * num_points, -1)[idx, :]
    neighbors = neighbors.view(batch_size, num_points, k, num_dims)

    neighbors = neighbors.permute(0, 3, 1, 2)  # B,C,N,k
    neighbor_1st = torch.index_select(neighbors, dim=-1, index=torch.tensor(data=[1], dtype=torch.long, device='cuda')) # B,C,N,1
    neighbor_1st = torch.squeeze(neighbor_1st, -1)  # B,3,N
    neighbor_2nd = torch.index_select(neighbors, dim=-1, index=torch.tensor(data=[1], dtype=torch.long, device='cuda')) # B,C,N,1
    neighbor_2nd = torch.squeeze(neighbor_2nd, -1)  # B,3,N

    edge1 = neighbor_1st-org_x
    edge2 = neighbor_2nd-org_x
    normals = torch.cross(edge1, edge2, dim=1) # B,3,N
    dist1 = torch.norm(edge1, dim=1, keepdim=True) # B,1,N
    dist2 = torch.norm(edge2, dim=1, keepdim=True) # B,1,N

    new_pts = torch.cat((org_x, normals, dist1, dist2, edge1, edge2), 1) # B,14,N

    return new_pts


class pcs_geobp(nn.Module):
    def __init__(self, group_num, kneighbors):
        super().__init__()
        self.group_num = group_num
        self.kneighbors = kneighbors
    
    def forward(self, xyz):
        B, N, C = xyz.shape
        xyz = xyz.cuda()
        
        # Perform farthest point sampling
        idx = pointnet2_utils.furthest_point_sample(xyz, self.group_num).long() 
        new_xyz = index_points(xyz, idx)
        xyz = geometric_backpropagation(xyz.permute(0, 2, 1)).permute(0, 2, 1)
        temp_xyz = geometric_backpropagation(new_xyz.permute(0, 2, 1)).permute(0, 2, 1)

        # Perform k-nearest neighbors search
        knn_idx = knn_point(self.kneighbors, xyz, temp_xyz)
        grouped_xyz = index_points(xyz, knn_idx)

        # Compute mean and standard deviation of the grouped points
        mean_xyz = temp_xyz.unsqueeze(dim=-2)
        std_xyz = torch.std(grouped_xyz - mean_xyz, dim=[1,2,3], keepdim=True)
        # Normalize the points
        knn_xyz = (grouped_xyz - mean_xyz) / (std_xyz + 1e-5)
        
        pairwise_dists = torch.cdist(grouped_xyz, mean_xyz, p=2)  # [B, N, K, 1]
        # Concatenate the normalized points with the new centroids
        knn_xyz = torch.cat([knn_xyz, temp_xyz.view(B, self.group_num, 1, -1).repeat(1, 1, self.kneighbors, 1),
                             pairwise_dists], dim=-1)
        
        return new_xyz, knn_xyz

    
class pcs_knnxyz(nn.Module):
    def __init__(self, group_num, kneighbors):
        super().__init__()
        self.group_num = group_num
        self.kneighbors = kneighbors
    
    def forward(self, xyz):
        B, N, C = xyz.shape
        xyz = xyz.cuda()
        idx = pointnet2_utils.furthest_point_sample(xyz, self.group_num).long() 
        new_xyz = index_points(xyz, idx)

        knn_idx = knn_point(self.kneighbors, xyz, new_xyz)
        grouped_xyz = index_points(xyz, knn_idx)

        mean_xyz = new_xyz.unsqueeze(dim=-2)
        std_xyz = torch.std(grouped_xyz - mean_xyz, keepdim=True)

        
        knn_xyz = (grouped_xyz - mean_xyz) / (std_xyz + 1e-5)
        pairwise_dists = torch.cdist(grouped_xyz, mean_xyz, p=2)  # [B, N, K, 1]

        knn_xyz = torch.cat([knn_xyz, 
                             new_xyz.view(B, self.group_num, 1, -1).repeat(1, 1, self.kneighbors, 1),
                             pairwise_dists
                             ], dim=-1)
        
        return new_xyz, knn_xyz
    
class tFCW_encoding(nn.Module):
    def __init__(self, metric=2, rescale=None):
        super().__init__()
        self.metric = metric
        self.rescale = rescale

    def forward(self, x):
        std_dev = torch.std(x, dim=-2, keepdim=True) + 1e-5
        x /= std_dev
        if self.rescale is not None:
            x = torch.cdist(x, x, p=self.metric)
            x[:, range(x.shape[1] - 1), range(1, x.shape[1])] *= self.rescale
        else:
            x = torch.cdist(x, x, p=self.metric)
        return x

if __name__ == '__main__':
    data = torch.randn(2, 1024, 3).cuda()
    processor = pcs_knnxyz(512, 90)
    new_xyz, knn_xyz = processor(data)
    print(knn_xyz.shape)  # Should be [2, 512, 2, 6]
    knn_xyz = knn_xyz.reshape(2*512, 7, 90)
    encoder = tFCW_encoding()

