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
    neighbor_2nd = torch.index_select(neighbors, dim=-1, index=torch.tensor(data=[2], dtype=torch.long, device='cuda')) # B,C,N,1
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
        # idx = farthest_point_sample(xyz, self.group_num)
        new_xyz = index_points(xyz, idx)
        xyz = geometric_backpropagation(xyz.permute(0, 2, 1)).permute(0, 2, 1)
        temp_xyz = geometric_backpropagation(new_xyz.permute(0, 2, 1)).permute(0, 2, 1)

        # Perform k-nearest neighbors search
        knn_idx = knn_point(self.kneighbors, xyz, temp_xyz)
        grouped_xyz = index_points(xyz, knn_idx)

        # Compute mean and standard deviation of the grouped points
        mean_xyz = temp_xyz.unsqueeze(dim=-2)
        std_xyz = torch.std(grouped_xyz - mean_xyz, dim=[1,2,3], keepdim=True)
        # print(mean_xyz.shape)
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
        # idx = farthest_point_sample(xyz, self.group_num)
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
            # x = tfcw_from_gram(x)
            x[:, range(x.shape[1] - 1), range(1, x.shape[1])] *= self.rescale
        else:
            x = torch.cdist(x, x, p=self.metric)
        return x


# class tFCW_encoding(nn.Module):
#     """
#     t-FCW encoding with multiple Gram matrix processing strategies.
    
#     Args:
#         metric: Distance metric (2 for Euclidean, 1 for Manhattan)
#         rescale: Alpha value for off-diagonal rescaling (None to disable)
#         variant: Processing strategy for Gram matrix
#             - 'full': Full Gram matrix G
#             - 'no_diag': G - diag(G), zero out diagonal
#             - 'cosine': Cosine similarity normalization
#             - 'tfcw': Our approach (default), pairwise distance from Gram
#     """
#     def __init__(self, metric=2, rescale=None):
#         super().__init__()
#         self.metric = metric
#         self.rescale = rescale
#         # self.variant = variant
#         self.variant = 'tfcw'
#         # assert variant in ['full', 'no_diag', 'cosine', 'tfcw'], \
#         #     f"variant must be one of ['full', 'no_diag', 'cosine', 'tfcw'], got {variant}"

#     def forward(self, x):
#         """
#         Args:
#             x: Input features (B, N, C) or (B*N, C, K) depending on context
        
#         Returns:
#             Processed correlation/distance matrix
#         """
#         # Standardization
#         std_dev = torch.std(x, dim=-2, keepdim=True) + 1e-5
#         x = x / std_dev
        
#         # Compute base Gram matrix
#         if x.dim() == 3:
#             # (B, N, C) -> (B, N, N)
#             G = torch.bmm(x, x.transpose(1, 2))
#         else:
#             # (B*N, C, K) -> (B*N, C, C)
#             G = torch.bmm(x.transpose(1, 2), x)
        
#         # Apply variant-specific processing
#         if self.variant == 'full':
#             output = self._process_full_gram(G)
#         elif self.variant == 'no_diag':
#             output = self._process_no_diag(G)
#         elif self.variant == 'cosine':
#             output = self._process_cosine(G)
#         elif self.variant == 'tfcw':
#             output = self._process_tfcw(G)
        
#         # Apply rescaling if specified
#         if self.rescale is not None:
#             output = self._apply_rescale(output)
        
#         return output
    
#     def _process_full_gram(self, G):
#         """Variant 1: Full Gram matrix (baseline)"""
#         return G
    
#     def _process_no_diag(self, G):
#         """Variant 2: G - diag(G), zero out diagonal elements"""
#         G_no_diag = G.clone()
#         # Create diagonal mask
#         mask = torch.eye(G.size(-1), device=G.device, dtype=torch.bool)
#         if G.dim() == 3:
#             mask = mask.unsqueeze(0)  # (1, N, N)
#         G_no_diag.masked_fill_(mask, 0.0)
#         return G_no_diag
    
#     def _process_cosine(self, G):
#         """Variant 3: Cosine similarity - G_ij / sqrt(G_ii * G_jj)"""
#         # Extract diagonal: G_ii = ||x_i||^2
#         diag_G = torch.diagonal(G, dim1=-2, dim2=-1)  # (B, N) or (N,)
        
#         # Compute sqrt(G_ii) for normalization
#         diag_sqrt = torch.sqrt(torch.clamp(diag_G, min=1e-8))
        
#         # Create normalization matrix: sqrt(G_ii) * sqrt(G_jj)
#         if G.dim() == 3:
#             # (B, N) -> (B, N, N)
#             norm_matrix = diag_sqrt.unsqueeze(-1) * diag_sqrt.unsqueeze(-2)
#         else:
#             # (N,) -> (N, N)
#             norm_matrix = diag_sqrt.unsqueeze(-1) * diag_sqrt.unsqueeze(0)
        
#         # Normalize: G / (sqrt(G_ii) * sqrt(G_jj))
#         G_normalized = G / (norm_matrix + 1e-8)
#         return G_normalized
    
#     def _process_tfcw(self, G):
#         """
#         Variant 4: Our t-FCW approach
#         Compute pairwise distance: D_ij = sqrt(G_ii + G_jj - 2*G_ij)
#         This is equivalent to ||x_i - x_j||_2
#         """
#         # Extract diagonal
#         diag_G = torch.diagonal(G, dim1=-2, dim2=-1)  # (B, N) or (N,)
        
#         # Compute D^2 = G_ii + G_jj - 2*G_ij
#         if G.dim() == 3:
#             # (B, N) -> (B, N, 1) + (B, 1, N) -> (B, N, N)
#             D_squared = diag_G.unsqueeze(-1) + diag_G.unsqueeze(-2) - 2 * G
#         else:
#             # (N,) -> (N, 1) + (1, N) -> (N, N)
#             D_squared = diag_G.unsqueeze(-1) + diag_G.unsqueeze(0) - 2 * G
        
#         # Take sqrt for Euclidean distance
#         D = torch.sqrt(torch.clamp(D_squared, min=0.0))
#         return D
    
#     def _apply_rescale(self, x):
#         """Apply alpha rescaling to off-diagonal elements"""
#         # Rescale off-diagonal stripe: x[:, i, i+1] *= alpha
#         if x.dim() == 3:
#             # Batch processing: (B, N, N)
#             idx = torch.arange(x.shape[1] - 1, device=x.device)
#             x[:, idx, idx + 1] *= self.rescale
#         else:
#             # Single matrix: (N, N)
#             idx = torch.arange(x.shape[0] - 1, device=x.device)
#             x[idx, idx + 1] *= self.rescale
#         return x


# def tfcw_from_gram(x):
#     # x: (B, N, C) where B is batch size, N is number of points, C is feature dim
#     G = torch.bmm(x, x.transpose(1, 2))  # (B, N, N)
#     diag_G = torch.diagonal(G, dim1=1, dim2=2)  # (B, N)

#     # diag_G.unsqueeze(-1): (B, N, 1)
#     # diag_G.unsqueeze(1):  (B, 1, N)
#     # broadcast to (B, N, N)
#     # D_squared = diag_G.unsqueeze(-1) + diag_G.unsqueeze(1) - 2 * G
#     D_squared = G
#     # For numerical stability, clamp negatives to zero before sqrt
#     D = (torch.clamp(D_squared, min=0.0))
#     return D

# diag_G.unsqueeze(-1) + diag_G.unsqueeze(1) - 2 * G-> 84.88 
# diag_G.unsqueeze(-1) + diag_G.unsqueeze(1) - 1 * G-> 82.58
# diag_G.unsqueeze(-1) + diag_G.unsqueeze(1) - 0 * G-> 81.32
# -2*G -> 82.98
# -G -> 82.98



if __name__ == '__main__':
    data = torch.randn(2, 1024, 3).cuda()
    processor = pcs_knnxyz(512, 90)
    new_xyz, knn_xyz = processor(data)
    print(knn_xyz.shape)  # Should be [2, 512, 2, 6]
    knn_xyz = knn_xyz.reshape(2*512, 7, 90)
    encoder = tFCW_encoding()

