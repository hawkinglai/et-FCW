import torch
import torch.nn as nn
from .surfaces import pcs_geobp, pcs_knnxyz, tFCW_encoding

def pooling(x=None):
    x = x.max(-1)[0] + x.mean(-1)
    return x

class clsNet(nn.Module):
    def __init__(self, input_points, num_stages, k_neighbors, rescale, metric, surface, dataset='mn40', mode='united'):
        """
        Args:
            mode (str): 'global', 'local', or 'united'. Controls which features are extracted.
        """
        super().__init__()
        self.input_points = input_points
        self.num_stages = num_stages
        self.k_neighbors = k_neighbors
        self.rescale = rescale
        self.metric = metric
        self.mode = mode  # <--- Store the mode
        self.pc_sampling_list = nn.ModuleList()
        self.dataset = dataset
        self.tfcw = tFCW_encoding(metric=self.metric, rescale=self.rescale)
        self.sigmoid = nn.Sigmoid()

        for i in range(len(self.input_points)):
            sampled_points = self.input_points[i] // 2
            processor = pcs_geobp if surface == 'geobp' else pcs_knnxyz
            self.pc_sampling_list.append(processor(sampled_points, self.k_neighbors[i]))

    def forward(self, xyz):
        feature_list = [] 
        
        for i in range(self.num_stages):
            # 1. Sampling & k-NN grouping
            xyz, knn_xyz = self.pc_sampling_list[i](xyz)
            b, n, k, d = knn_xyz.shape

            # ---------------------------------------------------------
            # GLOBAL BRANCH (Algorithm 1 / Standard t-FCW)
            # ---------------------------------------------------------
            if self.mode in ['global', 'united']:
                global_xyz = knn_xyz.permute(0, 3, 1, 2)
                global_xyz = pooling(global_xyz)
                global_xyz = self.tfcw(global_xyz) # b, n, d, d
                global_xyz = torch.flatten(global_xyz, start_dim=1)
                feature_list.append(global_xyz)

            # ---------------------------------------------------------
            # LOCAL BRANCH (Algorithm 2 / Empowered t-FCW)
            # ---------------------------------------------------------
            if self.mode in ['local', 'united']:
                local_xyz = knn_xyz.permute(0, 1, 3, 2)
                local_xyz = local_xyz.reshape(-1, d, k)
                local_xyz = self.tfcw(local_xyz) # b*n, d, d

                if self.dataset == 'mn40':
                    # Attention mechanism for ModelNet40
                    attention_xyz = local_xyz.reshape(b, n, d, d)
                    graph_size = attention_xyz.shape[-1] * attention_xyz.shape[-2] - 1
                    
                    # Compute attention weights based on variance from mean
                    attention_graph = (attention_xyz - attention_xyz.mean(dim=[-2, -1], keepdim=True)).pow(2)
                    attention_graph = attention_graph / (4 * (attention_graph.sum(dim=[-2, -1], keepdim=True) / graph_size + 1e-4)) + 0.5
                    
                    # Apply attention
                    attention_xyz = attention_xyz * self.sigmoid(attention_graph)
                    local_xyz = torch.flatten(attention_xyz, start_dim=2).permute(0, 2, 1)
                else:
                    # Standard flattening for other datasets
                    local_xyz = torch.flatten(local_xyz, start_dim=1).reshape(b, n, -1).permute(0, 2, 1)
                
                local_xyz = pooling(local_xyz)
                feature_list.append(local_xyz)

        # Concatenate all collected features (Global, Local, or Both)
        output = torch.cat(feature_list, dim=-1)
        
        # L2 Normalize the final feature vector
        output /= output.norm(dim=-1, keepdim=True)
        
        return output

if __name__ == '__main__':
    # Simple test to verify all modes work
    surface = 'knnxyz'
    data = torch.randn(2, 1024, 3).cuda()
    
    # Test United
    print("Testing United Mode...")
    model = clsNet(input_points=[1024, 512, 256, 128], num_stages=4, k_neighbors=[90, 45, 22, 11], 
                   rescale=0.1, metric=2, surface=surface, mode='united').cuda().eval()
    out_united = model(data)
    print(f"United Output Shape: {out_united.shape}")

    # Test Global
    print("Testing Global Mode...")
    model = clsNet(input_points=[1024, 512, 256, 128], num_stages=4, k_neighbors=[90, 45, 22, 11], 
                   rescale=0.1, metric=2, surface=surface, mode='global').cuda().eval()
    out_global = model(data)
    print(f"Global Output Shape: {out_global.shape}")

    # Test Local
    print("Testing Local Mode...")
    model = clsNet(input_points=[1024, 512, 256, 128], num_stages=4, k_neighbors=[90, 45, 22, 11], 
                   rescale=0.1, metric=2, surface=surface, mode='local').cuda().eval()
    out_local = model(data)
    print(f"Local Output Shape: {out_local.shape}")