import torch
import torch.nn as nn
from .surfaces import tFCW_encoding
from .risurconv_utils import sample_and_group


def pooling(x=None):
    x = x.max(-1)[0] + x.mean(-1)
    return x

class clsNet(nn.Module):
    def __init__(self, input_points, num_stages, n_samples, rescale, metric, surface, dataset='mn40'):
        super().__init__()
        self.input_points = input_points
        
        self.num_stages = num_stages
        self.n_samples = n_samples
        self.rescale = rescale
        self.metric = metric
        self.dataset = dataset
        self.tfcw = tFCW_encoding(metric=self.metric)
        self.sigmoid = nn.Sigmoid()


    def forward(self, xyz, norm):
        feature_list = [] # global feature, local feature
        for i in range(self.num_stages):
            input_points = self.input_points[i]//2
            xyz, ri_feat, norm, _ = sample_and_group(npoint=input_points, radius=None, nsample=self.n_samples[i], xyz=xyz, norm=norm)
            b, n, k, d = ri_feat.shape

            # global tfcw
            global_xyz = ri_feat.permute(0, 3, 1, 2)
            global_xyz = pooling(global_xyz)
            global_xyz = self.tfcw(global_xyz) # b, n, d, d
            global_xyz = torch.flatten(global_xyz, start_dim=1)
            feature_list.append(global_xyz)

            # local tfcw
            local_xyz = ri_feat.permute(0, 1, 3, 2)
            local_xyz = local_xyz.reshape(-1, d, k)
            local_xyz = self.tfcw(local_xyz) # b*n, d, d
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
    norm = torch.randn(2, 1024, 3).cuda()
    model = clsNet(input_points=[1024, 512, 256, 128], num_stages=4, n_samples=90, radius=0.1 ,rescale=0.1, metric=2, surface=surface).cuda()
    model = model.eval()
    output = model(data, norm)

    perm = torch.randperm(1024)
    data2 = data[:, perm, :]
    output2 = model(data2)
    max_diff = (output - output2).abs().max().item()
    print("Max abs diff after sorting:", max_diff)
