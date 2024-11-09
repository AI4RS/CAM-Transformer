# model.py
# note: skip PCT for now

import torch
import torch.nn as nn
import torch.nn.functional as F

from module import Embedding, OA, SA, ASCN, PT, CrossAttention, CAM, SE, ECA, CBAM, mCAM #, NeighborEmbedding

class NaivePCT(nn.Module):
    def __init__(self, n_dim):
        super().__init__()

        self.embedding = Embedding(n_dim, 128)

        self.sa1 = SA(128)
        self.sa2 = SA(128)
        self.sa3 = SA(128)
        self.sa4 = SA(128)

        self.linear = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2)
        )
    
    def forward(self, x):
        x = self.embedding(x)
        
        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        x = torch.cat([x1, x2, x3, x4], dim=1)

        x = self.linear(x)

        # x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x_max = torch.max(x, dim=-1)[0]
        x_mean = torch.mean(x, dim=-1)

        return x, x_max, x_mean


class SPCT(nn.Module):
    def __init__(self, n_dim):
        super().__init__()

        self.embedding = Embedding(n_dim, 128)

        self.sa1 = OA(128)
        self.sa2 = OA(128)
        self.sa3 = OA(128)
        self.sa4 = OA(128)

        self.linear = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2)
        )
    
    def forward(self, x):
        x = self.embedding(x)
        
        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        x = torch.cat([x1, x2, x3, x4], dim=1)

        x = self.linear(x)

        # x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x_max = torch.max(x, dim=-1)[0]
        x_mean = torch.mean(x, dim=-1)

        return x, x_max, x_mean

    
class ASCN_PCT(nn.Module):
    def __init__(self, n_dim):
        super().__init__()

        self.embedding = Embedding(n_dim, 128)

        self.ascn1 = ASCN(128)
        self.ascn2 = ASCN(128)
        self.ascn3 = ASCN(128)
        self.ascn4 = ASCN(128)

        self.linear = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2)
        )
    
    def forward(self, x):
        x = self.embedding(x)
        
        x1 = self.ascn1(x)
        x2 = self.ascn2(x1)
        x3 = self.ascn3(x2)
        x4 = self.ascn4(x3)
        x = torch.cat([x1, x2, x3, x4], dim=1)

        x = self.linear(x)

        # x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x_max = torch.max(x, dim=-1)[0]
        x_mean = torch.mean(x, dim=-1)

        return x, x_max, x_mean

class PT_PCT(nn.Module):
    def __init__(self, n_dim):
        super().__init__()

        self.embedding = Embedding(n_dim, 128)

        self.pt1 = PT(128)
        self.pt2 = PT(128)
        self.pt3 = PT(128)
        self.pt4 = PT(128)

        self.linear = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2)
        )
    
    def forward(self, x):
        x = self.embedding(x)
        
        x1 = self.pt1(x)
        x2 = self.pt2(x1)
        x3 = self.pt3(x2)
        x4 = self.pt4(x3)
        x = torch.cat([x1, x2, x3, x4], dim=1)

        x = self.linear(x)

        # x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x_max = torch.max(x, dim=-1)[0]
        x_mean = torch.mean(x, dim=-1)

        return x, x_max, x_mean

    
class SUM_PT_PCT(nn.Module):
    def __init__(self, n_dim):
        super().__init__()

        dim_swir = 141
        dim_vnir = 51
        self.embedding_a = Embedding(dim_swir, 128)
        self.embedding_b = Embedding(dim_vnir, 128)

        self.pt1_a = PT(128)
        self.pt2_a = PT(128)
        self.pt3_a = PT(128)
        self.pt4_a = PT(128)
        
        self.pt1_b = PT(128)
        self.pt2_b = PT(128)
        self.pt3_b = PT(128)
        self.pt4_b = PT(128)

        self.linear = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2)
        )
    
    def forward(self, x):
        # print(x.shape)
        xa = x[:,:141,:]
        xb = x[:,141:,:]
        
        xa = self.embedding_a(xa)
        x1a = self.pt1_a(xa)
        x2a = self.pt2_a(x1a)
        x3a = self.pt3_a(x2a)
        x4a = self.pt4_a(x3a)
        
        xb = self.embedding_b(xb)
        x1b = self.pt1_b(xb)
        x2b = self.pt2_b(x1b)
        x3b = self.pt3_b(x2b)
        x4b = self.pt4_b(x3b)
        
        x1 = x1a + x1b
        x2 = x2a + x2b
        x3 = x3a + x3b
        x4 = x4a + x4b
        
        x = torch.cat([x1, x2, x3, x4], dim=1)

        x = self.linear(x)

        # x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x_max = torch.max(x, dim=-1)[0]
        x_mean = torch.mean(x, dim=-1)

        return x, x_max, x_mean

class HPF_PT_PCT(nn.Module):
    def __init__(self, n_dim):
        super().__init__()

        dim_swir = 141
        dim_vnir = 51
        self.embedding_a = Embedding(dim_swir, 128)
        self.embedding_b = Embedding(dim_vnir, 128)

        self.pt1_a = PT(128)
        self.pt2_a = PT(128)
        self.pt3_a = PT(128)
        self.pt4_a = PT(128)
        
        self.pt1_b = PT(128)
        self.pt2_b = PT(128)
        self.pt3_b = PT(128)
        self.pt4_b = PT(128)
        
        self.CPA1a = CrossAttention(dim=128)
        self.CPA2a = CrossAttention(dim=128)
        self.CPA3a = CrossAttention(dim=128)
        self.CPA4a = CrossAttention(dim=128)
        
        self.CPA1b = CrossAttention(dim=128)
        self.CPA2b = CrossAttention(dim=128)
        self.CPA3b = CrossAttention(dim=128)
        self.CPA4b = CrossAttention(dim=128)

        self.linear = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2)
        )
    
    def forward(self, x):
        # print(x.shape)
        xa = x[:,:141,:]
        xb = x[:,141:,:]
        
        xa = self.embedding_a(xa)
        x1a = self.pt1_a(xa)
        x2a = self.pt2_a(x1a)
        x3a = self.pt3_a(x2a)
        x4a = self.pt4_a(x3a)
        
        xb = self.embedding_b(xb)
        x1b = self.pt1_b(xb)
        x2b = self.pt2_b(x1b)
        x3b = self.pt3_b(x2b)
        x4b = self.pt4_b(x3b)
        
        cpa1a = self.CPA1a(x1a, x1b)
        cpa2a = self.CPA2a(x2a, x2b)
        cpa3a = self.CPA3a(x3a, x3b)
        cpa4a = self.CPA4a(x4a, x4b)
        
        cpa1b = self.CPA1b(x1b, x1a)
        cpa2b = self.CPA2b(x2b, x2a)
        cpa3b = self.CPA3b(x3b, x3a)
        cpa4b = self.CPA4b(x4b, x4a)
        
        x1 = cpa1a + cpa1b
        x2 = cpa2a + cpa2b
        x3 = cpa3a + cpa3b
        x4 = cpa4a + cpa4b
        
        x = torch.cat([x1, x2, x3, x4], dim=1)

        x = self.linear(x)

        # x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x_max = torch.max(x, dim=-1)[0]
        x_mean = torch.mean(x, dim=-1)

        return x, x_max, x_mean
    
class CAM_PT_PCT(nn.Module):
    def __init__(self, n_dim):
        super().__init__()

        self.embedding = Embedding(n_dim, 128)

        self.pt1 = PT(128,1)
        self.pt2 = PT(128,2)
        self.pt3 = PT(128,3)
        self.pt4 = PT(128,4)
        
        self.cam1 = CAM(1)
        self.cam2 = CAM(2)
        self.cam3 = CAM(3)
        self.cam4 = CAM(4)

        self.linear = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2)
        )
    
    def forward(self, x):
        x = self.embedding(x)
        
        x1 = self.pt1(x)
        x1 = self.cam1(x1)
        x2 = self.pt2(x1)
        x2 = self.cam2(x2)
        x3 = self.pt3(x2)
        x3 = self.cam3(x3)
        x4 = self.pt4(x3)
        x4 = self.cam4(x4)
        x = torch.cat([x1, x2, x3, x4], dim=1)

        x = self.linear(x)

        # x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x_max = torch.max(x, dim=-1)[0]
        x_mean = torch.mean(x, dim=-1)

        return x, x_max, x_mean

class mCAM_PT_PCT(nn.Module):
    def __init__(self, n_dim):
        super().__init__()

        self.embedding = Embedding(n_dim, 128)

        self.pt1 = PT(128)
        self.pt2 = PT(128)
        self.pt3 = PT(128)
        self.pt4 = PT(128)
        
        self.mcam1 = mCAM(128)
        self.mcam2 = mCAM(128)
        self.mcam3 = mCAM(128)
        self.mcam4 = mCAM(128)

        self.linear = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2)
        )
    
    def forward(self, x):
        x = self.embedding(x)
        
        x1 = self.pt1(x)
        x1 = self.mcam1(x1)
        x2 = self.pt2(x1)
        x2 = self.mcam2(x2)
        x3 = self.pt3(x2)
        x3 = self.mcam3(x3)
        x4 = self.pt4(x3)
        x4 = self.mcam4(x4)
        x = torch.cat([x1, x2, x3, x4], dim=1)

        x = self.linear(x)

        # x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x_max = torch.max(x, dim=-1)[0]
        x_mean = torch.mean(x, dim=-1)

        return x, x_max, x_mean

class SE_PT_PCT(nn.Module):
    def __init__(self, n_dim):
        super().__init__()

        self.embedding = Embedding(n_dim, 128)

        self.pt1 = PT(128)
        self.pt2 = PT(128)
        self.pt3 = PT(128)
        self.pt4 = PT(128)
        
        self.se1 = SE(128)
        self.se2 = SE(128)
        self.se3 = SE(128)
        self.se4 = SE(128)

        self.linear = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2)
        )
    
    def forward(self, x):
        x = self.embedding(x)
        
        x1 = self.pt1(x)
        x1 = self.se1(x1)
        x2 = self.pt2(x1)
        x2 = self.se2(x2)
        x3 = self.pt3(x2)
        x3 = self.se3(x3)
        x4 = self.pt4(x3)
        x4 = self.se4(x4)
        x = torch.cat([x1, x2, x3, x4], dim=1)

        x = self.linear(x)

        # x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x_max = torch.max(x, dim=-1)[0]
        x_mean = torch.mean(x, dim=-1)

        return x, x_max, x_mean

class ECA_PT_PCT(nn.Module):
    def __init__(self, n_dim):
        super().__init__()

        self.embedding = Embedding(n_dim, 128)

        self.pt1 = PT(128)
        self.pt2 = PT(128)
        self.pt3 = PT(128)
        self.pt4 = PT(128)
        
        self.eca1 = ECA(128)
        self.eca2 = ECA(128)
        self.eca3 = ECA(128)
        self.eca4 = ECA(128)

        self.linear = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2)
        )
    
    def forward(self, x):
        x = self.embedding(x)
        
        x1 = self.pt1(x)
        x1 = self.eca1(x1)
        x2 = self.pt2(x1)
        x2 = self.eca2(x2)
        x3 = self.pt3(x2)
        x3 = self.eca3(x3)
        x4 = self.pt4(x3)
        x4 = self.eca4(x4)
        x = torch.cat([x1, x2, x3, x4], dim=1)

        x = self.linear(x)

        # x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x_max = torch.max(x, dim=-1)[0]
        x_mean = torch.mean(x, dim=-1)

        return x, x_max, x_mean

class CBAM_PT_PCT(nn.Module):
    def __init__(self, n_dim):
        super().__init__()

        self.embedding = Embedding(n_dim, 128)

        self.pt1 = PT(128)
        self.pt2 = PT(128)
        self.pt3 = PT(128)
        self.pt4 = PT(128)
        
        self.cbam1 = CBAM(128)
        self.cbam2 = CBAM(128)
        self.cbam3 = CBAM(128)
        self.cbam4 = CBAM(128)

        self.linear = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2)
        )
    
    def forward(self, x):
        x = self.embedding(x)
        
        x1 = self.pt1(x)
        x1 = self.cbam1(x1)
        x2 = self.pt2(x1)
        x2 = self.cbam2(x2)
        x3 = self.pt3(x2)
        x3 = self.cbam3(x3)
        x4 = self.pt4(x3)
        x4 = self.cbam4(x4)
        x = torch.cat([x1, x2, x3, x4], dim=1)

        x = self.linear(x)

        # x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x_max = torch.max(x, dim=-1)[0]
        x_mean = torch.mean(x, dim=-1)

        return x, x_max, x_mean

# class PCT(nn.Module):


class Classification(nn.Module):
    def __init__(self, num_categories=10):
        super().__init__()

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, num_categories)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

        self.dp1 = nn.Dropout(p=0.5)
        self.dp2 = nn.Dropout(p=0.5)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.linear1(x)))
        x = self.dp1(x)
        x = F.relu(self.bn2(self.linear2(x)))
        x = self.dp2(x)
        x = self.linear3(x)
        return x


class Segmentation(nn.Module):
    def __init__(self, part_num):
        super().__init__()

        self.part_num = part_num

        self.label_conv = nn.Sequential(
            nn.Conv1d(16, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.convs1 = nn.Conv1d(1024 * 3 + 64, 512, 1)
        self.convs2 = nn.Conv1d(512, 256, 1)
        self.convs3 = nn.Conv1d(256, self.part_num, 1)

        self.bns1 = nn.BatchNorm1d(512)
        self.bns2 = nn.BatchNorm1d(256)

        self.dp1 = nn.Dropout(0.5)
    
    def forward(self, x, x_max, x_mean, cls_label):
        batch_size, _, N = x.size()

        x_max_feature = x_max.unsqueeze(-1).repeat(1, 1, N)
        x_mean_feature = x_mean.unsqueeze(-1).repeat(1, 1, N)

        cls_label_one_hot = cls_label.view(batch_size, 16, 1)
        cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1, 1, N)

        x = torch.cat([x, x_max_feature, x_mean_feature, cls_label_feature], dim=1)  # 1024 * 3 + 64

        x = F.relu(self.bns1(self.convs1(x)))
        x = self.dp1(x)
        x = F.relu(self.bns2(self.convs2(x)))
        x = self.convs3(x)

        return x


# class NormalEstimation(nn.Module):


"""
Classification networks.
"""

class NaivePCTCls(nn.Module):
    def __init__(self, n_dim, num_categories=10):
        super().__init__()

        self.encoder = NaivePCT(n_dim)
        self.cls = Classification(num_categories)
    
    def forward(self, x):
        _, x, _ = self.encoder(x)
        x = self.cls(x)
        return x


class SPCTCls(nn.Module):
    def __init__(self, n_dim, num_categories=10):
        super().__init__()

        self.encoder = SPCT(n_dim)
        self.cls = Classification(num_categories)
    
    def forward(self, x):
        _, x, _ = self.encoder(x)
        x = self.cls(x)
        return x

class ASCN_PCTCls(nn.Module):
    def __init__(self, n_dim, num_categories=10):
        super().__init__()

        self.encoder = ASCN_PCT(n_dim)
        self.cls = Classification(num_categories)
    
    def forward(self, x):
        _, x, _ = self.encoder(x)
        x = self.cls(x)
        return x
    
class PT_PCTCls(nn.Module):
    def __init__(self, n_dim, num_categories=10):
        super().__init__()

        self.encoder = PT_PCT(n_dim)
        self.cls = Classification(num_categories)
    
    def forward(self, x):
        _, x, _ = self.encoder(x)
        x = self.cls(x)
        return x
    
class HPF_PT_PCTCls(nn.Module):
    def __init__(self, n_dim, num_categories=10):
        super().__init__()

        self.encoder = HPF_PT_PCT(n_dim)
        self.cls = Classification(num_categories)
    
    def forward(self, x):
        _, x, _ = self.encoder(x)
        x = self.cls(x)
        return x
    
class CAM_PT_PCTCls(nn.Module):
    def __init__(self, n_dim, num_categories=10):
        super().__init__()

        self.encoder = CAM_PT_PCT(n_dim)
        self.cls = Classification(num_categories)
    
    def forward(self, x):
        _, x, _ = self.encoder(x)
        x = self.cls(x)
        return x

class mCAM_PT_PCTCls(nn.Module):
    def __init__(self, n_dim, num_categories=10):
        super().__init__()

        self.encoder = mCAM_PT_PCT(n_dim)
        self.cls = Classification(num_categories)
    
    def forward(self, x):
        _, x, _ = self.encoder(x)
        x = self.cls(x)
        return x

class SE_PT_PCTCls(nn.Module):
    def __init__(self, n_dim, num_categories=10):
        super().__init__()

        self.encoder = SE_PT_PCT(n_dim)
        self.cls = Classification(num_categories)
    
    def forward(self, x):
        _, x, _ = self.encoder(x)
        x = self.cls(x)
        return x

class ECA_PT_PCTCls(nn.Module):
    def __init__(self, n_dim, num_categories=10):
        super().__init__()

        self.encoder = ECA_PT_PCT(n_dim)
        self.cls = Classification(num_categories)
    
    def forward(self, x):
        _, x, _ = self.encoder(x)
        x = self.cls(x)
        return x

class CBAM_PT_PCTCls(nn.Module):
    def __init__(self, n_dim, num_categories=10):
        super().__init__()

        self.encoder = CBAM_PT_PCT(n_dim)
        self.cls = Classification(num_categories)
    
    def forward(self, x):
        _, x, _ = self.encoder(x)
        x = self.cls(x)
        return x

# class PCTCls(nn.Module):


"""
Part Segmentation Networks.
"""

class NaivePCTSeg(nn.Module):
    def __init__(self, part_num=50):
        super().__init__()
    
        self.encoder = NaivePCT()
        self.seg = Segmentation(part_num)

    def forward(self, x, cls_label):
        x, x_max, x_mean = self.encoder(x)
        x = self.seg(x, x_max, x_mean, cls_label)
        return x


class SPCTSeg(nn.Module):
    def __init__(self, part_num=50):
        super().__init__()
    
        self.encoder = SPCT()
        self.seg = Segmentation(part_num)

    def forward(self, x, cls_label):
        x, x_max, x_mean = self.encoder(x)
        x = self.seg(x, x_max, x_mean, cls_label)
        return x


# class PCTSeg(nn.Module):


"""
Normal Estimation networks.
"""
