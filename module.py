# module.py
# note: skip SG and  neighbor embedding for now

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# from util import sample_and_knn_group

from matplotlib import pyplot as plt


class Embedding(nn.Module):
    """
    Input Embedding layer which consist of 2 stacked LBR layer.
    """

    def __init__(self, in_channels=3, out_channels=128):
        super(Embedding, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)

        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
    
    def forward(self, x):
        """
        Input
            x: [B, in_channels, N]
        
        Output
            x: [B, out_channels, N]
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class SA(nn.Module):
    """
    Self Attention module.
    """

    def __init__(self, channels):
        super(SA, self).__init__()

        self.da = channels // 4

        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels, 1)
        
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        """
        Input
            x: [B, de, N]
        
        Output
            x: [B, de, N]
        """
        # compute query, key and value matrix
        x_q = self.q_conv(x).permute(0, 2, 1)  # [B, N, da]
        # print('x_q size = ', x_q.size())
        x_k = self.k_conv(x)                   # [B, da, N]
        # print('x_k size = ', x_k.size())
        x_v = self.v_conv(x)                   # [B, de, N]
        # print('x_v size = ', x_v.size())
        # print('v_conv = ', self.v_conv.weight.size())

        # compute attention map and scale, the sorfmax
        energy = torch.bmm(x_q, x_k) / (math.sqrt(self.da))   # [B, N, N]
        # print('energy size = ', energy.size())
        attention = self.softmax(energy)                      # [B, N, N]

        # weighted sum
        x_s = torch.bmm(x_v, attention)  # [B, de, N]
        # print('x_s size = ', x_s.size())
        x_s = self.act(self.after_norm(self.trans_conv(x_s)))
        # print('x_s size = ', x_s.size())
        
        # residual
        x = x + x_s

        return x


# class SG(nn.Module):


# class NeighborEmbedding(nn.Module):


class OA(nn.Module):
    """
    Offset-Attention Module.
    """
    
    def __init__(self, channels):
        super(OA, self).__init__()

        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels, 1)

        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)  # change dim to -2 and change the sum(dim=1, keepdims=True) to dim=2

    def forward(self, x):
        """
        Input:
            x: [B, de, N]
        
        Output:
            x: [B, de, N]
        """
        x_q = self.q_conv(x).permute(0, 2, 1)
        x_k = self.k_conv(x)    
        x_v = self.v_conv(x)

        energy = torch.bmm(x_q, x_k)
        attention = self.softmax(energy)
        print('attention size = ', attention.size())
        print('attention sum size = ', attention.sum(dim=1, keepdims=True).size())
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))  # here

        x_r = torch.bmm(x_v, attention)
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r

        return x

class ASCN(nn.Module):
    """
    Attentional ShapeContextNet module.
    """

    def __init__(self, channels):
        super(ASCN, self).__init__()

        self.da = channels // 4

        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels, 1)
        
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        """
        Input
            x: [B, de, N]
        
        Output
            x: [B, de, N]
        """
        # compute query, key and value matrix
        x_q = self.q_conv(x).permute(0, 2, 1)  # [B, N, da]
        x_k = self.k_conv(x)                   # [B, da, N]        
        x_v = self.v_conv(x)                   # [B, de, N]

        # compute attention map and scale, the sorfmax
        energy = torch.bmm(x_q, x_k) / (math.sqrt(self.da))   # [B, N, N]
        attention = self.softmax(energy)                      # [B, N, N]

        # weighted sum
        x_s = torch.bmm(x_v, attention)  # [B, de, N]
        x_s = self.act(self.after_norm(self.trans_conv(x_s)))
        
        # residual
        x = x_v + x_s # ASCN differs here !!

        return x

    
class PT(nn.Module):
    """
    PointTransformer module.
    """

    def __init__(self, channels, NUM):
        super(PT, self).__init__()
        
        self.NUM = NUM
        self.da = channels

        self.q_conv = nn.Conv1d(channels, channels, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels, 1)
        
        self.fc1 = nn.Linear(128,channels) # 128 from output of input embedding layer
        
        self.fc_gamma = nn.Sequential(
            nn.Linear(channels,channels),
            nn.ReLU(),
            nn.Linear(channels,channels)
        )
        
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        """
        Input
            x: [B, de, N]
        
        Output
            x: [B, de, N]
        """
        # fc1
        x_input = x # [B,da,N]
        x = self.fc1(x.permute(0,2,1)) # [B,N,da]
        x = x.permute(0,2,1) # [B,da,N]
        
        # compute query, key and value matrix
        # x_q = self.q_conv(x).permute(0, 2, 1)  # [B, N, da]
        x_q = self.q_conv(x)                   # [B, da, N] # No transpose !!
        x_k = self.k_conv(x)                   # [B, da, N]        
        x_v = self.v_conv(x)                   # [B, de, N]
         
        # find the central points
        x_q = x_q[:,:,0].unsqueeze(2).repeat(1,1,x.size(2)) # [B,da,N]

        # compute attention map and scale, the sorfmax
        # energy = torch.bmm(x_q, x_k) / (math.sqrt(self.da))   # [B, N, N]
        # attention = self.softmax(energy)                      # [B, N, N]
        
        energy = self.fc_gamma(x_q.permute(0,2,1) - x_k.permute(0,2,1)) # [B,N,da]
        energy = energy.permute(0,2,1) / (math.sqrt(self.da)) # [B,da,N]
        attention = self.softmax(energy) # which dimension for softmax?
        
        # print('attention = ', attention.shape)

        # weighted sum
        # x_s = torch.bmm(x_v, attention)  # [B, de, N] 
        x_s = torch.mul(x_v, attention)  # [B, de, N] Hadamard product instead of dot-product !!
        x_s = self.act(self.after_norm(self.trans_conv(x_s)))
        
        # Visualize
        print('PT size = ', x_s.size())
        y_np = x_s.cpu()
        y_np = y_np.detach().numpy()
        plt.imshow(y_np[0,:,:])
        plt.savefig('visual/PT_fig_%s.png' %str(self.NUM))
        plt.show()
        
        # print('x_s = ', x_s.shape)
        # print('x_q = ', x_q.shape)
        # print('x_k = ', x_k.shape)
        # print('x_v = ', x_v.shape)
        # print('q_conv = ', self.q_conv)
        # print('k_conv = ', self.k_conv)
        # print('v_conv = ', self.v_conv)
        
        # residual
        x = x_s + x_input

        return x
    
class CAM(nn.Module):
    """ Channel attention module https://github.com/junfu1115/DANet/blob/master/encoding/nn/da_att.py"""
    # def __init__(self, in_dim):
    def __init__(self,NUM):
        super(CAM, self).__init__()
        # self.chanel_in = in_dim
        self.NUM = NUM


        self.gamma = nn.Parameter(torch.zeros(1),requires_grad=True)
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
                x : input feature maps( B X C X N)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        # m_batchsize, C, height, width = x.size()
        m_batchsize, C, N = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        # out = out.view(m_batchsize, C, height, width)
        
        # Visualize
        print('CAM size = ', out.size())
        y_np = out.cpu()
        y_np = y_np.detach().numpy()
        plt.imshow(y_np[0,:,:])
        plt.savefig('visual/CAM_fig_%s.png' %str(self.NUM))
        plt.show()

        out = self.gamma*out + x
        return out

import torch
import torch.nn as nn
import torch.nn.functional as F

class mCAM(nn.Module):
    """ Channel attention module https://github.com/junfu1115/DANet/blob/master/encoding/nn/da_att.py"""
    # def __init__(self, in_dim):
    def __init__(self, in_dim):
        super(mCAM, self).__init__()
        self.chanel_in = in_dim
        
        # self.query_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        # self.key_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.query_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)


        self.gamma = nn.Parameter(torch.zeros(1),requires_grad=True)
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
                x : input feature maps( B X C X N)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        # m_batchsize, C, height, width = x.size()
        m_batchsize, C, N = x.size()
        proj_query = self.query_conv(x)
        proj_query = self.query_conv(x).view(m_batchsize, C, -1)
        proj_key = self.key_conv(x).view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = self.value_conv(x).view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        # out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

class SE(nn.Module):
    """ Squeeze excitation module https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py"""
    def __init__(self, channel, reduction=16):
        super(SE, self).__init__()
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # b, c, _, _ = x.size()
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        # y = self.fc(y).view(b, c, 1, 1)
        y = self.fc(y).view(b, c, 1)
        
        # Visualize
        # print('y size = ', y.size())
        y_np = y.cpu()
        y_np = y_np.detach().numpy()
        plt.imshow(y_np)
        # plt.savefig('SE_fig.png')
        plt.show()
        
        out = x * y.expand_as(x)
        return x + out

class ECA(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(ECA, self).__init__()
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        # y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)

        # Multi-scale information fusion
        y = self.sigmoid(y)
        
        # Visualize
        y_np = y.cpu()
        y_np = y_np.detach().numpy()
        plt.imshow(y_np)
        # plt.savefig('ECA_fig.png')
        plt.show()

        out = x * y.expand_as(x)
        return x + out

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(CBAM, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                # avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                avg_pool = F.avg_pool1d( x, x.size(2), stride=x.size(2))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                # max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                max_pool = F.max_pool1d( x, x.size(2), stride=x.size(2))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        # scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).expand_as(x)
        
        # Visualize
        scale_ = F.sigmoid( channel_att_sum ).unsqueeze(2)
        scale_np = scale_.cpu()
        scale_np = scale_np.detach().numpy()
        plt.imshow(scale_np)
        # plt.savefig('CBAM_fig.png')
        plt.show()
        
        return x * scale

    
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        head_dim = dim // num_heads
        self.num_heads = num_heads
        self.scale = head_dim ** -0.5
        self.wq = nn.Linear(head_dim, dim , bias=False)
        self.wk = nn.Linear(head_dim, dim , bias=False)
        self.wv = nn.Linear(head_dim, dim , bias=False)
        self.proj = nn.Linear(dim * num_heads, dim)
        self.proj_drop = nn.Dropout(0.1)
        self.gamma = nn.Parameter(torch.zeros(1), requires_grad=True) # Added DANet
        
    def forward(self, feat_1, feat_2):
        feat_1 = feat_1.permute(0,2,1)
        feat_2 = feat_2.permute(0,2,1)
        # print('feat_1 = ', feat_1.shape)
        B, N, C = feat_1.size()
        pre = feat_1
        q = self.wq(feat_1.reshape(B, N, self.num_heads, C // self.num_heads)).permute(0, 2, 1, 3) # B x N x C -> B x N x H x (C/H)
        k = self.wk(feat_2.reshape(B, N, self.num_heads, C // self.num_heads)).permute(0, 2, 1, 3) # B x N x C -> B x N x H x (C/H)
        v = self.wv(feat_2.reshape(B, N, self.num_heads, C // self.num_heads)).permute(0, 2, 1, 3) # B x N x C -> B x N x H x (C/H)
        attn = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        x = torch.einsum('bhij,bhjd->bhid', attn, v).transpose(1, 2)
        x = x.reshape(B, N, C * self.num_heads)
        x = self.proj(x)
        x = self.proj_drop(x)
        res = self.gamma*x + pre # Added DANet
        res = res.permute(0,2,1)
        # print('res = ', res.shape)
        return res