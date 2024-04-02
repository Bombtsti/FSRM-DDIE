import math
import torch
from torch.nn.modules.container import ModuleList
import torch.nn.functional as F
from torch_geometric.nn import (GATConv,
                                SAGPooling,
                                LayerNorm,
                                global_mean_pool,
                                max_pool_neighbor_x,
                                global_add_pool, GCNConv)
from torch_geometric.data import Batch
import numpy as np

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# class MultiHeadAttention(torch.nn.Module):
#     def __init__(self, input_dim, n_heads, ouput_dim=None):
#
#         super(MultiHeadAttention, self).__init__()
#         self.d_k = self.d_v = input_dim // n_heads
#         self.n_heads = n_heads
#         if ouput_dim == None:
#             self.ouput_dim = input_dim
#         else:
#             self.ouput_dim = ouput_dim
#         self.W_Q = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
#         self.W_K = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
#         self.W_V = torch.nn.Linear(input_dim, self.d_v * self.n_heads, bias=False)
#         self.fc = torch.nn.Linear(self.n_heads * self.d_v, self.ouput_dim, bias=False)
#
#     def forward(self, X):
#         ## (S, D) -proj-> (S, D_new) -split-> (S, H, W) -trans-> (H, S, W)
#         Q = self.W_Q(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)
#         K = self.W_K(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)
#         V = self.W_V(X).view(-1, self.n_heads, self.d_v).transpose(0, 1)
#
#         scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
#         # context: [n_heads, len_q, d_v], attn: [n_heads, len_q, len_k]
#         attn = torch.nn.Softmax(dim=-1)(scores)
#         context = torch.matmul(attn, V)
#         # context: [len_q, n_heads * d_v]
#         context = context.transpose(1, 2).reshape(-1, self.n_heads * self.d_v)
#         output = self.fc(context)
#         return output


class AttentionPooling(torch.nn.Module):
    def __init__(self, input_dim):
        super(AttentionPooling, self).__init__()
        self.input_dim = input_dim
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        # 使用全连接层计算注意力权重
        attention_weights = self.softmax(x)
        # 使用注意力权重对输入进行加权平均
        pooled_output = torch.sum(x * attention_weights, dim=1)
        return pooled_output

# class EncoderLayer(torch.nn.Module):
#     def __init__(self, input_dim, n_heads):
#         super(EncoderLayer, self).__init__()
#         self.attn = MultiHeadAttention(input_dim, n_heads)
#         self.AN1 = torch.nn.LayerNorm(input_dim)
#
#         self.l1 = torch.nn.Linear(input_dim, input_dim)
#         self.AN2 = torch.nn.LayerNorm(input_dim)
#
#     def forward(self, X):
#         output = self.attn(X)
#         X = self.AN1(output + X)
#
#         output = self.l1(X)
#         X = self.AN2(output + X)
#
#         return X

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class SN(torch.nn.Module):  # twin network
    def __init__(self, vector_size):
        super(SN, self).__init__()

        self.vector_size = vector_size

        self.l1 = torch.nn.Linear(self.vector_size, self.vector_size)
        self.bn1 = torch.nn.BatchNorm1d(self.vector_size)
        # self.att2 = EncoderLayer(self.vector_size,4)
        self.l2 = torch.nn.Linear(self.vector_size, 50)


        self.l3 = torch.nn.Linear(50, self.vector_size)
        self.bn3 = torch.nn.BatchNorm1d(self.vector_size)
        self.l4 = torch.nn.Linear(self.vector_size, self.vector_size)

        self.dr = torch.nn.Dropout(0.1)

        self.ac = gelu

    def forward(self, X1,X2):
        X1 = self.dr(self.bn1(self.ac(self.l1(X1))))
        # X1 = self.att2(X1)
        X1 = self.l2(X1)
        X_AE1 = self.dr(self.bn3(self.ac(self.l3(X1))))
        X_AE1 = self.l4(X_AE1)

        X2 = self.dr(self.bn1(self.ac(self.l1(X2))))
        # X2 = self.att2(X2)
        X2 = self.l2(X2)
        X_AE2 = self.dr(self.bn3(self.ac(self.l3(X2))))
        X_AE2 = self.l4(X_AE2)

        X = torch.cat((X1, X2), 1)
        X_AE = torch.cat((X_AE1, X_AE2), 1)

        return X, X_AE


class SSI_DDI(torch.nn.Module):
    def __init__(self, in_features=55, hidd_dim=64, heads_out_feat_params=[32,32,32,32], blocks_params=[2, 2, 2, 2]):
        super().__init__()
        self.in_features = in_features
        self.hidd_dim = hidd_dim
        self.n_blocks = len(blocks_params)

        self.initial_norm = LayerNorm(self.in_features)
        self.blocks = []
        self.net_norms = ModuleList()
        self.attr_pool = AttentionPooling(hidd_dim)
        for i, (head_out_feats, n_heads) in enumerate(zip(heads_out_feat_params, blocks_params)):
            block = SSI_DDI_Block(n_heads, in_features, head_out_feats, final_out_feats=self.hidd_dim)
            self.add_module(f"block{i}", block)
            self.blocks.append(block)
            self.net_norms.append(LayerNorm(head_out_feats * n_heads))
            in_features = head_out_feats * n_heads
        self.sn = SN(hidd_dim)
        self.mlp = torch.nn.Linear(hidd_dim, 100)

    def forward(self, data):
        h_data, t_data, = data

        h_data.x = self.initial_norm(h_data.x, h_data.batch)
        t_data.x = self.initial_norm(t_data.x, t_data.batch)

        repr_h = []
        repr_t = []

        for i, block in enumerate(self.blocks):
            out1, out2 = block(h_data), block(t_data)

            h_data = out1[0]
            t_data = out2[0]
            r_h = out1[1]
            r_t = out2[1]

            repr_h.append(r_h)
            repr_t.append(r_t)

            h_data.x = F.elu(self.net_norms[i](h_data.x, h_data.batch))
            t_data.x = F.elu(self.net_norms[i](t_data.x, t_data.batch))

        repr_h = torch.stack(repr_h, dim=-2)
        repr_t = torch.stack(repr_t, dim=-2)

        repr_h = self.attr_pool(repr_h)
        repr_t = self.attr_pool(repr_t)

        repr_before = torch.cat((repr_h, repr_t), dim=1)  # 64+64=128
        repr,repr_after = self.sn(repr_h,repr_t)
        repr_ori = self.mlp(repr_h+repr_t)
        repr = repr+0.1*repr_ori  #+ 0.1 91.96 0.5 9463 * 0.2 89.72


        return repr,repr_before,repr_after


class SSI_DDI_Block(torch.nn.Module):
    def __init__(self, n_heads, in_features, head_out_feats, final_out_feats):
        super().__init__()
        self.n_heads = n_heads
        self.in_features = in_features
        self.out_features = head_out_feats
        self.conv = GATConv(in_features, head_out_feats, n_heads)
        # self.conv = GCNConv(in_features, head_out_feats*n_heads)
        self.readout = SAGPooling(n_heads * head_out_feats, min_score=-1)

    def forward(self, data):
        data.x = self.conv(data.x, data.edge_index)
        att_x, att_edge_index, att_edge_attr, att_batch, att_perm, att_scores = self.readout(data.x, data.edge_index,
                                                                                             batch=data.batch)
        global_graph_emb = global_add_pool(att_x, att_batch)

        # data = max_pool_neighbor_x(data)
        return data, global_graph_emb

def get_data(h_list, t_list):
    h_loader = Batch.from_data_list(h_list).to(device)
    t_loader = Batch.from_data_list(t_list).to(device)
    data = (h_loader,t_loader)
    return data