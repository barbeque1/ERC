import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.nn import RGCNConv, GraphConv, FAConv
from torch.nn import Parameter
import numpy as np, itertools, random, copy, math
import math
import scipy.sparse as sp
from model_GCN import GCNII_lyc
import ipdb
from HypergraphConv import HypergraphConv
from torch_geometric.nn import GCNConv
from itertools import permutations
# from torch_geometric.nn.pool.topk_pool import topk
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, global_add_pool as gsp
from torch_geometric.nn.inits import glorot
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import add_self_loops
from high_fre_conv import highConv
from collections import Counter
from clip import load
class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)

class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, residual=False, variant=False):
        super(GraphConvolution, self).__init__() 
        self.variant = variant
        if self.variant:
            self.in_features = 2*in_features 
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features,self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj , h0 , lamda, alpha, l):
        theta = math.log(lamda/l+1)
        hi = torch.spmm(adj, input)
        if self.variant:
            support = torch.cat([hi,h0],1)
            r = (1-alpha)*hi+alpha*h0
        else:
            support = (1-alpha)*hi+alpha*h0
            r = support
        output = theta*torch.mm(support, self.weight)+(1-theta)*r
        if self.residual:
            output = output+input
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x, dia_len):
        """
        x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        tmpx = torch.zeros(0).cuda()
        tmp = 0
        for i in dia_len:
            a = x[tmp:tmp+i].unsqueeze(1)
            a = a + self.pe[:a.size(0)]
            tmpx = torch.cat([tmpx,a], dim=0)
            tmp = tmp+i
        #x = x + self.pe[:x.size(0)]
        tmpx = tmpx.squeeze(1)
        return self.dropout(tmpx)

class HyperGCN(nn.Module):
    def __init__(self, a_dim, v_dim, l_dim, n_dim, nlayers, nhidden, nclass,
             dropout, lamda, alpha, variant, return_feature, use_residue,
             window_p=10, window_f=10,
             new_graph='full', n_speakers=2, modals=['a', 'v', 'l'],
             use_speaker=True, use_modal=False, num_L=3, num_K=4,
             dataset_name='IEMOCAP'):

        super(HyperGCN, self).__init__()
        
        self.dataset_name = dataset_name
        # self.utt_fc = nn.Linear(self.hid_dim * 3, self.hid_dim)
        # 改了后
        # self.utt_fc = nn.Linear(nhidden * 3, nhidden)
        self.utt_fc = nn.LazyLinear(nhidden)

        self.return_feature = return_feature
        self.use_residue = use_residue
        self.window_p = window_p
        self.window_f = window_f
        self.new_graph = new_graph
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda
        self.modals = modals
        self.modal_embeddings = nn.Embedding(3, n_dim)
        self.speaker_embeddings = nn.Embedding(n_speakers, n_dim)
        self.use_speaker = use_speaker
        self.use_modal = use_modal
        self.use_position = False

        self.fc1 = nn.Linear(n_dim, nhidden)
        self.num_L = num_L
        self.num_K = num_K
        self.layer_weights = nn.Parameter(torch.zeros(self.num_K))
        self.beta = nn.Parameter(torch.tensor(0.0))

        for ll in range(num_L):
            setattr(self, 'hyperconv%d' % (ll + 1), HypergraphConv(nhidden, nhidden))

        self.act_fn = nn.ReLU()
        self.hyperedge_weight = nn.Parameter(torch.ones(1000))
        self.EW_weight = nn.Parameter(torch.ones(5200))
        self.hyperedge_attr1 = nn.Parameter(torch.rand(nhidden))
        self.hyperedge_attr2 = nn.Parameter(torch.rand(nhidden))

        for kk in range(num_K):
            setattr(self, 'conv%d' % (kk + 1), highConv(nhidden, nhidden))

        corruption = "node_shuffle"
        self.corruption = getattr(self, "_%s" % corruption)

    def _build_window_edges(self, nodes):
        """
        nodes: list[int]，按时间顺序排列的节点 id
        return: list[(src, dst)]，不含自环
        """
        edges = []
        T = len(nodes)

        for t in range(T):
            left = max(0, t - self.window_p)
            right = min(T - 1, t + self.window_f)
            src = nodes[t]

            for j in range(left, right + 1):
                if j == t:
                    continue
                dst = nodes[j]
                edges.append((src, dst))

        return edges
        # 
    def build_utterance_features(self, a, v, l):
        if self.dataset_name == 'MELD':
            u = l
        else:
            u = torch.cat([l, a, v], dim=-1)

        u = self.utt_fc(u)
        return u
        # 新增一个函数，构建 utterance-level GNN 的边索引
    # def create_utterance_gnn_index(self, u, l, dia_len, spk_idx):
    #     node_count = 0
    #     index = []

    #     for dlen in dia_len:

    #         # 构建当前对话的节点
    #         nodes = list(range(dlen))
    #         nodes = [n + node_count for n in nodes]

    #         # ===== window edges =====
    #         index += self._build_window_edges(nodes)

    #         # ===== speaker edges (ELR) =====
    #         spk = spk_idx[node_count: node_count + dlen]

    #         for i in range(dlen):
    #             for j in range(dlen):
    #                 if i != j and spk[i] == spk[j]:
    #                     index.append((nodes[i], nodes[j]))

    #         # ===== long-distance edges =====
    #         feats = u[node_count: node_count + dlen]
    #         index += self._build_long_range_edges(feats, nodes, topk=1)

    #         node_count += dlen


    #     if len(index) > 0:
    #         edge_index = torch.LongTensor(index).t().contiguous().cuda()
    #     else:
    #         edge_index = torch.empty((2, 0), dtype=torch.long).cuda()

    #     return edge_index
    def create_utterance_gnn_index(self, u, l, dia_len, spk_idx):
        node_count = 0
        index = []

        for dlen in dia_len:
            # 当前对话节点
            nodes = list(range(dlen))
            nodes = [n + node_count for n in nodes]

            # ===== 所有数据集都保留：window edges =====
            index += self._build_window_edges(nodes)

            # ===== 所有数据集都保留：speaker edges =====
            spk = spk_idx[node_count: node_count + dlen]
            for i in range(dlen):
                for j in range(i + 1, dlen):
                    if spk[i] == spk[j]:
                        index.append((nodes[i], nodes[j]))


            # ===== 只给 IEMOCAP 加 long-range =====
            if self.dataset_name == 'IEMOCAP':
                feats = u[node_count: node_count + dlen]
                index += self._build_long_range_edges(feats, nodes, topk=1)

            # ===== MELD 先不加 long-range =====
            elif self.dataset_name == 'MELD':
                # feats = u[node_count: node_count + dlen]
                # index += self._build_long_range_edges(feats, nodes, topk=2)
                pass
            node_count += dlen

        device = u.device
        if len(index) > 0:
            edge_index = torch.LongTensor(index).t().contiguous().to(device)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long).to(device)

        return edge_index

        # 新增一个函数，将 utterance-level 的特征扩展到对应的模态节点上，供后续计算相似度矩阵和信息熵时使用
    # ===== ELR ADDITION =====
    # 新增一个函数，构建 long-range 边，基于特征相似度选择 top-k 相关节点连接
    def _build_long_range_edges(self, feats, nodes, topk=3):
        """
        feats: 当前对话的特征 [T, D]
        nodes: 对应节点id
        """

        if feats.size(0) <= 1:
            return []

        feats = F.normalize(feats, p=2, dim=-1)
        sim = torch.matmul(feats, feats.t())

        edges = []
        T = sim.size(0)

        for i in range(T):

            sim_i = sim[i].clone()
            sim_i[i] = -1e9

            k = min(topk, T - 1)

            _, idx = torch.topk(sim_i, k=k)

            for j in idx.tolist():

                edges.append((nodes[i], nodes[j]))
                edges.append((nodes[j], nodes[i]))

        return edges

    def expand_utterance_to_modal_nodes(self, utt_out, dia_len):
        expanded = []
        start = 0

        for dlen in dia_len:
            cur = utt_out[start:start + dlen]      # [dlen, D]
            cur_expand = torch.cat([cur, cur, cur], dim=0)  # [3*dlen, D]
            expanded.append(cur_expand)
            start += dlen

        return torch.cat(expanded, dim=0)

    def _node_shuffle(self, X):
        perm = torch.randperm(X.size(0))
        neg_X = X[perm]
        return neg_X

    def utterance_selector(self, key, context):
        s1 = torch.einsum("bu,u->b", context, key) / (
            1e-6 + torch.norm(context, dim=-1) * torch.norm(key, dim=-1, keepdim=True)
        )
        return s1

    def utterance_selector_2(self, a, b):
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a)

        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b)

        if len(a.shape) == 1:
            a = a.unsqueeze(0)

        if len(b.shape) == 1:
            b = b.unsqueeze(0)

        a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
        b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
        return (a_norm * b_norm).sum(-1)

    def get_batch_entropy(self, tokens):
        dia_entro = []
        for batch_index, data in enumerate(tokens):
            dia_tokens = [token for st in data for token in st]
            length = len(dia_tokens)
            tokens_dicts = Counter(dia_tokens)
            entro = []
            for utt in data:
                shanno = 0
                for token in utt:
                    prob = tokens_dicts[token] / length
                    shanno -= prob * math.log(prob, 2)
                entro.append(shanno)
                entro.append(shanno)
                entro.append(shanno)
            dia_entro.append(torch.Tensor(entro))

        batch_entro = torch.cat(dia_entro, dim=0)
        return batch_entro

    def get_batch_entropy_1(self, tokens):
        dia_entro = []
        for batch_index, data in enumerate(tokens):
            dia_tokens = [token for st in data for token in st]
            length = len(dia_tokens)
            tokens_dicts = Counter(dia_tokens)
            entro = []
            for utt in data:
                shanno = 0
                for token in utt:
                    prob = tokens_dicts[token] / length
                    shanno -= prob * math.log(prob, 2)
                entro.append(shanno)
            dia_entro.append(torch.Tensor(entro))

        batch_entro = torch.cat(dia_entro, dim=0)
        return batch_entro

    def forward(self, a, v, l, dia_len, qmask, epoch, Sentence):

        num_modality = len(self.modals)
        qmask = torch.cat([qmask[:x, i, :] for i, x in enumerate(dia_len)], dim=0)
        spk_idx = torch.argmax(qmask, dim=-1)
        spk_emb_vector = self.speaker_embeddings(spk_idx)

        if self.use_speaker:
            if 'l' in self.modals:
                l += spk_emb_vector

        if self.use_position:
            if 'l' in self.modals:
                l = self.l_pos(l, dia_len)
            if 'a' in self.modals:
                a = self.a_pos(a, dia_len)
            if 'v' in self.modals:
                v = self.v_pos(v, dia_len)

        if self.use_modal:
            emb_idx = torch.LongTensor([0, 1, 2]).cuda()
            emb_vector = self.modal_embeddings(emb_idx)

            if 'a' in self.modals:
                a += emb_vector[0].reshape(1, -1).expand(a.shape[0], a.shape[1])
            if 'v' in self.modals:
                v += emb_vector[1].reshape(1, -1).expand(v.shape[0], v.shape[1])
            if 'l' in self.modals:
                l += emb_vector[2].reshape(1, -1).expand(l.shape[0], l.shape[1])

        # ---------------- Hyper GCN ----------------
        hyperedge_index, edge_index, features, batch, hyperedge_type1 = self.create_hyper_index(
            a, v, l, dia_len, self.modals
        )
        x1 = self.fc1(features)
        weight = self.hyperedge_weight[0:hyperedge_index[1].max().item() + 1]
        EW_weight = self.EW_weight[0:hyperedge_index.size(1)]

        edge_attr = self.hyperedge_attr1 * hyperedge_type1 + self.hyperedge_attr2 * (1 - hyperedge_type1)
        out = x1
        for ll in range(self.num_L):
            out = getattr(self, 'hyperconv%d' % (ll + 1))(
                out, hyperedge_index, weight, edge_attr, EW_weight, dia_len
            )
        # ---------------- 话语级 GNN 分支 ----------------
        # 将多模态表示聚合为每个 utterance 的节点特征
        u = self.build_utterance_features(a, v, l)
        # 构建话语图边：同说话人邻接 + 滑动窗口时序边
        utt_edge_index = self.create_utterance_gnn_index(u, l, dia_len, spk_idx)

        # 在话语图上进行 K 层残差式图卷积更新，并保存每层输出
        utt_out = u
        utt_layer_outputs = []

        for kk in range(self.num_K):
            utt_out = utt_out + getattr(self, 'conv%d' % (kk + 1))(utt_out, utt_edge_index)
            utt_layer_outputs.append(utt_out)

        # 层加权融合
        stacked_h = torch.stack(utt_layer_outputs, dim=0)   # [K, N, D]
        alpha = torch.softmax(self.layer_weights, dim=0)    # [K]
        alpha = alpha.view(-1, 1, 1)

        agg_h = (alpha * stacked_h).sum(dim=0)              # [N, D]
        last_h = utt_layer_outputs[-1]
        beta = torch.sigmoid(self.beta)

        utt_out = beta * last_h + (1 - beta) * agg_h

        # 将 utterance 节点表示扩展回模态节点粒度，与超图分支对齐
        utt_out = self.expand_utterance_to_modal_nodes(utt_out, dia_len)
        # 融合超图分支(out)与话语图分支(utt_out)特征
        out2 = torch.cat([out, utt_out], dim=1)

        # gnn_edge_index, gnn_features = self.create_gnn_index(a, v, l, dia_len, self.modals)
        # gnn_edge_index, gnn_features = self.create_gnn_index(a, v, l, dia_len, self.modals)
        # gnn_out = x1
        # for kk in range(self.num_K):
        #     gnn_out = gnn_out + getattr(self, 'conv%d' % (kk + 1))(gnn_out, gnn_edge_index)

        # out2 = torch.cat([out, gnn_out], dim=1)
        if self.use_residue:
            out2 = torch.cat([features, out2], dim=-1)

        # 主输出和辅助输出保持同维度，避免分类头输入不一致
        out1 = out2

        out1 = self.reverse_features(dia_len, out1)
        out2 = self.reverse_features(dia_len, out2)

        # ---------------- 相似度矩阵分支 ----------------
        gnn_edge_index = utt_edge_index
        utt_len = u.size(0)
        score1 = []

        for index in range(utt_len):
            key = x1[index, :]
            s1 = self.utterance_selector(key, x1)
            s1 = s1.unsqueeze(0)
            score1.append(s1)

        s1 = torch.cat(score1, dim=0)
        attention_weights = F.softmax(s1, dim=1)
        weighted_features = torch.matmul(attention_weights, x1)

        gnn_out_2 = weighted_features
        for kk in range(self.num_K):
            gnn_out_2 = gnn_out_2 + getattr(self, 'conv%d' % (kk + 1))(gnn_out_2, gnn_edge_index)

        gnn_out_2 = self.expand_utterance_to_modal_nodes(gnn_out_2, dia_len)

        out3 = torch.cat([out, gnn_out_2], dim=1)
        if self.use_residue:
            out3 = torch.cat([features, out3], dim=-1)
        out3 = self.reverse_features(dia_len, out3)

        # ---------------- 信息熵分支 ----------------
        # device = gnn_features.device
        if self.dataset_name == 'MELD':
            out4 = out1
        else:
            device = x1.device

            if num_modality == 3:
                s2 = self.get_batch_entropy(Sentence).to(device)
                attention_weights = F.softmax(s2, dim=0)
                weighted_features = attention_weights.unsqueeze(-1) * x1
            elif num_modality == 1:
                s2 = self.get_batch_entropy_1(Sentence).to(device)
                attention_weights = F.softmax(s2, dim=0)
                weighted_features = attention_weights.unsqueeze(-1) * x1
            else:
                weighted_features = x1

            gnn_out_3 = weighted_features
            for kk in range(self.num_K):
                gnn_out_3 = gnn_out_3 + getattr(self, 'conv%d' % (kk + 1))(gnn_out_3, gnn_edge_index)

            out4 = torch.cat([weighted_features, gnn_out_3], dim=1)
            if self.use_residue:
                out4 = torch.cat([features, out4], dim=-1)
            out4 = self.reverse_features(dia_len, out4)

        return out1, out2, out3, out4

    def create_hyper_index(self, a, v, l, dia_len, modals):
        self_loop = False
        num_modality = len(modals)
        node_count = 0
        edge_count = 0
        batch_count = 0
        index1 = []
        index2 = []
        tmp = []
        batch = []
        edge_type = torch.zeros(0).cuda()
        in_index0 = torch.zeros(0).cuda()
        hyperedge_type1 = []

        for i in dia_len:
            nodes = list(range(i * num_modality))
            nodes = [j + node_count for j in nodes]
            nodes_l = nodes[0:i * num_modality // 3]
            nodes_a = nodes[i * num_modality // 3:i * num_modality * 2 // 3]
            nodes_v = nodes[i * num_modality * 2 // 3:]

            index1 = index1 + nodes_l + nodes_a + nodes_v
            for _ in range(i):
                index1 = index1 + [nodes_l[_]] + [nodes_a[_]] + [nodes_v[_]]

            for _ in range(i + 3):
                if _ < 3:
                    index2 = index2 + [edge_count] * i
                else:
                    index2 = index2 + [edge_count] * 3
                edge_count = edge_count + 1

            if node_count == 0:
                ll = l[0:i]
                aa = a[0:i]
                vv = v[0:i]
                features = torch.cat([ll, aa, vv], dim=0)
                temp = i
            else:
                ll = l[temp:temp + i]
                aa = a[temp:temp + i]
                vv = v[temp:temp + i]
                features_temp = torch.cat([ll, aa, vv], dim=0)
                features = torch.cat([features, features_temp], dim=0)
                temp = temp + i

            Gnodes = []
            Gnodes.append(nodes_l)
            Gnodes.append(nodes_a)
            Gnodes.append(nodes_v)

            for _ in range(i):
                Gnodes.append([nodes_l[_], nodes_a[_], nodes_v[_]])

            for ii, _ in enumerate(Gnodes):
                perm = list(permutations(_, 2))
                tmp = tmp + perm

            batch = batch + [batch_count] * i * 3
            batch_count = batch_count + 1
            hyperedge_type1 = hyperedge_type1 + [1] * i + [0] * 3

            node_count = node_count + i * num_modality

        index1 = torch.LongTensor(index1).view(1, -1)
        index2 = torch.LongTensor(index2).view(1, -1)
        hyperedge_index = torch.cat([index1, index2], dim=0).cuda()

        if self_loop:
            max_edge = hyperedge_index[1].max()
            max_node = hyperedge_index[0].max()
            loops = torch.cat([
                torch.arange(0, max_node + 1, 1).repeat_interleave(2).view(1, -1),
                torch.arange(max_edge + 1, max_edge + 1 + max_node + 1, 1).repeat_interleave(2).view(1, -1)
            ], dim=0).cuda()
            hyperedge_index = torch.cat([hyperedge_index, loops], dim=1)

        edge_index = torch.LongTensor(tmp).T.cuda()
        batch = torch.LongTensor(batch).cuda()
        hyperedge_type1 = torch.LongTensor(hyperedge_type1).view(-1, 1).cuda()

        return hyperedge_index, edge_index, features, batch, hyperedge_type1

    # # 新增这个函数
    # def _build_long_range_edges(self, feats, nodes, topk=3):
    #     if feats.size(0) <= 1:
    #         return []

    #     feats = F.normalize(feats, p=2, dim=-1)
    #     sim = torch.matmul(feats, feats.t())

    #     edges = []
    #     T = sim.size(0)

    #     for i in range(T):
    #         sim_i = sim[i].clone()
    #         sim_i[i] = -1e9
    #         k = min(topk, T - 1)
    #         _, idx = torch.topk(sim_i, k=k)

    #         for j in idx.tolist():
    #             edges.append((nodes[i], nodes[j]))
    #             edges.append((nodes[j], nodes[i]))  # 双向边

    #     return edges


    # def create_gnn_index(self, a, v, l, dia_len, modals):
    #     num_modality = len(modals)
    #     node_count = 0
    #     index = []
    #     tmp = []

    #     for i in dia_len:
    #         nodes = list(range(i * num_modality))
    #         nodes = [j + node_count for j in nodes]

    #         nodes_l = nodes[0:i * num_modality // 3]
    #         nodes_a = nodes[i * num_modality // 3:i * num_modality * 2 // 3]
    #         nodes_v = nodes[i * num_modality * 2 // 3:]

    #         # 同模态：滑动窗口边
    #         index += self._build_window_edges(nodes_l)
    #         index += self._build_window_edges(nodes_a)
    #         index += self._build_window_edges(nodes_v)

    #         # 同一 utterance 的跨模态边
    #         Gnodes = []
    #         for t in range(i):
    #             Gnodes.append([nodes_l[t], nodes_a[t], nodes_v[t]])

    #         for group in Gnodes:
    #             tmp += list(permutations(group, 2))

    #         if node_count == 0:
    #             ll = l[0:i]
    #             aa = a[0:i]
    #             vv = v[0:i]
    #             features = torch.cat([ll, aa, vv], dim=0)
    #             temp = i
    #         else:
    #             ll = l[temp:temp + i]
    #             aa = a[temp:temp + i]
    #             vv = v[temp:temp + i]
    #             features_temp = torch.cat([ll, aa, vv], dim=0)
    #             features = torch.cat([features, features_temp], dim=0)
    #             temp = temp + i

    #         node_count = node_count + i * num_modality

    #     # 更稳的 edge_index 构造，避免空边时报错
    #     if len(index) > 0:
    #         intra_edges = torch.LongTensor(index).t().contiguous()
    #     else:
    #         intra_edges = torch.empty((2, 0), dtype=torch.long)

    #     if len(tmp) > 0:
    #         cross_edges = torch.LongTensor(tmp).t().contiguous()
    #     else:
    #         cross_edges = torch.empty((2, 0), dtype=torch.long)

    #     edge_index = torch.cat([intra_edges, cross_edges], dim=1).cuda()

    #     return edge_index, features
    
    # def create_gnn_index(self, a, v, l, dia_len, modals):
    #     num_modality = len(modals)
    #     node_count = 0
    #     index = []
    #     tmp = []

    #     temp = 0

    #     for i in dia_len:
    #         nodes = list(range(i * num_modality))
    #         nodes = [j + node_count for j in nodes]

    #         nodes_l = nodes[0:i * num_modality // 3]
    #         nodes_a = nodes[i * num_modality // 3:i * num_modality * 2 // 3]
    #         nodes_v = nodes[i * num_modality * 2 // 3:]

    #         # 当前对话对应的特征
    #         ll = l[temp:temp + i]
    #         aa = a[temp:temp + i]
    #         vv = v[temp:temp + i]

    #         # 同模态：滑动窗口边
    #         index += self._build_window_edges(nodes_l)
    #         index += self._build_window_edges(nodes_a)
    #         index += self._build_window_edges(nodes_v)

    #         # 同模态：long-range 边
    #         index += self._build_long_range_edges(ll, nodes_l, topk=3)
    #         index += self._build_long_range_edges(aa, nodes_a, topk=3)
    #         index += self._build_long_range_edges(vv, nodes_v, topk=3)

    #         # 同一 utterance 的跨模态边
    #         Gnodes = []
    #         for t in range(i):
    #             Gnodes.append([nodes_l[t], nodes_a[t], nodes_v[t]])

    #         for group in Gnodes:
    #             tmp += list(permutations(group, 2))

    #         temp += i
    #         node_count += i * num_modality

    #     if len(index) > 0:
    #         intra_edges = torch.LongTensor(index).t().contiguous()
    #     else:
    #         intra_edges = torch.empty((2, 0), dtype=torch.long)

    #     if len(tmp) > 0:
    #         cross_edges = torch.LongTensor(tmp).t().contiguous()
    #     else:
    #         cross_edges = torch.empty((2, 0), dtype=torch.long)

    #     device = l.device
    #     edge_index = torch.cat([intra_edges, cross_edges], dim=1).to(device)

    #     return edge_index

    def reverse_features(self, dia_len, features):
        l = []
        a = []
        v = []

        for i in dia_len:
            ll = features[0:1 * i]
            aa = features[1 * i:2 * i]
            vv = features[2 * i:3 * i]
            features = features[3 * i:]
            l.append(ll)
            a.append(aa)
            v.append(vv)

        tmpl = torch.cat(l, dim=0)
        tmpa = torch.cat(a, dim=0)
        tmpv = torch.cat(v, dim=0)
        features = torch.cat([tmpl, tmpa, tmpv], dim=-1)
        return features