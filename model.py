#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

import datetime
import math

import numpy
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
from tqdm import tqdm


# from entmax import entmax_bisect


class FindSimilarIntentSess(Module):
    def __init__(self, hidden_size):
        super(FindSimilarIntentSess, self).__init__()
        self.hidden_size = hidden_size
        self.neighbor_n = 3
        self.dropout40 = nn.Dropout(0.40)

    def compute_sim(self, sess_emb):
        # cos_sim2 = torch.cdist(sess_emb.unsqueeze(1), sess_emb.unsqueeze(0), p=2)
        # sess_emb = F.normalize(sess_emb, p=2, dim=0)
        fenzi = torch.matmul(sess_emb, sess_emb.permute(1, 0))  # 512*512
        fenmu_l = torch.sum(sess_emb * sess_emb + 0.000001, 1)
        fenmu_l = torch.sqrt(fenmu_l).unsqueeze(1)
        fenmu = torch.matmul(fenmu_l, fenmu_l.permute(1, 0))
        cos_sim = fenzi / fenmu  # 512*512
        # cos_sim = cos_sim2
        cos_sim = nn.Softmax(dim=-1)(cos_sim)

        return cos_sim

    # 定义高斯核函数

    def forward(self, sess_emb):
        k_v = self.neighbor_n

        cos_sim = self.compute_sim(sess_emb)
        # cos_sim = (sess_emb,sess_emb.permute(1, 0))
        # cos_sim = F.cosine_similarity(sess_emb.unsqueeze(1), sess_emb.unsqueeze(0), dim=2)
        # cos_sim2 = self.compute_jaccard_sim(sess_emb)
        # cos_sim2 = nn.Softmax(dim=-1)(cos_sim2)
        # cos_sim = cos_sim2+cos_sim
        # cos_sim = nn.Softmax(dim=-1)(cos_sim)
        # cos_sim =torch.nn.functional.cosine_similarity(sess_emb, sess_emb.permute(1, 0), dim=1, eps=1e-8)
        # cos_sim = nn.Softmax(dim=-1)(cos_sim)
        if cos_sim.size()[0] < k_v:
            k_v = cos_sim.size()[0]
        cos_topk, topk_indice = torch.topk(cos_sim, k=k_v, dim=1)
        cos_topk = nn.Softmax(dim=-1)(cos_topk)
        sess_topk = sess_emb[topk_indice]
        cos_sim2 = cos_topk.unsqueeze(2).expand(cos_topk.size()[0], cos_topk.size()[1], self.hidden_size)
        neighbor_sess = torch.sum(cos_sim2 * sess_topk, 1)
        neighbor_sess = self.dropout40(neighbor_sess)  # [b,d]

        cos_topk2, topk_indice2 = torch.topk(cos_sim, k=1, dim=1, largest=False)
        last = sess_emb[topk_indice2]
        return neighbor_sess, last



class PositionalEncoding(nn.Module):
    """位置编码"""

    # num_hiddens:向量长度，max_len：最长序列长度
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 创建一个足够长的P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)



class GNN(Module):
    def __init__(self, hidden_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = self.hidden_size * 2
        self.gate_size = 3 * self.hidden_size
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.Wg = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.Wg2 = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.Wg3 = nn.Linear(2 * self.hidden_size, self.hidden_size)

    def GNNCell(self, A, hidden):
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        inputs = torch.cat([input_in, input_out], 2)
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        return hy

    def forward(self, A, hidden):
        # for i in range(1):
        #     hidden = self.GNNCell(A, hidden)
        # return hidden
        hidden1 = F.normalize(self.GNNCell(A, hidden), p=2, dim=-1)
        hidden11 = hidden + hidden1
        hidden2 = F.normalize(self.GNNCell(A, hidden11), p=2, dim=-1)
        hidden22 = hidden11 + hidden2
        hidden3 = F.normalize(self.GNNCell(A, hidden22), p=2, dim=-1)
        return hidden22 + hidden3
        # hidden1 = F.normalize(self.GNNCell(A, hidden), p=2, dim=-1)
        # hidden11 = hidden + hidden1
        # hidden2 = F.normalize(self.GNNCell(A, hidden11), p=2, dim=-1)
        # hidden22 = hidden11 + hidden2
        # hidden3 = F.normalize(self.GNNCell(A, hidden22), p=2, dim=-1)
        # hidden33 = hidden22 + hidden3
        # hidden4 = F.normalize(self.GNNCell(A, hidden33), p=2, dim=-1)
        # hidden44 = hidden4+hidden33
        # hidden5 = F.normalize(self.GNNCell(A, hidden44), p=2, dim=-1)
        # return hidden2+hidden11

class SRSMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SRSMLP, self).__init__()
        self.linear_one = nn.Linear(input_size, hidden_size)
        self.linear_two = nn.Linear(hidden_size, output_size)
        self.linear_three = nn.Linear(hidden_size, output_size)

    def forward(self, hidden, mask, x):
        output = []
        session_lengths = torch.sum(mask, 1)
        batch_size = hidden.size(0)
        for i in range(batch_size):
            length = session_lengths[i]
            mm = hidden[:, length - x:length]
            ht = hidden[:, length - x:length].sum(dim=1)
        # for i in range(batch_size):
        #     length = session_lengths[i]
        #     ht = hidden[i, length-x:length,:].sum(dim=0, keepdims=True)
        #     output.append(ht)
        # output_tensor = torch.stack(output)
        # return output_tensor
        # q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
        # q2 = self.linear_two(hidden)  # batch_size x seq_length x latent_size
        # alpha = self.linear_three(torch.sigmoid(q1 + q2))
        # a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)
        return ht


class SessionGraph(Module):
    def __init__(self, opt, n_node):
        super(SessionGraph, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.batch_size = opt.batchSize
        self.nonhybrid = opt.nonhybrid
        # self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        # self.embedding2 = nn.Embedding(self.n_node, self.hidden_size)
        self.embedding = nn.Embedding(self.n_node, self.hidden_size, padding_idx=0, max_norm=1.5)
        self.pos_embedding = nn.Embedding(300, self.hidden_size, padding_idx=0, max_norm=1.5)
        # self.LayerNorm = LayerNorm(2 * self.hidden_size, eps=1e-12)
        self.gnn = GNN(self.hidden_size, step=opt.step)
        # self.gcn = GCN(self.hidden_size,self.hidden_size/2,1)
        # self.GAT = GAT(self.hidden_size,0.2)
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_one2 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear1 = nn.Linear(2 * self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two2 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.linear_three2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.linear_transform2 = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.w1 = Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.positional = PositionalEncoding(self.hidden_size, 0.2)
        self.FindNeighbor = FindSimilarIntentSess(self.hidden_size)
        self.w_ne = opt.w_ne
        # self.w_ne = Parameter(torch.Tensor(1))
        self.scale = self.hidden_size ** -0.5
        self.linear_sim = nn.Linear(self.hidden_size, self.hidden_size)
        self.ie_query = nn.Linear(2 * self.hidden_size, self.hidden_size, bias=False)
        self.ie_key = nn.Linear(2 * self.hidden_size, self.hidden_size, bias=False)
        self.ie_query2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.ie_key2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.w1 = Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.w2 = Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.w3 = Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.w4 = Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.linear = nn.Linear(2 * self.hidden_size, self.hidden_size, bias=False)
        # self.SRSMLP =SRSMLP(self.hidden_size,self.hidden_size,self.hidden_size)
        self.candidate_models = nn.ModuleList(
            [SRSMLP(self.hidden_size, self.hidden_size, self.hidden_size) for _ in range(6)])
        self.architectural_weights = nn.Parameter(torch.randn(6))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def SSL(self, sess_emb_hgnn, sess_emb_lgcn):
        def row_shuffle(embedding):
            corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
            return corrupted_embedding

        def row_column_shuffle(embedding):
            corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
            corrupted_embedding = corrupted_embedding[:, torch.randperm(corrupted_embedding.size()[1])]
            return corrupted_embedding

        def score(x1, x2):
            return torch.sum(torch.mul(x1, x2), 1)

        pos = score(sess_emb_hgnn, sess_emb_lgcn)
        negative_sample = torch.randn_like(sess_emb_hgnn)
        # neg1 = score(sess_emb_lgcn, row_column_shuffle(sess_emb_hgnn))
        neg1 = score(sess_emb_lgcn, negative_sample)
        one = torch.cuda.FloatTensor(neg1.shape[0]).fill_(1)
        # one = zeros = torch.ones(neg1.shape[0])
        con_loss = torch.sum(-torch.log(1e-8 + torch.sigmoid(pos)) - torch.log(1e-8 + (one - torch.sigmoid(neg1))))
        return con_loss

    def ContrastiveLoss(self, anchor, positive, negative):
        distance_pos = F.cosine_similarity(anchor, positive)
        distance_neg = F.cosine_similarity(anchor, negative)
        # loss = torch.mean(torch.relu(distance_neg - distance_pos + self.margin))
        loss = torch.mean(torch.relu(distance_pos - distance_neg + 1.0))
        return loss

    def row_column_shuffle(self, embedding):
        corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
        corrupted_embedding = corrupted_embedding[:, torch.randperm(corrupted_embedding.size()[1])]
        return corrupted_embedding

    def determine_window_size(self, length):
        # 这个函数用于确定窗口大小k，可以根据需要调整逻辑
        # 例如，可以设置固定的窗口大小或基于序列长度的函数
        # 下面是一个简单的例子，基于序列长度的平方根
        return max(1, int(torch.sqrt(torch.tensor(length)).item()))

        # return max(1, int(torch.pow(torch.tensor(length), 1 / 3).item()))

    def compute_scores(self, hidden, mask, seq_hidden2, mask2, test_train):
        outputs = torch.stack([model(hidden, mask, i) for model, i in zip(self.candidate_models, np.arange(1, 7))])
        # weight = F.softmax(outputs, dim = 0).unsqueeze(0).unsqueeze(-1)
        weight = F.softmax(self.architectural_weights, dim = 0).unsqueeze(0).unsqueeze(-1)
        final_output = torch.sum(weight*outputs.permute(1,0,2), dim=1)

        outputs2 = torch.stack(
            [model(seq_hidden2, mask2, i) for model, i in zip(self.candidate_models, np.arange(1, 7))])
        weight2 = F.softmax(self.architectural_weights, dim=0).unsqueeze(0).unsqueeze(-1)
        final_output2 = torch.sum(weight2*outputs2.permute(1,0,2), dim=1)


        session_lengths = torch.sum(mask, 1)
        batch_size, seq_length, _ = hidden.size()
        fused_embedding = torch.zeros(batch_size, self.hidden_size).to(hidden.device)

        # for i in range(batch_size):
        #     length = session_lengths[i]
        #     # 根据长度确定窗口大小k
        #     k = min(length, self.determine_window_size(length))
        #     # 获取最后k个项目的嵌入并进行平均
        #     # fused_embedding[i] = hidden[i, length - k:length].mean(dim=0)
        #     fused_embedding[i] = hidden[i, length - k:length].sum(dim=0)
        # ht = fused_embedding
        if test_train == 2:
            best_order = torch.argmax(weight).item()
            final_output = outputs.permute(1, 0, 2)[:, best_order]
            best_order2 = torch.argmax(weight2).item()
            final_output2 = outputs2.permute(1, 0, 2)[:, best_order2]
        ht = final_output
        # ht = torch.sum(hidden, 1) / torch.sum(mask, 1).unsqueeze(-1)

        # ht = torch.sum(hidden, 1) / torch.sum(mask, 1).unsqueeze(-1)
        # ht = torch.mean(hidden, dim=1)
        # cc = torch.cat(hidden,dim=1)
        # batch_size x latent_size
        # ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]
        # ht2 = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 2]  # batch_size x latent_size
        # ht3 = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 3]  # batch_size x latent_size
        # ht = ht1 + ht2 + ht3
        # ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]

        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
        q2 = self.linear_two(hidden)  # batch_size x seq_length x latent_size
        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        a = torch.sum(alpha * hidden * mask.view(mask.shape[0], -1, 1).float(), 1)
        batch_size, seq_length, _ = seq_hidden2.size()


        # for i in range(batch_size):
        #     length = session_lengths2[i]
        #     # 根据长度确定窗口大小k
        #     k = min(length, self.determine_window_size(length))
        #     # 获取最后k个项目的嵌入并进行平均
        #     # fused_embedding2[i] = seq_hidden2[i, length - k:length].mean(dim=0)
        #     fused_embedding2[i] = seq_hidden2[i, length - k:length].sum(dim=0)
        #
        # ht = fused_embedding2
        ht2 = final_output2

        # ht2 = torch.sum(seq_hidden2, 1) / torch.sum(mask2, 1).unsqueeze(-1)
        # ht = torch.mean(seq_hidden2, dim=1)
        # ht2 = seq_hidden2[torch.arange(mask2.shape[0]).long(), torch.sum(mask2, 1) - 1]
        # ht2 = seq_hidden2[torch.arange(mask2.shape[0]).long(), torch.sum(mask2, 1) - 2]  # batch_size x latent_size
        # ht3 = seq_hidden2[torch.arange(mask2.shape[0]).long(), torch.sum(mask2, 1) - 3]  # batch_size x latent_size
        # ht = ht1 + ht2 + ht3
        q12 = self.linear_one2(ht2).view(ht2.shape[0], 1, ht2.shape[1])  # batch_size x 1 x latent_size
        q22 = self.linear_two2(seq_hidden2)  # batch_size x seq_length x latent_size
        alpha2 = self.linear_three2(torch.sigmoid(q12 + q22))
        a2 = torch.sum(alpha2 * seq_hidden2 * mask2.view(mask2.shape[0], -1, 1).float(), 1)
        b = self.embedding.weight[1:]  # n_nodes x latent_size

        anchor = a
        positive_sample = a2

        neighbor_sess, last = self.FindNeighbor(anchor)
        s_before = neighbor_sess.clone()
        # print("对比学习前：", s_before)
        neighbor_sess2, last2 = self.FindNeighbor(positive_sample)
        negative_sample = torch.randn_like(neighbor_sess)
        con_loss = self.ContrastiveLoss(neighbor_sess, neighbor_sess2, negative_sample)
        s_after = neighbor_sess.clone()
        # print("对比学习后：",   s_after)
        a3 = a + 7 * neighbor_sess
        # con_loss = 0
        scores = torch.matmul(a3, b.transpose(1, 0))
        return scores, con_loss, neighbor_sess

    def forward(self, inputs, A):
        hidden = self.embedding(inputs)
        hidden = self.gnn(A, hidden)
        return hidden


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, i, data, test_train):
    alias_inputs, A, items, mask, targets, alias_inputs2, mask2 = data.get_slice(i)
    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
    alias_inputs2 = trans_to_cuda(torch.Tensor(alias_inputs2).long())
    items = trans_to_cuda(torch.Tensor(items).long())
    A = trans_to_cuda(torch.Tensor(A).float())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    mask2 = trans_to_cuda(torch.Tensor(mask2).long())
    hidden = model(items, A)
    get = lambda i: hidden[i][alias_inputs[i]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
    get2 = lambda i: hidden[i][alias_inputs2[i]]
    seq_hidden2 = torch.stack([get2(i) for i in torch.arange(len(alias_inputs2)).long()])
    scores, con_loss,neighbor_sess = model.compute_scores(seq_hidden, mask, seq_hidden2, mask2, test_train)
    return targets, scores, con_loss,neighbor_sess


def train_test(model, train_data, test_data):
    model.scheduler.step()
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    for i, j in zip(tqdm(slices), np.arange(len(slices))):
        model.optimizer.zero_grad()
        targets, scores, con_loss,neighbor_sess = forward(model, i, train_data,1)
        targets = trans_to_cuda(torch.Tensor(targets).long())
        loss = model.loss_function(scores, targets - 1)
        loss = loss + con_loss
        s_before = neighbor_sess.clone()
        loss.backward()
        model.optimizer.step()
        s_after = neighbor_sess.clone()
        # print("befor:",s_before)
        # print("after:",s_after)
        total_loss += loss
        if j % int(len(slices) / 5 + 1) == 0:
            print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))
    print('\tLoss:\t%.3f' % total_loss)
    print('start predicting: ', datetime.datetime.now())
    model.eval()
    top_K = [5, 10, 20]
    metrics = {}
    for k in top_K:
        metrics['hit%d' % k] = []
        metrics['mrr%d' % k] = []

    # metrics = {'hit%d' % k: [] for k in top_K}
    # hit, mrr = [], []

    slices = test_data.generate_batch(model.batch_size)
    for i in tqdm(slices):
        targets, scores, con_loss,neighbor_sess = forward(model, i, test_data, 1)
        for k in top_K:
            sub_scores = scores.topk(k)[1]
            sub_scores = trans_to_cpu(sub_scores).detach().numpy()
            for score, target, mask in zip(sub_scores, targets, test_data.mask):
                metrics['hit%d' % k].append(np.isin(target - 1, score))
                if len(np.where(score == target - 1)[0]) == 0:
                    metrics['mrr%d' % k].append(0)
                else:
                    metrics['mrr%d' % k].append(1 / (np.where(score == target - 1)[0][0] + 1))
        # metrics['hit%d' % k] = np.mean(metrics['hit%d' % k]) * 100
        # metrics['mrr%d' % k] = np.mean(metrics['mrr%d' % k]) * 100
    return metrics


# def test(model, test_data):
#     model.eval()
#     hit5, mrr5 = [], []
#     hit10, mrr10 = [], []
#     hit20, mrr20 = [], []
#     slices = test_data.generate_batch(model.batch_size)
#     for i in slices:
#         targets, scores = forward(model, i, test_data)
#         sub_scores = scores.topk(5)[1]
#         sub_scores = trans_to_cpu(sub_scores).detach().numpy()
#         for score, target, mask in zip(sub_scores, targets, test_data.mask):
#             hit5.append(np.isin(target - 1, score))
#             if len(np.where(score == target - 1)[0]) == 0:
#                 mrr5.append(0)
#             else:
#                 mrr5.append(1 / (np.where(score == target - 1)[0][0] + 1))
#     for i in slices:
#         targets, scores = forward(model, i, test_data)
#         sub_scores = scores.topk(10)[1]
#         sub_scores = trans_to_cpu(sub_scores).detach().numpy()
#         for score, target, mask in zip(sub_scores, targets, test_data.mask):
#             hit10.append(np.isin(target - 1, score))
#             if len(np.where(score == target - 1)[0]) == 0:
#                 mrr10.append(0)
#             else:
#                 mrr10.append(1 / (np.where(score == target - 1)[0][0] + 1))
#     for i in slices:
#         targets, scores = forward(model, i, test_data)
#         sub_scores = scores.topk(20)[1]
#         sub_scores = trans_to_cpu(sub_scores).detach().numpy()
#         for score, target, mask in zip(sub_scores, targets, test_data.mask):
#             hit20.append(np.isin(target - 1, score))
#             if len(np.where(score == target - 1)[0]) == 0:
#                 mrr20.append(0)
#             else:
#                 mrr20.append(1 / (np.where(score == target - 1)[0][0] + 1))
#     hit5 = np.mean(hit5) * 100
#     mrr5 = np.mean(mrr5) * 100
#     hit10 = np.mean(hit10) * 100
#     mrr10 = np.mean(mrr10) * 100
#     hit20 = np.mean(hit20) * 100
#     mrr20 = np.mean(mrr20) * 100
#     return hit5, mrr5, hit10, mrr10, hit20, mrr20
