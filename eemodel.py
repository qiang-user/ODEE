import random
import math
import torch
import torch.nn as nn
from transformers import AutoModel
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import torch.nn.functional as F
import numpy as np
from collections import defaultdict


class LayerNorm(nn.Module):
    def __init__(self, input_dim, cond_dim=0, center=True, scale=True, epsilon=None, conditional=False,
                 hidden_units=None, hidden_activation='linear', hidden_initializer='xaiver', **kwargs):
        super(LayerNorm, self).__init__()
        """
        input_dim: inputs.shape[-1]
        cond_dim: cond.shape[-1]
        """
        self.center = center
        self.scale = scale
        self.conditional = conditional
        self.hidden_units = hidden_units
        # self.hidden_activation = activations.get(hidden_activation) keras中activation为linear时，返回原tensor,unchanged
        self.hidden_initializer = hidden_initializer
        self.epsilon = epsilon or 1e-12
        self.input_dim = input_dim
        self.cond_dim = cond_dim

        if self.center:
            self.beta = nn.Parameter(torch.zeros(input_dim))
        if self.scale:
            self.gamma = nn.Parameter(torch.ones(input_dim))

        if self.conditional:
            if self.hidden_units is not None:
                self.hidden_dense = nn.Linear(in_features=self.cond_dim, out_features=self.hidden_units, bias=False)
            if self.center:
                self.beta_dense = nn.Linear(in_features=self.cond_dim, out_features=input_dim, bias=False)
            if self.scale:
                self.gamma_dense = nn.Linear(in_features=self.cond_dim, out_features=input_dim, bias=False)

        self.initialize_weights()

    def initialize_weights(self):

        if self.conditional:
            if self.hidden_units is not None:
                if self.hidden_initializer == 'normal':
                    torch.nn.init.normal(self.hidden_dense.weight)
                elif self.hidden_initializer == 'xavier':  # glorot_uniform
                    torch.nn.init.xavier_uniform_(self.hidden_dense.weight)

            if self.center:
                torch.nn.init.constant_(self.beta_dense.weight, 0)
            if self.scale:
                torch.nn.init.constant_(self.gamma_dense.weight, 0)

    def forward(self, inputs, cond=None):
        """
            如果是条件Layer Norm，则cond不是None
        """
        if self.conditional:
            if self.hidden_units is not None:
                cond = self.hidden_dense(cond)

            for _ in range(len(inputs.shape) - len(cond.shape)):
                cond = cond.unsqueeze(1)  # cond = K.expand_dims(cond, 1)

            if self.center:
                beta = self.beta_dense(cond) + self.beta
            if self.scale:
                gamma = self.gamma_dense(cond) + self.gamma
        else:
            if self.center:
                beta = self.beta
            if self.scale:
                gamma = self.gamma

        outputs = inputs
        if self.center:
            mean = torch.mean(outputs, dim=-1).unsqueeze(-1)
            outputs = outputs - mean
        if self.scale:
            variance = torch.mean(outputs ** 2, dim=-1).unsqueeze(-1)
            std = (variance + self.epsilon) ** 0.5
            outputs = outputs / std
            outputs = outputs * gamma
        if self.center:
            outputs = outputs + beta

        return outputs


class Biaffine(nn.Module):
    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True):
        super(Biaffine, self).__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        weight = torch.zeros((n_out, n_in + int(bias_x), n_in + int(bias_y)))
        nn.init.xavier_normal_(weight)
        self.weight = nn.Parameter(weight, requires_grad=True)

    def extra_repr(self):
        s = f"n_in={self.n_in}, n_out={self.n_out}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"

        return s

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        # [batch_size, n_out, seq_len, seq_len]
        s = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y)
        # remove dim 1 if n_out == 1
        s = s.permute(0, 2, 3, 1)

        return s


class Predictor(nn.Module):
    def __init__(self, cls_num, hid_size, biaffine_size, dropout=0.):
        super().__init__()
        self.mlp_sub = MLP(n_in=hid_size, n_out=biaffine_size, dropout=dropout)
        self.mlp_obj = MLP(n_in=hid_size, n_out=biaffine_size, dropout=dropout)
        self.biaffine = Biaffine(n_in=biaffine_size, n_out=cls_num, bias_x=True, bias_y=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y):
        ent_sub = self.dropout(self.mlp_sub(x))
        ent_obj = self.dropout(self.mlp_obj(y))

        outputs = self.biaffine(ent_sub, ent_obj)

        return outputs


class MLP(nn.Module):
    def __init__(self, n_in, n_out, dropout=0.):
        super().__init__()

        self.linear = nn.Linear(n_in, n_out)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear(x)
        x = self.activation(x)
        return x


class FFNN(nn.Module):
    def __init__(self, input_dim, hid_dim, cls_num, dropout=0):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hid_dim)
        self.linear2 = nn.Linear(hid_dim, cls_num)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.activation(x)
        x = self.linear2(x)

        return x


class AdaptiveFusion(nn.Module):
    def __init__(self, hid_size, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        # q_linear 输入维度是768.输出维度也是768
        self.q_linear = nn.Linear(hid_size, hid_size)
        # k_linear 输入768 输出1536
        self.k_linear = nn.Linear(hid_size, hid_size * 2)

        self.factor = math.sqrt(hid_size)

        self.gate1 = Gate(hid_size, dropout=dropout)
        self.gate2 = Gate(hid_size, dropout=dropout)

    # x 输入的句子中的词的embding--为论文中的H  g ：所有的事件信息组成的embing--为论文中的E  s:目标事件的embding---e下标记t
    def forward(self, x, s, g):
        # x [B, L, H]
        # s [B, K, H]
        # g [B, N, H]
        # x = self.dropout(x)
        # s = self.dropout(s)
        q = self.q_linear(x)
        k_v = self.k_linear(g)
        k, v = torch.chunk(k_v, chunks=2, dim=-1)
        scores = torch.bmm(q, k.transpose(1, 2)) / self.factor
        # scores = self.dropout(scores)
        scores = torch.softmax(scores, dim=-1)
        #  8*145*768   使用注意力机制得到的词向量表示
        g = torch.bmm(scores, v)
        # g为论文中的等式4Eg
        g = g.unsqueeze(2).expand(-1, -1, s.size(1), -1)
        h = x.unsqueeze(2).expand(-1, -1, s.size(1), -1)
        s = s.unsqueeze(1).expand(-1, x.size(1), -1, -1)
        # 左边的h为论文中等式5的h上标记g
        h = self.gate1(h, g)
        # 左边的h为论文中等式6左边的V上标t---即event-aware word representations--
        h = self.gate2(h, s)
        return h


class Gate(nn.Module):
    def __init__(self, hid_size, dropout=0.2):
        super().__init__()
        self.linear = nn.Linear(hid_size * 2, hid_size)
        self.dropout = nn.Dropout(dropout)
        # self.linear2 = nn.Linear(hid_size, hid_size)

    def forward(self, x, y):
        '''
        :param x: B, L, K, H
        :param y: B, L, K, H
        :return:
        '''
        o = torch.cat([x, y], dim=-1)
        o = self.dropout(o)
        gate = self.linear(o)
        gate = torch.sigmoid(gate)
        o = gate * x + (1 - gate) * y
        # o = F.gelu(self.linear2(self.dropout(o)))
        return o


# 计算trigger矩阵
class triggerReLU(nn.Module):
    def __init__(self, hid_size, dropout=0.2):
        super(triggerReLU, self).__init__()
        self.linear = nn.Linear(hid_size * 2, hid_size)
        self.dropout = nn.Dropout(dropout)
        self.tri_matric = nn.Linear(hid_size, 1)

    def forward(self, x, y, training=False):
        # x的维度 8*26*768  事件类型拼成的句子的embedding
        # y的维度 8*100*768 原来的句子的embedding表示
        batch_size, tokens1, hid_size = x.size()
        _, tokens2, _ = y.size()
        pos_emb_x = _sinusoidal_position_embedding(batch_size, tokens1, hid_size)
        pos_emb_y = _sinusoidal_position_embedding(batch_size, tokens2, hid_size)

        cos_pos_x = pos_emb_x[..., 1::2].repeat_interleave(2, dim=-1)
        sin_pos_x = pos_emb_x[..., ::2].repeat_interleave(2, dim=-1)
        cos_pos_y = pos_emb_y[..., 1::2].repeat_interleave(2, dim=-1)
        sin_pos_y = pos_emb_y[..., ::2].repeat_interleave(2, dim=-1)

        x2 = torch.stack([-x[..., 1::2], x[..., ::2]], -1).reshape(x.shape)
        x = x * cos_pos_x + x2 * sin_pos_x
        y2 = torch.stack([-y[..., 1::2], y[..., ::2]], -1).reshape(y.shape)
        y = y * cos_pos_y + y2 * sin_pos_y
        # 这个例子自己好好理解一下，并举个具体的数字
        result = []
        for i in range(batch_size):
            a = x[i]  # 对应 batch 的子张量 A
            b = y[i]  # 对应 batch 的子张量 B

            # 扩展维度
            a_expanded = a.unsqueeze(1).expand(-1, b.size(0), -1)  # 形状: 2*3*4
            b_expanded = b.unsqueeze(0).expand(a.size(0), -1, -1)  # 形状: 2*3*4

            # 沿最后一个维度拼接
            concatenated = torch.cat((a_expanded, b_expanded), dim=-1)  # 形状: 2*3*8
            result.append(concatenated)

        # 将结果列表转换为张量
        result = torch.stack(result)
        seq_len1 = result.size(1)
        seq_len2 = result.size(2)
        result = result.reshape(batch_size, seq_len1 * seq_len2, result.size(-1))  # 形状: 2*2*3*8
        result = self.linear(result)
        result = self.dropout(result)
        result = torch.relu(result)

        result = self.tri_matric(result)

        if training == False:
            return torch.sigmoid(result)
        return result


# 计算argument矩阵
class argReLU(nn.Module):
    def __init__(self, hid_size, dropout=0.1):
        super(argReLU, self).__init__()
        self.d_model = 768  # 输入和输出的特征维度
        self.nhead = 8  # 注意力机制的头数
        self.dim_feedforward = 2048  # 前馈神经网络的隐藏层大小
        self.dropout_rate = 0.2  # dropout率
        self.linear = nn.Linear(hid_size, 18)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead,
                                                        dim_feedforward=self.dim_feedforward,
                                                        dropout=self.dropout_rate)

    def forward(self, y, training=False):
        batch_size, tokens, hid_size = y.size()
        pos_emb_y = _sinusoidal_position_embedding(batch_size, tokens, hid_size)

        cos_pos_y = pos_emb_y[..., 1::2].repeat_interleave(2, dim=-1)
        sin_pos_y = pos_emb_y[..., ::2].repeat_interleave(2, dim=-1)

        y2 = torch.stack([-y[..., 1::2], y[..., ::2]], -1).reshape(y.shape)
        y = y * cos_pos_y + y2 * sin_pos_y
        # todo casEE---是不是这块可能有问题呢？
        src = y.permute(1, 0, 2)  # (序列长度, 批次大小, 特征维度)
        x = self.encoder_layer(src).permute(1, 0, 2)
        batch_size, tokens2, _ = y.size()
        x = self.linear(x).permute(0, 2, 1)
        x = self.dropout(x)

        if training == False:
            return torch.sigmoid(x)
        # x = x.permute(0, 2, 1)
        return x


# 计算触发词与论元的关系矩阵
# 还是因为关系的数量太多？--或者解码根本不对呢
class roleReLU(nn.Module):
    # dropout的影响
    def __init__(self, hid_size, dropout=0.2):
        super(roleReLU, self).__init__()
        self.linear = nn.Linear(hid_size * 2, hid_size)
        self.dropout = nn.Dropout(dropout)
        self.rel_size = 65
        self.tag_size = 4
        self.relation_matrix = nn.Linear(hid_size, self.rel_size * self.tag_size)

    def forward(self, y, event_idx, training=False):

        # batch_size, tokens2, feature_dim = y.size()
        # head_representation = y.unsqueeze(2).expand(batch_size, tokens2, tokens2, feature_dim).reshape(batch_size, tokens2*tokens2, feature_dim)
        # tail_representation = y.repeat(1, tokens2, 1)
        # entity_pairs = torch.cat([head_representation, tail_representation], dim=-1)
        batch_size, tokens, hid_size = y.size()
        pos_emb_y = _sinusoidal_position_embedding(batch_size, tokens, hid_size)
        cos_pos_y = pos_emb_y[..., 1::2].repeat_interleave(2, dim=-1)
        sin_pos_y = pos_emb_y[..., ::2].repeat_interleave(2, dim=-1)
        y2 = torch.stack([-y[..., 1::2], y[..., ::2]], -1).reshape(y.shape)
        y = y * cos_pos_y + y2 * sin_pos_y
        result = []
        if training == True:
            for i in range(batch_size):
                # 有没有可能这块出问题了呢？---
                a = y[i][event_idx[i], :]  # 对应 batch 的子张量 A
                b = y[i][event_idx[i], :]  # 对应 batch 的子张量 B
                # 扩展维度
                a_expanded = a.unsqueeze(1).expand(-1, b.size(0), -1)  # 形状: 2*3*4
                b_expanded = b.unsqueeze(0).expand(a.size(0), -1, -1)  # 形状: 2*3*4
                # 沿最后一个维度拼接
                concatenated = torch.cat((a_expanded, b_expanded), dim=-1)  # 形状: 2*3*8
                result.append(concatenated)
        else:
            for i in range(batch_size):
                a = y[i]  # 对应 batch 的子张量 A
                b = y[i]  # 对应 batch 的子张量 B
                # 扩展维度
                a_expanded = a.unsqueeze(1).expand(-1, b.size(0), -1)  # 形状: 2*3*4
                b_expanded = b.unsqueeze(0).expand(a.size(0), -1, -1)  # 形状: 2*3*4
                # 沿最后一个维度拼接
                concatenated = torch.cat((a_expanded, b_expanded), dim=-1)  # 形状: 2*3*8
                result.append(concatenated)
        # 将结果列表转换为张量
        result = torch.stack(result)
        seq_len = result.size(1)
        result = result.reshape(batch_size, seq_len * result.size(1), result.size(-1))
        result = result.cuda()
        # 有没有可能问题出现在这里，每次你都用同一个linear嘛
        # self.gates = nn.ModuleList([nn.Linear(4*768*2, 4) for i in range(12)])
        result = self.linear(result)
        result = self.dropout(result)
        result = torch.relu(result)
        result = self.relation_matrix(result).reshape(batch_size, seq_len, seq_len, self.rel_size,
                                                      self.tag_size).permute(0, 3, 1, 2, 4)

        if training == False:
            return torch.sigmoid(result)
        return result


class myModel(nn.Module):
    def __init__(self, config):
        super(myModel, self).__init__()
        self.inner_dim = config.tri_hid_size
        self.tri_hid_size = config.tri_hid_size
        self.eve_hid_size = config.eve_hid_size
        self.event_num = config.tri_label_num
        self.role_num = config.rol_label_num
        self.teacher_forcing = True
        # gamma参数含义
        self.gamma = config.gamma
        self.arg_hid_size = config.arg_hid_size
        # self.layers = config.layers

        self.bert = AutoModel.from_pretrained(config.bert_name, cache_dir="./cache/", output_hidden_states=True)

        # self.tri_hid_size = 256
        # self.arg_hid_size = 384
        self.dropout = nn.Dropout(config.dropout)
        self.tri_linear = nn.Linear(config.bert_hid_size, self.tri_hid_size * 2)
        self.arg_linear = nn.Linear(config.bert_hid_size, self.arg_hid_size * 2)
        self.role_linear = nn.Linear(config.bert_hid_size, config.eve_hid_size * config.rol_label_num * 2)
        # 第一个参数表示词汇表的大小 ，第二个参数表示嵌入向量的大小
        # 这指定了填充标记的索引。这个索引的嵌入向量将被初始化为全零，并且在训练期间不会被更新。这对于处理变长序列并使用填充使它们等长是很有用的
        # 初始化时就把事件类型的词向量进行了表示
        self.eve_embedding = nn.Embedding(self.event_num + 1, config.bert_hid_size,
                                          padding_idx=self.event_num)  # .from_pretrained(self.reset_event_parameters(config.vocab, config.tokenizer), freeze=False)

        self.gate = AdaptiveFusion(config.bert_hid_size, dropout=config.dropout)

        self.triggerRelu = triggerReLU(config.bert_hid_size, dropout=config.dropout)
        self.argumentRelu = argReLU(config.bert_hid_size, dropout=config.dropout)
        self.roleReLu = roleReLU(config.bert_hid_size, dropout=config.dropout)

    def reset_event_parameters(self, vocab, tokenizer):
        labels = [vocab.tri_id2label[i] for i in range(self.event_num)]
        inputs = tokenizer(labels)
        input_ids = pad_sequence([torch.LongTensor(x) for x in inputs["input_ids"]], True)
        attention_mask = pad_sequence([torch.BoolTensor(x) for x in inputs["attention_mask"]], True)
        mask = pad_sequence([torch.BoolTensor(x[1:-1]) for x in inputs["attention_mask"]], True)
        bert_embs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_embs = bert_embs[0][:, 1:-1]
        min_value = bert_embs.min().item()
        bert_embs = torch.masked_fill(bert_embs, mask.eq(0).unsqueeze(-1), min_value)
        bert_embs, _ = torch.max(bert_embs, dim=1)
        bert_embs = torch.cat([bert_embs, torch.zeros((1, bert_embs.size(-1)))], dim=0)
        return bert_embs.detach()

    # 位置编码，最后返回结果8*133*256
    def _sinusoidal_position_embedding(self, batch_size, seq_len, output_dim):
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)

        # 生成一个指数衰减的正弦和余弦函数的周期，用于位置编码，其中 output_dim是位置编码的维度
        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.repeat((batch_size, *([1] * len(embeddings.shape))))
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
        embeddings = embeddings.cuda()
        return embeddings

    # 注意力得分logits--应该是论文中的Score--维度8*6*133*133
    # qw是8*97*6*256  word_mask2d是8*97*97---qw是词向量的表示
    def _pointer(self, qw, kw, word_mask2d):
        B, L, K, H = qw.size()
        # 一句话长度为97  （维度8*97*256）这句话中每个词的词向量表示--开始这8个样本都是一样的
        pos_emb = self._sinusoidal_position_embedding(B, L, H)
        # cos_pos,sin_pos: (batch_size, seq_len, 1, inner_dim)
        cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
        sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
        qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
        qw2 = qw2.reshape(qw.shape)
        qw = qw * cos_pos + qw2 * sin_pos
        # 位置信息和vt信息进行融合
        kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
        kw2 = kw2.reshape(kw.shape)
        kw = kw * cos_pos + kw2 * sin_pos

        # 计算下来的维度8*6*105*105
        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)

        grid_mask2d = word_mask2d.unsqueeze(1).expand(B, K, L, L).float()
        # logits代表了Cij分数矩阵
        logits = logits * grid_mask2d - (1 - grid_mask2d) * 1e12
        return logits

    def forward(self, inputs, att_mask, word_mask1d, tri_labels, arg_labels, role_labels, event_idx
                ):
        """
        :param inputs: [B, L]
        :param att_mask: [B, L]
        :param word_mask1d: [B, L]
        :param word_mask2d: [B, L, L]
        :param span_labels: [B, L, L, 2], [..., 0] is trigger span label, [..., 1] is argument span label
        :param tri_labels: [B, L, L, C]
        :param event_mask: [B, L]
        :param prob: float (0 - 1)
        :return:
        """
        outputs = {}

        # 一个句子的长度
        L = word_mask1d.size(1)
        # inputs---8*147
        bert_embs = self.bert(input_ids=inputs, attention_mask=att_mask)  #
        # BERT模型的第一个元素输出的是最后一层的隐藏状态
        bert_embs = bert_embs[0]
        B, _, H = bert_embs.size()
        # 8*223*768不包括cls和sep
        bert_embs = bert_embs[:, 1:1 + L]
        B, L, H = bert_embs.size()

        if self.training:
            x = bert_embs
            # x = self.dropout(bert_embs)
            # 获取事件类型组成的句子的embedding结果 8*26*768
            Het = x[:, 0:26, :]
            # 获取本来的句子组成的embedding结果 8*100*768
            Hsentence = x[:, 26:, :]
            # 单纯计算出来之后,mask这些东西你还没有使用呀,
            # 8*(26*171)*1
            tri_logits = self.triggerRelu(Het, Hsentence, self.training)
            # 8*18*171
            arg_logits = self.argumentRelu(Hsentence, self.training)
            # 8*64*69*69*4
            role_logits = self.roleReLu(Hsentence, event_idx, self.training)
            return tri_logits, arg_logits, role_logits
        else:
            x = bert_embs
            # x = self.dropout(bert_embs)
            # 获取事件类型组成的句子的embedding结果 8*26*768
            Het = x[:, 0:26, :]
            # 获取本来的句子组成的embedding结果 8*100*768
            Hsentence = x[:, 26:, :]
            # tri_logits = self.triggerRelu(Het, Hsentence,self.training)
            # arg_logits = self.argumentRelu(Hsentence,self.training)
            role_logits = self.roleReLu(Hsentence, event_idx, self.training)
            # tri_mask1d = word_mask1d.unsqueeze(1).expand(-1, tri_logits.size(1), -1)[:,:,tri_logits.size(1):]
            # tri_mask1d_copy = word_mask1d[:,26:].unsqueeze(1).expand(-1, tri_logits.size(1), -1)

            # arg_mask1d = word_mask1d.unsqueeze(1).expand(-1, arg_logits.size(1), -1)[:,:,tri_logits.size(1):]
            role_mask1d = word_mask1d[:, 26:].unsqueeze(1).expand(-1, role_logits.size(2), -1).unsqueeze(1).expand(-1,
                                                                                                                   65,
                                                                                                                   -1,
                                                                                                                   -1).unsqueeze(
                1).expand(-1, 4, -1, -1, -1).permute(0, 2, 3, 4, 1)
            # tri_b_index--第几个样本索引，tri_e_index-事件类型索引  tri_word_index-触发词索引
            # tri_b_index, tri_e_index, tri_word_index = ((tri_logits > 0.5).long() + tri_mask1d.long()).eq(2).nonzero(
            #     as_tuple=True)
            # arg_b_index, arg_role_index, arg_argument_index = ((arg_logits > 0.5).long() + arg_mask1d.long()).eq(
            #     2).nonzero(as_tuple=True)

            # role_b_index, role_role_index, role_trigger_index, role_arg_index, role_v_index = (
            #     (role_logits > 0.5).long()+role_mask1d.long()).eq(2).nonzero(as_tuple=True)

            # _gold_tuples["ti"].add((t_s, t_e))
            # _gold_tuples["tc"].add((t_s, t_e, event_type))
            # _gold_tuples["ai"].add((a_s, a_e, event_type))
            # _gold_tuples["ac"].add((a_s, a_e, event_type, role))
            # 在最后一个维度上进行拼接得到结果 例如x(1,2,3) y(4,5,6) 拼接后结果((1,4),(2,5),(3,6))--对于ti--会有不同事件类型共同使用同一个触发词
            # outputs["ti"] = torch.cat([x.unsqueeze(-1) for x in [tri_b_index, tri_word_index]],
            #                           dim=-1).cpu().numpy()

            # outputs["tc"] = torch.cat([x.unsqueeze(-1) for x in [tri_b_index, tri_e_index, tri_word_index]],
            #                           dim=-1).cpu().numpy()
            #
            # outputs["ai"] = torch.cat([x.unsqueeze(-1) for x in [arg_b_index, arg_role_index, arg_argument_index]],
            #                           dim=-1).cpu().numpy()
            #
            # # outputs["ac"] = None
            # outputs["ac"] = torch.cat([x.unsqueeze(-1) for x in
            #                            [role_b_index, role_role_index, role_trigger_index, role_arg_index,
            #                             role_v_index]],
            #                           dim=-1).cpu().numpy()
            return role_logits, word_mask1d


def _sinusoidal_position_embedding(batch_size, seq_len, output_dim):
    position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)
    # 生成一个指数衰减的正弦和余弦函数的周期，用于位置编码，其中 output_dim是位置编码的维度
    indices = torch.arange(0, output_dim // 2, dtype=torch.float)
    indices = torch.pow(10000, -2 * indices / output_dim)
    embeddings = position_ids * indices
    embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
    embeddings = embeddings.repeat((batch_size, *([1] * len(embeddings.shape))))
    embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
    embeddings = embeddings.cuda()
    return embeddings
