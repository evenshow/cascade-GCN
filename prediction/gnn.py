import math
"""GMNN 使用的图神经网络结构，包括推理网络 q 和生成网络 p。"""

import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
from layer import GraphConvolution


class GNNq(nn.Module):
    """GMNN 中的推理网络 q，用于估计标签分布。"""
    #GNNq的作用是Encoder,从节点的原始特征X和图结构A中推理出节点标签Y的概率分布
    def __init__(self, opt, adj):
        super(GNNq, self).__init__()
        self.opt = opt
        self.adj = adj

        opt_ = dict([('in', opt['num_feature']), ('out', opt['hidden_dim'])])
        self.m1 = GraphConvolution(opt_, adj)

        opt_ = dict([('in', opt['hidden_dim']), ('out', opt['num_class'])])
        self.m2 = GraphConvolution(opt_, adj)

        if opt['cuda']:
            self.cuda()

    def reset(self):
        self.m1.reset_parameters()
        self.m2.reset_parameters()

    def forward(self, x):
        """执行两层图卷积并返回 logits 及中间嵌入。"""
        x = F.dropout(x, self.opt['input_dropout'], training=self.training)
        x = self.m1(x)
        embedding = x.detach()
        x = F.relu(x)
        x = F.dropout(x, self.opt['dropout'], training=self.training)
        x = self.m2(x)
        return x, embedding


class GNNp(nn.Module):
    """GMNN 中的生成网络 p，拟合标签生成过程。"""
    #Decodr,从节点的标签分布出发，生成或重建这些标签（或者说学习标签之间的依赖关系）
    def __init__(self, opt, adj):
        super(GNNp, self).__init__()
        self.opt = opt
        self.adj = adj

        opt_ = dict([('in', opt['num_class']), ('out', opt['hidden_dim'])])
        self.m1 = GraphConvolution(opt_, adj)

        opt_ = dict([('in', opt['hidden_dim']), ('out', opt['num_class'])])
        self.m2 = GraphConvolution(opt_, adj)

        if opt['cuda']:
            self.cuda()

    def reset(self):
        self.m1.reset_parameters()
        self.m2.reset_parameters()

    def forward(self, x):
        """执行两层图卷积并返回 logits 及中间嵌入。"""
        x = F.dropout(x, self.opt['input_dropout'], training=self.training)
        x = self.m1(x)
        embedding = x.detach()
        x = F.relu(x)
        x = F.dropout(x, self.opt['dropout'], training=self.training)
        x = self.m2(x)
        return x, embedding
