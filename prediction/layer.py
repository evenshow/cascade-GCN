"""图神经网络层定义模块，包含基础的图卷积实现"""

import math
import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class GraphConvolution(nn.Module):
    """简单的图卷积层，使用邻接矩阵对特征进行传播。"""

    def __init__(self, opt, adj):
        super(GraphConvolution, self).__init__()
        # 保存配置与图结构，方便在前向传播中使用
        self.opt = opt
        self.in_size = opt['in']
        self.out_size = opt['out']
        self.adj = adj
        # 可训练的权重矩阵，尺寸为输入维度×输出维度
        self.weight = Parameter(torch.Tensor(self.in_size, self.out_size))
        self.reset_parameters()

    def reset_parameters(self):
        """初始化权重参数，采用均匀分布。"""
        stdv = 1. / math.sqrt(self.out_size)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        """图卷积的核心操作：先线性变换再进行邻接矩阵传播。"""
        # 线性变换，将输入特征映射到输出空间
        m = torch.mm(x, self.weight)
        # 稀疏矩阵乘法，完成邻居信息聚合
        m = torch.spmm(self.adj, m)
        return m
