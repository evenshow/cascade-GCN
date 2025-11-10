import math
import numpy as np
"""GMNN 训练过程的封装工具，包含优化器与训练循环，加入中文注释。"""

import torch
from torch import nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim import Optimizer


def get_optimizer(name, parameters, lr, weight_decay=0):
    """根据名称返回对应的优化器实例。"""
    if name == 'sgd':
        return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'rmsprop':
        return torch.optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adagrad':
        return torch.optim.Adagrad(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adam':
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adamax':
        return torch.optim.Adamax(parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise Exception("Unsupported optimizer: {}".format(name))


def change_lr(optimizer, new_lr):
    """动态调整学习率。"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


class Trainer(object):
    """封装 GMNN 的训练、评估与预测逻辑。"""
    def __init__(self, opt, model):
        self.opt = opt
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        if opt['cuda']:
            self.criterion.cuda()
        self.optimizer = get_optimizer(self.opt['optimizer'], self.parameters, self.opt['lr'], self.opt['decay'])

    def reset(self):
        """重置模型参数与优化器。"""
        self.model.reset()
        self.optimizer = get_optimizer(self.opt['optimizer'], self.parameters, self.opt['lr'], self.opt['decay'])

    def update(self, inputs, target, idx):
        """在指定节点集合上执行一次标准的交叉熵优化。"""
        if self.opt['cuda']:
            inputs = inputs.cuda()
            target = target.cuda()
            idx = idx.cuda()

        self.model.train()
        self.optimizer.zero_grad()

        logits = self.model(inputs)
        loss = self.criterion(logits[idx], target[idx])

        loss.backward()
        self.optimizer.step()
        return loss.item()

    def update_soft(self, inputs, target, idx):
        """使用软标签进行优化，同时返回节点嵌入。"""
        if self.opt['cuda']:
            inputs = inputs.cuda()
            target = target.cuda()
            idx = idx.cuda()

        self.model.train()
        self.optimizer.zero_grad()

        logits, embedding = self.model(inputs)
        logits = torch.log_softmax(logits, dim=-1)
        loss = -torch.mean(torch.sum(target[idx] * logits[idx], dim=-1))

        loss.backward()
        self.optimizer.step()
        return loss.item(), embedding

    def evaluate(self, inputs, target, idx):
        """评估模型在指定节点集上的准确率。"""
        if self.opt['cuda']:
            inputs = inputs.cuda()
            target = target.cuda()
            idx = idx.cuda()

        self.model.eval()

        logits = self.model(inputs)
        loss = self.criterion(logits[idx], target[idx])
        preds = torch.max(logits[idx], dim=1)[1]
        correct = preds.eq(target[idx]).double()
        accuracy = correct.sum() / idx.size(0)

        return loss.item(), preds, accuracy.item()

    def predict(self, inputs, tau=1):
        """输出软标签分布，可选择温度参数 tau。"""
        if self.opt['cuda']:
            inputs = inputs.cuda()

        self.model.eval()

        logits, embedding = self.model(inputs)
        logits = logits / tau

        logits = torch.softmax(logits, dim=-1).detach()

        return logits

    def save(self, filename):
        """保存模型与优化器状态。"""
        params = {
            'model': self.model.state_dict(),
            'optim': self.optimizer.state_dict()
        }
        try:
            torch.save(params, filename)
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

    def load(self, filename):
        """从磁盘加载模型参数。"""
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optim'])
