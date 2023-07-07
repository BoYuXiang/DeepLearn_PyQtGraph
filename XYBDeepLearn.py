import random

import torch as tr

device_id = 0


class XYBDataReader:
    pass


class XYBOptimizer:
    param = []

    def add_param(self, param):
        self.param.append(param)

    def update_value(self, learn: float, mountain: float):
        for p in self.param:
            slop = p.grad.data
            p.data = mountain * p.data + (1 - mountain) * (p.data - (slop * learn))
            print(p.grad.data)
            p.grad.data.zero_()

    def clear_grad(self):
        for p in self.param:
            p.grad.data.zero_()


class XYBDeepLearn:
    def deep_init(self):
        self
        device_id = tr.device('cuda')

    def device(self):
        self
        return device_id


class XYBLayer:
    w = tr.tensor([0], dtype=float, device=device_id)
    b = tr.tensor([0], dtype=float, device=device_id)
    active_type = 'sigmoid'
    a = None

    def __init__(self, w_dim: tuple, b_dim: tuple, active_type: str):
        self.w = tr.randn(w_dim, dtype=float, device=device_id)
        self.b = tr.randn(b_dim, dtype=float, device=device_id)
        self.active_type = active_type
        self.w.requires_grad = True
        self.b.requires_grad = True

    def forward(self, x: tr.Tensor):
        '''
        if self.a is not None:
            if random.random() < 0.1:
                print('forget')
                return self.a
        '''
        z = tr.matmul(self.w, x) + self.b

        if self.active_type == 'sigmoid':
            self.a = 1 / (1 + tr.exp(z * (-1)))
        if self.active_type == 'relu':
            self.a = tr.relu(z)
        if self.active_type == 'tan_h':
            pex = tr.exp(z)
            nex = tr.exp(z * (-1))
            self.a = (pex - nex)/(pex + nex)
        if self.active_type == 'softmax':
            soft_e = tr.exp(z)
            sum_e = tr.sum(soft_e)
            self.a = soft_e / sum_e
        return self.a

