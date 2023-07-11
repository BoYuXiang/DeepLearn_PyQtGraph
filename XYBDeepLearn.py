import random

import torch as tr


class XYBDeepModel(tr.nn.Module):

    def __init__(self):
        super(XYBDeepModel, self).__init__()
        self.device_id = tr.device('cuda')

        self.dense01 = tr.nn.Linear(2, 6, bias=True, device=self.device_id)
        self.dense02 = tr.nn.Linear(6, 4, bias=True, device=self.device_id)
        self.dense03 = tr.nn.Linear(4, 2, bias=True, device=self.device_id)
        self.dense04 = tr.nn.Linear(2, 1, bias=True, device=self.device_id)

        self.active01 = tr.nn.Tanh()
        self.active02 = tr.nn.Tanh()
        self.active03 = tr.nn.Sigmoid()
        self.output_layer = tr.nn.Softmax(dim=0)

    def forward(self, x):
        x = x.to(self.device_id)
        x = self.dense01(x)
        x = self.active01(x)
        x = self.dense02(x)
        x = self.active02(x)
        x = self.dense03(x)
        x = self.active03(x)
        x = self.dense04(x)
        x = self.output_layer(x)
        return x
