import random

import torch as tr
import cv2
import os
import numpy as np
from XYBDeepLearn import XYBCommon


class XYBDecoderModel(tr.nn.Module):
    def __init__(self):
        super(XYBDecoderModel, self).__init__()

        self.dense01 = tr.nn.Linear(128, 128, bias=True, device=XYBCommon.device_id)
        self.dense01_a = tr.nn.LeakyReLU()
        self.dense02_a = tr.nn.LeakyReLU()
        self.dense03_a = tr.nn.LeakyReLU()

        self.unflatten = tr.nn.Unflatten(1, (32, 2, 2))

        self.conv01 = tr.nn.ConvTranspose2d(in_channels=32, out_channels=28, kernel_size=3, stride=1, device=XYBCommon.device_id)
        self.conv02 = tr.nn.ConvTranspose2d(in_channels=28, out_channels=16, kernel_size=5, stride=1, device=XYBCommon.device_id)
        self.conv03 = tr.nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=7, stride=3, device=XYBCommon.device_id)
        self.active01 = tr.nn.LeakyReLU()
        self.active02 = tr.nn.LeakyReLU()
        self.active03 = tr.nn.LeakyReLU()
        self.active04 = tr.nn.LeakyReLU()

        # Config ImageTexture
        data_test = tr.ones(1, 32, 2, 2, device=XYBCommon.device_id)
        data_test = self.conv01(data_test)
        data_test = self.conv02(data_test)
        data_test = self.conv03(data_test)

        print('-------------------DeCoder Shape')
        print(data_test.shape)
        print('-------------------DeCoder Shape')

    def forward(self, x):
        x = self.dense01(x)
        x = self.dense01_a(x)

        x = self.unflatten(x)

        x = self.active01(x)
        x = self.conv01(x)
        x = self.active02(x)
        x = self.conv02(x)
        x = self.active03(x)
        x = self.conv03(x)
        x = self.active04(x)
        return x

