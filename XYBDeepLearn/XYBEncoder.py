import random

import torch as tr
import cv2
import os
import numpy as np
from XYBDeepLearn import XYBCommon


class XYBEncoderModel(tr.nn.Module):
    def __init__(self):
        super(XYBEncoderModel, self).__init__()
        self.conv01 = tr.nn.Conv2d(in_channels=1, out_channels=12, kernel_size=5, stride=3, device=XYBCommon.device_id)
        self.conv02 = tr.nn.Conv2d(in_channels=12, out_channels=16, kernel_size=5, stride=1, device=XYBCommon.device_id)
        self.conv03 = tr.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, device=XYBCommon.device_id)
        self.active01 = tr.nn.LeakyReLU()
        self.active02 = tr.nn.LeakyReLU()
        self.active03 = tr.nn.LeakyReLU()

        # Config ImageTexture
        data_test = tr.ones(1, 1, XYBCommon.ImageSize, XYBCommon.ImageSize, device=XYBCommon.device_id)
        data_test = self.conv01(data_test)
        data_test = self.conv02(data_test)
        data_test = self.conv03(data_test)
        print('-------------------EnCoder Shape')
        print(data_test.shape)
        data_test = tr.nn.Flatten()(data_test)
        print(data_test.shape)
        print('-------------------EnCoder Shape')

        conv_dim = data_test.shape[1]
        self.dense01 = tr.nn.Linear(128, 128, bias=True, device=XYBCommon.device_id)
        self.dense01_a = tr.nn.LeakyReLU()

    def forward(self, x):
        x = self.conv01(x)
        x = self.active01(x)
        x = self.conv02(x)
        x = self.active02(x)
        x = self.conv03(x)
        x = self.active03(x)
        x = tr.nn.Flatten()(x)

        x = self.dense01(x)
        x = self.dense01_a(x)
        return x

