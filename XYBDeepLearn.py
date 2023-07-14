import random

import torch as tr
import cv2
import os
import numpy as np

ImageSize = 64


def img_rot270(img):
    return np.rot90(np.rot90(np.rot90(img)))


class XYBDataLoader:
    def __init__(self,  device_id: int):
        self.data_list = []
        self.tag_list = []
        self.test_list = []
        self.pop_list_data = []
        self.pop_list_tag = []
        self.device_id = device_id

    def load_img(self, path: str, tag, max_data_length: int = 2000):
        file_list = os.listdir(path)
        i = 0
        for file in file_list:
            if i > max_data_length:
                break
            fullpath = path + file

            data = img_rot270(cv2.imread(fullpath, 0))
            data = ~cv2.resize(data, (ImageSize, ImageSize))
            data = tr.tensor(data, dtype=tr.float32, device=self.device_id).reshape(1, 1, ImageSize, ImageSize)

            self.data_list.append(data/255)
            self.tag_list.append(tr.tensor([tag], dtype=tr.float32, device=self.device_id))
            i += 1

    def load_test_img(self, path: str):
        file_list = os.listdir(path)
        i = 0
        for file in file_list:
            fullpath = path + file

            data = img_rot270(cv2.imread(fullpath, 0))
            data = ~cv2.resize(data, (ImageSize, ImageSize))
            data = tr.tensor(data, dtype=tr.float32).reshape(1, 1, ImageSize, ImageSize)

            self.test_list.append(data/255)
            i += 1

    def get_data_batch(self, mini_batch: int):

        batch_data_x = []
        batch_data_y = []
        for i in range(0, mini_batch):
            index = random.randint(0, len(self.data_list)-1)

            if len(self.data_list) < 2:
                self.data_list = self.pop_list_data
                self.tag_list = self.pop_list_tag
                self.pop_list_data = []
                self.pop_list_tag = []
                r_x = tr.cat(batch_data_x, dim=0)
                r_y = tr.cat(batch_data_y, dim=0)
                return r_x, r_y, True

            batch_data_x.append(self.data_list[index])
            batch_data_y.append(self.tag_list[index])
            self.pop_list_data.append(self.data_list.pop(index))
            self.pop_list_tag.append(self.tag_list.pop(index))

        r_x = tr.cat(batch_data_x, dim=0)
        r_y = tr.cat(batch_data_y, dim=0)
        return r_x, r_y, False


class XYBDeepModel(tr.nn.Module):

    def __init__(self):
        super(XYBDeepModel, self).__init__()
        self.device_id = tr.device('cuda')
        self.conv01 = tr.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=8, device=self.device_id)
        self.pool01 = tr.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv02 = tr.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=4, device=self.device_id)
        self.pool02 = tr.nn.MaxPool2d(kernel_size=2, stride=2)

        # Config ImageTexture
        data_test = tr.ones(5, 1, ImageSize, ImageSize, device=self.device_id)
        data_test = self.conv01(data_test)
        data_test = self.pool01(data_test)
        data_test = self.conv02(data_test)
        data_test = self.pool02(data_test)

        print('------------Data Shape')
        print(data_test.shape)
        data_test = tr.nn.Flatten()(data_test)
        print(data_test.shape)
        print('------------Data Shape')

        self.dense01 = tr.nn.Linear(data_test.shape[1], 128, bias=True, device=self.device_id)
        self.active01 = tr.nn.Tanh()
        self.dense02 = tr.nn.Linear(128, 32, bias=True, device=self.device_id)
        self.active02 = tr.nn.Tanh()
        self.dense03 = tr.nn.Linear(32, 10, bias=True, device=self.device_id)

        self.output_layer = tr.nn.Softmax(dim=1)

        print('------------Result Shape')
        data_test = self.dense01(data_test)
        data_test = self.active01(data_test)
        data_test = self.dense02(data_test)
        data_test = self.active02(data_test)
        data_test = self.dense03(data_test)
        data_test = self.output_layer(data_test)
        print(data_test.shape)
        print('------------Result Shape')

    def forward(self, x):
        x = self.conv01(x)
        x = self.pool01(x)
        x = self.conv02(x)
        x = self.pool02(x)
        x = tr.nn.Flatten()(x)

        x = self.dense01(x)
        x = self.active01(x)
        x = self.dense02(x)
        x = self.active02(x)
        x = self.dense03(x)
        x = self.output_layer(x)
        return x
