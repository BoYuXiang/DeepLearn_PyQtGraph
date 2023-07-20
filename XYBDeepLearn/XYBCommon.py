import random

import torch as tr
import cv2
import os
import numpy as np
import torchvision.io
import torchvision.transforms.functional as fn
ImageSize = 28
device_id = tr.device('cuda')


def img_rot270(img):
    return np.rot90(np.rot90(np.rot90(img)))


class XYBDataLoader:
    def __init__(self):
        self.data_list = []
        self.tag_list = []
        self.test_list = []
        self.pop_list_data = []
        self.pop_list_tag = []

    def load_img(self, path: str, tag, max_data_length: int = 500):
        file_list = os.listdir(path)
        i = 0
        for file in file_list:
            if i > max_data_length:
                break
            fullpath = path + file

            data = torchvision.io.read_image(fullpath, torchvision.io.ImageReadMode.RGB).to(device_id)
            data = fn.resize(data, size=[64, 64]) 
            if data is None:
                break

            self.data_list.append(data/255)
            self.tag_list.append(tr.tensor([tag], dtype=tr.float32, device=device_id, requires_grad=False))
            i += 1

    def load_test_img(self, path: str):
        file_list = os.listdir(path)
        i = 0
        for file in file_list:
            fullpath = path + file

            data = torchvision.io.read_image(fullpath, torchvision.io.ImageReadMode.RGB).to(device_id)
            if data is None:
                break

            self.test_list.append(data/255)
            i += 1

    def get_data_batch(self, mini_batch: int):
        batch_data_x = []
        batch_data_y = []
        for i in range(0, mini_batch):
            if len(self.data_list) < 1:
                self.data_list = self.pop_list_data
                self.tag_list = self.pop_list_tag
                self.pop_list_data = []
                self.pop_list_tag = []
                r_x = tr.cat(batch_data_x, dim=0)
                r_y = tr.cat(batch_data_y, dim=0)
                return r_x, r_y, True

            index = random.randint(0, len(self.data_list)-1)

            batch_data_x.append(self.data_list[index])
            batch_data_y.append(self.tag_list[index])
            self.pop_list_data.append(self.data_list.pop(index))
            self.pop_list_tag.append(self.tag_list.pop(index))

        r_x = tr.cat(batch_data_x, dim=0)
        r_y = tr.cat(batch_data_y, dim=0)
        return r_x, r_y, False
