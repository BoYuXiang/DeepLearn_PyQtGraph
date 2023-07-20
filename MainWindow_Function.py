import random

import cupy as cp
import torch as tr

from XYBDeepLearn import XYBDecoder, XYBEncoder, XYBCommon
import matplotlib.pyplot as plt
import pyqtgraph as pyqt
import sys
import numpy as np
import torch.optim
from torchsummary import summary
from torchvision import io, datasets
from PyQt5 import QtCore, QtGui, QtWidgets
from torch.utils.data import DataLoader

from PyQt5.QtCore import QTimer, QRect
import torchvision
import matplotlib.pyplot as plt

class MainWindowControl(object):
    def __init__(self):
        self.encoder_model = XYBEncoder.XYBEncoderModel()
        self.decoder_model = XYBDecoder.XYBDecoderModel()

        self.op = tr.optim.Adam([
            {'params': self.encoder_model.parameters()},
            {'params': self.decoder_model.parameters()}
        ], lr=0.01)

        self.widget = pyqt.GraphicsLayoutWidget()
        self.preview_window = self.widget.addPlot(tilte='Preview')
        self.preview_window.setAspectLocked(True)
        self.loss_window = self.widget.addPlot(tilte='Loss')
        self.loss_window.setAspectLocked(True)
        self.preview_window.showGrid(x=True, y=True)
        self.loss_window.showGrid(x=True, y=True)
        self.loss_window.setLabel('left', "Coss", units='0-1')
        self.loss_window.setLabel('bottom', "Iteration", units='times')

        self.scatter_point = pyqt.ScatterPlotItem(x=[0, 1], y=[0, 1], color=(255, 0, 0))
        self.preview_window.addItem(self.scatter_point)

        trans = torchvision.transforms.Compose([
            torchvision.transforms.Resize(28),  # image size int or tuple
            # Add more transforms here
            torchvision.transforms.ToTensor()  # convert to tensor at the end
        ])
        data_train = torchvision.datasets.MNIST(
            root='./Train/',
            download=False,
            transform=trans
        )
        self.data = torch.utils.data.DataLoader(
            dataset=data_train,
            batch_size=1600,
            shuffle=True
        )

        self.time = QTimer()

        self.preview_img = pyqt.ImageItem(image=np.ones((2, 2, 3)) * 255)
        self.original_img = pyqt.ImageItem(image=np.ones((2, 2, 3)) * 255)
        self.create_img = pyqt.ImageItem(image=np.ones((28, 28, 3)) * 255)
        # for i, batch in enumerate(self.data):
        #    print(i)
        #    print(batch[0].shape)
        #    show_data = torchvision.utils.make_grid(batch[0])
#
        #    print(show_data.shape)
        #    self.original_img.setImage(image=show_data.data.numpy().transpose(1, 2, 0))
        #    break

        self.preview_img.setRect(QRect(0, 0, 1, 1))
        self.original_img.setRect(QRect(1, 0, 1, 1))
        self.create_img.setRect(QRect(-1, -1, 1, 1))
        self.loss_window.addItem(self.preview_img)
        self.loss_window.addItem(self.original_img)
        self.loss_window.addItem(self.create_img)

        color_map = pyqt.ColorMap(None, color=[
            QtGui.QColor(255, 0, 0),
            QtGui.QColor(0, 255, 0),
            QtGui.QColor(0, 0, 255),
            QtGui.QColor(255, 0, 255),
            QtGui.QColor(0, 255, 255),
            QtGui.QColor(255, 255, 255),
            QtGui.QColor(255, 255, 0),
            QtGui.QColor(255, 180, 0),
            QtGui.QColor(180, 255, 0),
            QtGui.QColor(255, 0, 180),
        ])
        img_selected_color = pyqt.ImageItem(image=np.ones((1, 1)))
        img_selected_color.setRect(QRect(0, 0, 1, 1))
        self.preview_window.addColorBar(
            img_selected_color,  # 关联一张图片，用来选取颜色
            colorMap=color_map,     # 自定义颜色映射列表
            values=(0, 9),  # 默认左端到右端的值
            orientation='h',    # 是否横过来 w 横着的  h 竖着的
            label='ColorNumber'    # 标题
        )

    def clear_canvas(self):
        pass

    def init_deeplearn(self):
        pass

    def start(self):
        print('start')
        self.time.timeout.connect(self.update)
        self.time.start(24)

    def update(self):
        train_data, label = next(iter(self.data))
        train_data = train_data.to(XYBCommon.device_id)
        self.encoder_model.train()
        self.decoder_model.train()

        self.op.zero_grad()
        tr_p = self.encoder_model.forward(train_data)
        tr_res = self.decoder_model.forward(tr_p)

        loss = tr.sum((train_data - tr_res) ** 2)
        loss = loss / (tr_res.shape[2] * tr_res.shape[3])
        loss.backward(retain_graph=True)
        print(loss)
        print(tr_p.shape)
        self.op.step()

        x = tr_p[:, 0].data.cpu().numpy()
        y = tr_p[:, 1].data.cpu().numpy()
        c = []
        for i in range(0, tr_p.shape[0]):
            number = label[i]
            if number == 0:
                c.append([255, 0, 0])
            if number == 1:
                c.append([0, 0, 255])
            if number == 2:
                c.append([0, 255, 0])
            if number == 3:
                c.append([255, 255, 0])
            if number == 4:
                c.append([0, 255, 255])
            if number == 5:
                c.append([255, 255, 255])
            if number == 6:
                c.append([255, 0, 255])
            if number == 7:
                c.append([150, 0, 255])
            if number == 8:
                c.append([0, 150, 255])
            if number == 9:
                c.append([0, 0, 150])
        c = np.array(c)
        self.scatter_point.setData(x=x, y=y, brush=c)

        data_grid = torchvision.utils.make_grid(train_data.cpu(), nrow=40, padding=2)
        preview_grid = torchvision.utils.make_grid(tr_res.cpu(), nrow=40, padding=2)
        self.original_img.setImage(image=XYBCommon.img_rot270(data_grid.data.numpy().transpose(1, 2, 0)))
        self.preview_img.setImage(image=XYBCommon.img_rot270(preview_grid.data.numpy().transpose(1, 2, 0)))
        self.preview_img.setRect(QRect(0, 0, 1, 1))
        self.original_img.setRect(QRect(1, 0, 1, 1))
        return

    def update_paint(self):
        pass

    def stop(self):
        print('stop')
        self.time.stop()

    def reset(self):
        self.value_change(random.randint(0, 1000)/1000, random.randint(0, 1000)/1000)

    def value_change(self, x, y):
        self.encoder_model.eval()
        self.decoder_model.eval()

        tr_p = tr.tensor([[x, y]], dtype=tr.float32, device=XYBCommon.device_id).reshape(1, 2)
        tr_res = self.decoder_model(tr_p)

        preview_grid = torchvision.utils.make_grid(tr_res.cpu(), nrow=20, padding=2)
        self.create_img.setImage(image=XYBCommon.img_rot270(preview_grid.data.numpy().transpose(1, 2, 0)))
