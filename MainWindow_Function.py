import random

import cupy as cp
import torch as tr

import XYBDeepLearn
import XYBDeepLearn as xl
import matplotlib.pyplot as plt
import pyqtgraph as pyqt
import sys
import numpy as np
import torch.optim
from torchsummary import summary

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QTimer, QRect


class MainWindowControl(object):
    widget = pyqt.GraphicsLayoutWidget()
    plot_window = None
    plot_window02 = None
    time = QTimer()
    plot_item = pyqt.PlotDataItem()
    img_item = pyqt.ImageItem()

    def __init__(self):
        self.plot_window = self.widget.addPlot(tilte='Show')
        self.plot_window02 = self.widget.addPlot(tilte='S')
        self.plot_window.showGrid(x=True, y=True)
        self.plot_window02.showGrid(x=True, y=True)
        self.plot_window02.setLabel('left', "Coss", units='0-1')
        self.plot_window02.setLabel('bottom', "Iteration", units='times')

        self.img_item = pyqt.ImageItem(image=np.zeros((64, 64, 1)))
        self.img_item.setRect(QRect(-1, -1, 2, 2))

        self.x = [0, 1]
        self.y = [0, 1]
        self.plot_item = pyqt.ScatterPlotItem(
            size=10,
            hoverable=True,
            hoverSize=5,
            hoverBrush=(255, 255, 255)
        )

        self.plot_window.addItem(self.img_item)
        self.plot_window.addItem(self.plot_item)

        self.loop_item = self.plot_window02.plot(
            pen=(0, 255, 0),
            filllevel=0.0,
            brush=(50, 50, 200, 200),
            symbol='o',
            symbolSize=3)
        self.time = QTimer()
        self.time.timeout.connect(self.update)

        self.widget.nextRow()
        self.plot_window03 = self.widget.addPlot(tilte='Gaussian Distribution')

    data_length = 20

    def get_random_data(self, mini_batch: int):
        r = np.random.permutation(range(0, self.data_length))
        res = []

        while len(r) > 0:
            r_one = []
            for i in range(0, mini_batch):
                if len(r) == 0:
                    break
                r_one.append(r[0])
                r = np.delete(r, [0, 0])
            res.append(r_one)

        return res

    data_p = []
    data_y = []
    tr_res = []
    model = XYBDeepLearn.XYBDeepModel()
    op = tr.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    def init_deeplearn(self):
        self.op = tr.optim.SGD(self.model.parameters(), lr=0.1, momentum=0.4)
        draw_x = []
        draw_y = []
        draw_color = []
        self.data_p = []
        self.data_y = []

        for i in range(0, 10):
            p = [random.random()*2-1, random.random()*2-1]
            self.data_p.append(p)
            self.data_y.append([0])

            draw_x.append(p[0])
            draw_y.append(p[1])
            draw_color.append((255, 0, 0, 255))

        for i in range(0, 10):
            p = [random.random()*2-1, random.random()*2-1]
            self.data_p.append(p)
            self.data_y.append([1])

            draw_x.append(p[0])
            draw_y.append(p[1])
            draw_color.append((0, 255, 0, 255))

        self.plot_item.setData(x=draw_x, y=draw_y, brush=draw_color)

        data_point_x = []
        data_point_y = []
        data_p = []
        for x in range(0, 64):
            for y in range(0, 64):
                p = [-1 + x * (2 / 64), -1 + y * (2 / 64)]
                data_point_x.append(p[0])
                data_point_y.append(p[1])
                data_p.append(p)
        self.tr_res = tr.tensor(data_p, dtype=tr.float32)
        self.loop_x = []
        self.loop_y = []
        self.loop_count = 0

    def start(self):
        print('start')
        self.time.start(24)

    loop_count = 0
    loop_item = pyqt.PlotDataItem()
    loop_x = []
    loop_y = []

    def update(self):
        mini_batch = self.get_random_data(4)
        coss = 0

        self.model.train()
        for batch in mini_batch:
            self.op.zero_grad()

            res_x = []
            res_y = []
            for index in batch:
                res_x.append(self.data_p[index])
                res_y.append(self.data_y[index])
            tr_x = tr.tensor(res_x, device=self.model.device_id)
            tr_y = tr.tensor(res_y, device=self.model.device_id)
            # summary(self.model, (len(batch), 2))
            prob_y = self.model.forward(tr_x)
            loss = tr.sum(-tr_y * tr.log(prob_y))
            loss.backward(retain_graph=True)
            self.op.step()
            coss += (tr.sum(tr.abs(tr_y - prob_y))) / (prob_y.shape[0] * prob_y.shape[1])

        self.loop_x.append(self.loop_count)
        self.loop_y.append(coss.data.cpu().numpy().flatten()[0])
        self.loop_item.setData(self.loop_x, self.loop_y)

        self.model.eval()
        res_data = self.model.forward(self.tr_res)
        img_data = res_data.data.cpu().numpy().flatten().reshape((64, 64, 1))
        self.img_item.setImage(img_data)

        self.loop_count += 1

    def stop(self):
        print('stop')
        self.time.stop()

    def reset(self):
        self.init_deeplearn()
