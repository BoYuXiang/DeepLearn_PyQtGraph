import random

import cupy as cp
import torch as tr
import XYBDeepLearn as xl
import matplotlib.pyplot as plt
import pyqtgraph as pyqt
import sys
import numpy as np

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QTimer, QRect


class MainWindowControl(object):
    widget = pyqt.GraphicsLayoutWidget()
    plot_window = None
    plot_window02 = None
    time = QTimer()
    x = []
    y = []
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

    lr = xl.XYBDeepLearn()
    op = xl.XYBOptimizer()
    tr_res = []

    layer_01 = None
    layer_02 = None
    layer_03 = None
    layer_04 = None

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

    def init_deeplearn(self, item_show: pyqt.ScatterPlotItem):
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

        self.lr = xl.XYBDeepLearn()
        self.op = xl.XYBOptimizer()
        self.lr.deep_init()

        self.layer_01 = xl.XYBLayer((8, 2), (8, 1), active_type='tan_h')
        self.layer_02 = xl.XYBLayer((5, 8), (5, 1), active_type='relu')
        self.layer_03 = xl.XYBLayer((3, 5), (3, 1), active_type='sigmoid')
        self.layer_04 = xl.XYBLayer((1, 3), (1, 1), active_type='softmax')

        self.op.add_param(self.layer_01.w)
        self.op.add_param(self.layer_01.b)
        self.op.add_param(self.layer_02.w)
        self.op.add_param(self.layer_02.b)
        self.op.add_param(self.layer_03.w)
        self.op.add_param(self.layer_03.b)
        self.op.add_param(self.layer_04.w)
        self.op.add_param(self.layer_04.b)

        data_point_x = []
        data_point_y = []
        data_p = []
        for x in range(0, 64):
            for y in range(0, 64):
                p = [-1 + x * (2 / 64), -1 + y * (2 / 64)]
                data_point_x.append(p[0])
                data_point_y.append(p[1])
                data_p.append(p)
        self.tr_res = tr.tensor(data_p, dtype=float, device=self.lr.device()).transpose(0, 1)

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
        mini_batch = self.get_random_data(5)

        coss = 0
        for batch in mini_batch:
            res_x = []
            res_y = []
            for index in batch:
                res_x.append(self.data_p[index])
                res_y.append(self.data_y[index])
            tr_x = tr.tensor(res_x, dtype=float, device=self.lr.device()).transpose(0, 1)
            tr_y = tr.tensor(res_y, dtype=float, device=self.lr.device()).transpose(0, 1)

            a01 = self.layer_01.forward(tr_x)
            a02 = self.layer_02.forward(a01)
            a03 = self.layer_03.forward(a02)
            a04 = self.layer_04.forward(a03)
            loss = tr.sum(-tr_y * tr.log(a04))
            loss.backward(retain_graph=True)
            coss += (tr.sum(tr.abs(tr_y - a04))) / (a04.shape[0] * a04.shape[1])

            self.op.update_value(0.1, 0.01)

        a01 = self.layer_01.forward(self.tr_res)
        a02 = self.layer_02.forward(a01)
        a03 = self.layer_03.forward(a02)
        a04 = self.layer_04.forward(a03)
        self.op.clear_grad()

        self.loop_x.append(self.loop_count)
        self.loop_y.append(coss.data.cpu().numpy().flatten()[0])
        self.loop_item.setData(self.loop_x, self.loop_y)
        self.img_item.setImage(a04.data.cpu().numpy().flatten().reshape((64, 64, 1)))

        self.loop_count += 1

    def stop(self):
        print('stop')
        self.time.stop()

    def reset(self):
        self.init_deeplearn(self.plot_item)
