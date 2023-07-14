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


def normal_distribution(x):
    u = 0.5
    o = 0.2
    y_data = np.exp((x - u) ** 2 / (2 * (o ** 2)) * (-1))
    v = 1 / np.sqrt(2 * np.pi * o)
    return v * y_data


class MainWindowControl(object):

    def __init__(self):
        self.model = XYBDeepLearn.XYBDeepModel()
        self.op = tr.optim.Adam(self.model.parameters(), lr=0.001)
        self.data = XYBDeepLearn.XYBDataLoader(self.model.device_id)

        self.widget = pyqt.GraphicsLayoutWidget()
        self.plot_window = self.widget.addPlot(tilte='Preview')
        self.plot_window.setAspectLocked(True)
        self.loss_window = self.widget.addPlot(tilte='Loss')
        self.loss_window.setAspectLocked(True)
        self.plot_window.showGrid(x=True, y=True)
        self.loss_window.showGrid(x=True, y=True)
        self.loss_window.setLabel('left', "Coss", units='0-1')
        self.loss_window.setLabel('bottom', "Iteration", units='times')

        self.x = [0, 1]
        self.y = [0, 1]
        self.plot_item = pyqt.ScatterPlotItem(
            size=10,
            hoverable=True,
            hoverSize=5,
            hoverBrush=(255, 255, 255)
        )
        self.plot_window.addItem(self.plot_item)

        self.loop_item = self.loss_window.plot(
            pen=(0, 255, 0),
            filllevel=0.0,
            brush=(50, 50, 200, 200),
            symbol='o',
            symbolSize=3)
        self.time = QTimer()
        self.time.timeout.connect(self.update)

        self.time_paint = QTimer()
        self.time_paint.timeout.connect(self.update_paint)

        self.loop_count = 0
        self.coss = 0
        self.loop_x = []
        self.loop_y = []
        self.test_text = []

        # Normal Distribution
        normal_data = np.random.random((XYBDeepLearn.ImageSize, XYBDeepLearn.ImageSize))
        self.paint_img = pyqt.ImageItem(image= normal_data * 255)
        self.paint_img.setRect(0, -6, 5, 5)
        self.plot_window.addItem(self.paint_img)

        # Normal Distribution Bar Show
        number_bar = 30
        step_one = 1 / number_bar

        bar_height = []
        bar_x = []
        for i in range(0, number_bar):
            bar_height.append(0)
            bar_x.append(i*step_one + step_one*0.5)

        for x in range(0, XYBDeepLearn.ImageSize):
            for y in range(0, XYBDeepLearn.ImageSize):
                p = normal_data[x][y]
                index = int(p / step_one)
                bar_height[index] += 1

        bar_height = np.array(bar_height)
        aver = np.max(bar_height)
        bar_height = bar_height / aver
        self.plot_bar = pyqt.BarGraphItem(x=1, height=1, width=1, brush=(0, 255, 0, 255))
        self.plot_bar.setOpts(x=bar_x, height=bar_height, width=step_one)
        self.loss_window.addItem(self.plot_bar)

        g_x = []
        g_y = []
        for i in range(0, 100):
            x = i / 100
            y = normal_distribution(x)
            g_x.append(x)
            g_y.append(y)
        self.loss_window.plot(x=g_x, y=g_y, symbol='o', symbolSize=4, color=(255, 0, 0))

    def clear_canvas(self):
        normal_data = np.random.random((XYBDeepLearn.ImageSize, XYBDeepLearn.ImageSize))
        self.paint_img.setImage(normal_data)
        # Normal Distribution Bar Show
        number_bar = 30
        step_one = 1 / number_bar

        bar_height = []
        bar_x = []
        for i in range(0, number_bar):
            bar_height.append(0)
            bar_x.append(i*step_one + step_one*0.5)

        for x in range(0, XYBDeepLearn.ImageSize):
            for y in range(0, XYBDeepLearn.ImageSize):
                p = normal_data[x][y]
                index = int(p / step_one)
                bar_height[index] += 1

        bar_height = np.array(bar_height)
        aver = np.max(bar_height)
        bar_height = bar_height / aver

        self.plot_bar.setOpts(x=bar_x, height=bar_height, width=step_one)

    def init_deeplearn(self):
        self.loop_x = []
        self.loop_y = []
        self.loop_count = 0
        self.test_text = []
        self.coss = 0

    def start(self):
        print('start')
        self.time.start(24)
        self.time_paint.start(60)

    def update(self):
        data_batch = self.data.get_data_batch(300)

        self.model.train()
        self.op.zero_grad()

        x = data_batch[0]
        y = data_batch[1]

        prob_y = self.model.forward(x)
        loss = tr.sum(-y * tr.log(prob_y)) / 300

        loss.backward(retain_graph=True)
        self.coss += loss
        self.op.step()

        if data_batch[2] is True:
            self.model.eval()

            for i in range(0, len(self.data.test_list)):
                test_x = self.data.test_list[i].to(self.model.device_id)
                test_y = self.model.forward(test_x)
                test_y = tr.argmax(test_y).data.cpu().numpy().flatten()[0]
                self.test_text[i].setText('[%0.1f]' % test_y)

            self.loop_x.append(self.loop_count)
            self.loop_y.append(self.coss.data.cpu().numpy().flatten()[0])
            self.loop_item.setData(self.loop_x, self.loop_y)

            self.loop_count += 1
            self.coss = 0

    def update_paint(self):
        test_x = tr.tensor(self.paint_img.image, dtype=tr.float32, device=self.model.device_id)
        test_x = test_x.reshape(1, 1, XYBDeepLearn.ImageSize, XYBDeepLearn.ImageSize)
        test_y = self.model.forward(test_x)
        test_y = tr.argmax(test_y).data.cpu().numpy().flatten()[0]
        self.paint_img_text.setText('[%0.1f]' % test_y)

    def stop(self):
        print('stop')
        self.time.stop()

        test_x = tr.tensor(self.paint_img.image, dtype=tr.float32, device=self.model.device_id)
        test_x = test_x.reshape(1, 1, XYBDeepLearn.ImageSize, XYBDeepLearn.ImageSize)
        test_y = self.model.forward(test_x)
        test_y = tr.argmax(test_y).data.cpu().numpy().flatten()[0]
        self.paint_img_text.setText('[%0.1f]' % test_y)

    def reset(self):
        # normal_data = np.random.random((XYBDeepLearn.ImageSize, XYBDeepLearn.ImageSize))
        normal_data = np.random.normal(0.5, 0.2, (XYBDeepLearn.ImageSize, XYBDeepLearn.ImageSize))
        # Normal Distribution Bar Show
        number_bar = 30
        step_one = 1 / number_bar

        bar_height = []
        bar_x = []
        for i in range(0, number_bar):
            bar_height.append(0)
            bar_x.append(i * step_one + step_one * 0.5)

        for x in range(0, XYBDeepLearn.ImageSize):
            for y in range(0, XYBDeepLearn.ImageSize):
                p = normal_data[x][y]
                index = int(p / step_one)
                if index < len(bar_height):
                    bar_height[index] += 1
        self.paint_img.setImage(normal_data * 255)

        bar_height = np.array(bar_height)
        aver = np.max(bar_height)
        bar_height = bar_height / aver

        self.plot_bar.setOpts(x=bar_x, height=bar_height, width=step_one)
