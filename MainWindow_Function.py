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

        self.paint_img_text = pyqt.TextItem('Real Text Paint', color=(255, 0, 0))
        self.paint_img_text.setPos(6, -6)
        self.paint_img = pyqt.ImageItem(image=np.ones((XYBDeepLearn.ImageSize, XYBDeepLearn.ImageSize)) * 0)
        self.paint_img.setRect(0, -6, 5, 5)
        self.plot_window.addItem(self.paint_img)
        self.plot_window.addItem(self.paint_img_text)

        kern = np.ones((3, 3))
        self.paint_img.setDrawKernel(kern, mask=kern, center=(1, 1), mode='set')
        self.paint_img.setLevels([0, 0])

    def clear_canvas(self):
        self.paint_img.setImage(image=np.ones((XYBDeepLearn.ImageSize, XYBDeepLearn.ImageSize)) * 0)
        kern = np.ones((3, 3))
        self.paint_img.setDrawKernel(kern, mask=kern, center=(1, 1), mode='set')
        self.paint_img.setLevels([0, 0])

    def init_deeplearn(self):
        self.loop_x = []
        self.loop_y = []
        self.loop_count = 0
        self.test_text = []
        self.coss = 0

        self.data.load_img('X:/PyDeepLearn/Train/9/', [0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        self.data.load_img('X:/PyDeepLearn/Train/8/', [0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
        self.data.load_img('X:/PyDeepLearn/Train/7/', [0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
        self.data.load_img('X:/PyDeepLearn/Train/6/', [0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
        self.data.load_img('X:/PyDeepLearn/Train/5/', [0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
        self.data.load_img('X:/PyDeepLearn/Train/4/', [0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        self.data.load_img('X:/PyDeepLearn/Train/3/', [0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        self.data.load_img('X:/PyDeepLearn/Train/2/', [0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        self.data.load_img('X:/PyDeepLearn/Train/1/', [0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        self.data.load_img('X:/PyDeepLearn/Train/0/', [1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        self.data.load_test_img('X:/PyDeepLearn/Test/')
        # Image Test DataReadLoader Show
        x = 0
        for img in self.data.test_list:
            img_item = pyqt.ImageItem(
                image=img.reshape(XYBDeepLearn.ImageSize, XYBDeepLearn.ImageSize).data.numpy()
            )
            img_item.setRect(QRect(x, 0, 1, 1))
            self.plot_window.addItem(img_item)

            text_item = pyqt.TextItem('Test', color=(0, 255, 0))
            text_item.setParentItem(img_item)
            text_item.setPos(x, 0)
            self.test_text.append(text_item)
            self.plot_window.addItem(text_item)
            x += 1

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
        self.time_paint.stop()

        test_x = tr.tensor(self.paint_img.image, dtype=tr.float32, device=self.model.device_id)
        test_x = test_x.reshape(1, 1, XYBDeepLearn.ImageSize, XYBDeepLearn.ImageSize)
        test_y = self.model.forward(test_x)
        test_y = tr.argmax(test_y).data.cpu().numpy().flatten()[0]
        self.paint_img_text.setText('[%0.1f]' % test_y)

    def reset(self):
        self.init_deeplearn()
