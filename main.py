import random

import cupy as cp
import torch as tr
import XYBDeepLearn as xl
import matplotlib.pyplot as plt
import pyqtgraph as pyqt
import pyqtgraph.examples
import sys
import numpy as np
import XYBDeepLearn

from PyQt5 import QtCore, QtGui, QtWidgets
import QtWindow.MainWindow

'''
model = XYBDeepLearn.XYBDeepModel()
data = tr.ones(1, 3, 6, 6, device=model.device_id)
print(model.conv01.weight)
res = model.forward(data)
print(res.shape)
print(res)
'''

app = QtWidgets.QApplication(sys.argv)
MainWindow = QtWidgets.QMainWindow()
Ui = QtWindow.MainWindow.Ui_MainWindow()
Ui.setupUi(MainWindow)
pyqtgraph.examples.run()
MainWindow.show()
sys.exit(app.exec_())

'''
fig = plt.figure()

plt.subplot(111)

x = []
y = []

print(np.random.random((4, 4)))

u = 0
o = 1
for i in range(0, 1000):
    x_data = -5 + (10 / 1000)*i
    y_data = np.exp((x_data - u)**2 / (2 * (o ** 2)) * (-1))
    v = 1 / np.sqrt(2 * np.pi * o)

    x.append(x_data)
    y.append(v * y_data)


plt.plot(x, y, color=(1, 0, 1))


plt.show()
'''
