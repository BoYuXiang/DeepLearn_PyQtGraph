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
