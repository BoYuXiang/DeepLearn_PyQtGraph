# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


import cupy as cp
import torch as tr
import XYBDeepLearn as xl
import matplotlib.pyplot as plt
import pyqtgraph as pyqt
import sys

import MainWindow_Function
from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 800)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setBold(False)
        MainWindow.setFont(font)
        MainWindow.setStyleSheet("")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setSpacing(0)
        self.gridLayout.setObjectName("gridLayout")
        self.splitter = QtWidgets.QSplitter(self.centralwidget)
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setObjectName("splitter")
        self.W_Control = QtWidgets.QTabWidget(self.splitter)
        self.W_Control.setObjectName("W_Control")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.tab_3)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.W_InitData = QtWidgets.QPushButton(self.tab_3)
        self.W_InitData.setObjectName("W_InitData")
        self.verticalLayout.addWidget(self.W_InitData)
        self.W_Start = QtWidgets.QPushButton(self.tab_3)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setBold(False)
        self.W_Start.setFont(font)
        self.W_Start.setObjectName("W_Start")
        self.verticalLayout.addWidget(self.W_Start)
        self.W_Stop = QtWidgets.QPushButton(self.tab_3)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setBold(False)
        self.W_Stop.setFont(font)
        self.W_Stop.setObjectName("W_Stop")
        self.verticalLayout.addWidget(self.W_Stop)
        self.W_Reset = QtWidgets.QPushButton(self.tab_3)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setBold(False)
        self.W_Reset.setFont(font)
        self.W_Reset.setObjectName("W_Reset")
        self.verticalLayout.addWidget(self.W_Reset)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)
        self.gridLayout_2.addLayout(self.verticalLayout, 0, 0, 1, 1)
        self.W_Control.addTab(self.tab_3, "")
        self.W_Show = QtWidgets.QTabWidget(self.splitter)
        self.W_Show.setMinimumSize(QtCore.QSize(800, 0))
        self.W_Show.setObjectName("W_Show")
        self.gridLayout.addWidget(self.splitter, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1190, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.W_Show.setCurrentIndex(-1)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.W_Show.addTab(self.lr.widget, "Show")
        self.W_Start.clicked.connect(self.lr.start)
        self.W_Stop.clicked.connect(self.lr.stop)
        self.W_Reset.clicked.connect(self.lr.reset)

    lr = MainWindow_Function.MainWindowControl()
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.W_InitData.setText(_translate("MainWindow", "InitData"))
        self.W_Start.setText(_translate("MainWindow", "Start"))
        self.W_Stop.setText(_translate("MainWindow", "Stop"))
        self.W_Reset.setText(_translate("MainWindow", "Reset"))
        self.W_Control.setTabText(self.W_Control.indexOf(self.tab_3), _translate("MainWindow", "Tab 2"))
