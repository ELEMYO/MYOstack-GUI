# Graphical interface for signal visualization and interaction with ELEMYO MYOstack sensors
# 2024-11-18 by ELEMYO ( https://github.com/ELEMYO/MYOSTACK-GUI )
# 
# Changelog:
#     2024-11-18 - improved user interface, recording to *bin file added
#     2022-05-13 - improved user interface
#     2021-04-30 - envelope plot added
#     2021-04-23 - initial release

# Code is placed under the MIT license
# Copyright (c) 2021 ELEMYO
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# ===============================================

import sys
import os
from importlib import metadata

missing = {'pyserial', 'pyqtgraph', 'PyQt5', 'numpy', 'scipy'} 
for dist in metadata.distributions():
    if dist.name in missing:
        missing.remove(dist.name)

if missing:
    for module in missing:
        os.system("python -m pip install " + module)

from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import Qt
import serial
import pyqtgraph as pg
import numpy as np
import time
from scipy.signal import butter, lfilter
import serial.tools.list_ports
from scipy.fftpack import fft
from serial import SerialException
from datetime import datetime
import struct

# Main window
class GUI(QtWidgets.QMainWindow):
    # Initialize constructor
    def __init__(self):
          super(GUI, self).__init__()
          self.initUI()
    # Custom constructor 
    def initUI(self): 
        # Values
        self.delay = 0.2 # Graphics update delay
        self.setWindowTitle("ELEMYO MYOstack GUI v1.2.0")
        self.setWindowIcon(QtGui.QIcon('img/icon.png'))
        
        self.fs = 1001 # Sampling frequency in Hz
        self.dt = 1/self.fs  # Time between two signal sempling in s
        
        self.passLowFrec = 10 # Low frequency for band-pass filter
        self.passHighFrec = 500 # High frequency for band-pass filter
        
        self.timeWidth = 10 # Plot window length in seconds
        self.dataWidth = int((self.timeWidth+0.5)*self.fs) # Maximum count of plotting data points (10.5 seconds window)
        self.Data = np.zeros((9, self.dataWidth)) # Raw data array, first index - sensor number, second index - sensor data
        self.DataEnvelope = np.zeros((9, self.dataWidth)) # Envelope of row data, first index - sensor number, second index - sensor data
        self.l = 0 # Current sensor data point
        self.Time = [0]*self.dataWidth # Time array (in seconds)
        
        self.MovingAverage = MovingAverage(self.fs) # Variable for data envelope (for moving average method)
        
        self.recordingFileName_BIN = '' # Recording file name
        self.recordingFileName_TXT = '' # Recording file name
        self.recordingFile_BIN = 0 # Recording file 
        self.recordingFile_TXT = 0 # Recording file
        self.loadFileName = '' # Data load file name
        self.loadFile = 0 # Data load variable
        self.sliderpos = 0 # Position of data slider 
        self.loadDataLen = 0 # Number of signal samples in data file
        self.loadData = 0 # Data from load file
        
        self.FFT = np.zeros((9, 2000)) # Fast Fourier transform data
        
        # Accessory variables for data read from serial
        self.ms_len = 0;
        self.msg_end = 0
        
        # Menu panel
        self.COMports = QtWidgets.QComboBox()        
        self.COMports.setDisabled(False)
        self.COMports.setSizeAdjustPolicy(0)
        
        self.liveFromSerialAction = QtWidgets.QAction(QtGui.QIcon('img/play.png'), 'Start/Stop live from serial ', self)
        self.liveFromSerialAction.setCheckable(True)
        self.liveFromSerialAction.setChecked(False)
        self.liveFromSerialAction.triggered.connect(self.liveFromSerial)

        self.dataRecordingAction = QtWidgets.QAction(QtGui.QIcon('img/rec.png'), 'Start/Stop recording', self)
        self.dataRecordingAction.triggered.connect(self.dataRecording)
        self.dataRecordingAction.setCheckable(True)
        self.dataRecordingAction.setDisabled(True)
        
        self.refreshAction = QtWidgets.QAction(QtGui.QIcon('img/refresh.png'), 'Refresh screen (R)', self)
        self.refreshAction.setShortcut('r')
        self.refreshAction.triggered.connect(self.refreshForAction)
        self.refreshAction.setDisabled(True)   
        
        self.pauseAction = QtWidgets.QAction(QtGui.QIcon('img/pause.png'), 'Pause (Space)', self)
        self.pauseAction.setCheckable(True)
        self.pauseAction.setChecked(False)
        self.pauseAction.triggered.connect(self.pause)
        self.pauseAction.setShortcut('Space')
        self.pauseAction.setDisabled(True)   
               
        dataLoadAction = QtWidgets.QAction(QtGui.QIcon('img/load.png'), 'Select playback file', self)
        dataLoadAction.triggered.connect(self.dataLoad)
        
        self.PlaybackAction = QtWidgets.QAction(QtGui.QIcon('img/playback.png'), 'Start/Stop playback from file: \nFILE NOT SELECTED', self)
        self.PlaybackAction.triggered.connect(self.Playback)
        self.PlaybackAction.setCheckable(True)
        self.PlaybackAction.setDisabled(True)
        
        self.slider = QtWidgets.QScrollBar(QtCore.Qt.Horizontal)
        self.slider.setValue(0)
        self.slider.setFixedWidth(40)
        self.slider.setDisabled(True)  
        
        self.sensorsNumberAction = QtWidgets.QLabel(' SENSORS NUMBER: ', self)
        self.sensorsNumberAction1 = QtWidgets.QLabel('     ', self)
        self.sensorsNumber = QtWidgets.QDoubleSpinBox()
        self.sensorsNumber.setRange(1, 9)
        self.sensorsNumber.setDecimals(0)
        self.sensorsNumber.setDisabled(True)
        self.sensorsNumber.setValue(9) 
        
        self.MYOstackVersion = QtWidgets.QLabel('       MYOstack version:  ', self)
        self.MYOstackVersionCheck = QtWidgets.QComboBox() 
        self.MYOstackVersionCheck.addItem("v2.0")
        self.MYOstackVersionCheck.addItem("v1.1")
        self.MYOstackVersionCheck.addItem("v1.0")
        self.MYOstackVersion1 = QtWidgets.QLabel('       ', self)
        
        self.rawSignalAction = QtWidgets.QCheckBox('MAIN SIGNAL', self)
        self.rawSignalAction.setChecked(True)
        self.rawSignalAction1 = QtWidgets.QLabel('       ', self)  

        self.EnvelopeSignalAction = QtWidgets.QCheckBox('ENVELOPE:', self)
        self.EnvelopeSignalAction.setChecked(True)
        self.EnvelopeSignalAction1 = QtWidgets.QLabel('    ', self)
        self.envelopeSmoothingСoefficient = QtWidgets.QDoubleSpinBox()
        self.envelopeSmoothingСoefficient.setSingleStep(0.01)
        self.envelopeSmoothingСoefficient.setRange(0, 1)
        self.envelopeSmoothingСoefficient.setValue(0.95)
        
        self.bandstopAction = QtWidgets.QCheckBox('BANDSTOP FILTER:', self)
        self.bandstopAction.setCheckable(True)
        
        self.notchActiontypeBox=QtWidgets.QComboBox()
        self.notchActiontypeBox.addItem("50 Hz")
        self.notchActiontypeBox.addItem("60 Hz")
        self.notchActiontypeBox.setDisabled(True)
                        
        self.bandpassAction = QtWidgets.QCheckBox('BANDPASS FILTER:', self)
        self.bandpassAction.setCheckable(True)
        self.bandpassAction.setChecked(False)
        self.bandpassAction1 = QtWidgets.QLabel('  -  ', self)
        self.bandpassAction2 = QtWidgets.QLabel('       ', self)
        
        self.passLowFreq = QtWidgets.QSpinBox()
        self.passLowFreq.setRange(10, 500)
        self.passLowFreq.setValue(10)
        self.passLowFreq.setDisabled(True)
                      
        self.passHighFreq = QtWidgets.QSpinBox()
        self.passHighFreq.setRange(10, 500)
        self.passHighFreq.setValue(500)
        self.passHighFreq.setDisabled(True)     
        
        self.sensorSelectedAction = QtWidgets.QLabel('Sensor: ', self)
        
        self.sensorSelectedActionBox=QtWidgets.QComboBox()
        self.sensorSelectedActionBox.addItem("1")

#--------------------------        
        # Toolbar
        toolbar = []
        toolbar.append(self.addToolBar('Tool1'))
        toolbar.append(self.addToolBar('Tool2'))
        toolbar.append(self.addToolBar('Tool3'))
        toolbar[0].addWidget(self.COMports)
        toolbar[0].addAction(self.liveFromSerialAction)
        toolbar[0].addAction(self.dataRecordingAction)
        toolbar[0].addAction(self.refreshAction)
        toolbar[0].addAction(self.pauseAction)
        toolbar[1].addAction(dataLoadAction)
        toolbar[1].addAction(self.PlaybackAction)
        toolbar[1].addWidget(self.slider)
        toolbar[2].addWidget(self.sensorsNumberAction)
        toolbar[2].addWidget(self.sensorsNumber)
        toolbar[2].addWidget(self.MYOstackVersion)
        toolbar[2].addWidget(self.MYOstackVersionCheck)
        toolbar[2].addWidget(self.MYOstackVersion1)
        toolbar[2].addWidget(self.rawSignalAction)
        toolbar[2].addWidget(self.EnvelopeSignalAction1)
        toolbar[2].addWidget(self.EnvelopeSignalAction)
        toolbar[2].addWidget(self.envelopeSmoothingСoefficient)
        toolbar[2].addWidget(self.rawSignalAction1)
        toolbar[2].addWidget(self.bandstopAction)
        toolbar[2].addWidget(self.notchActiontypeBox)
        toolbar[2].addWidget(self.bandpassAction2)
        toolbar[2].addWidget(self.bandpassAction)
        toolbar[2].addWidget(self.passLowFreq)
        toolbar[2].addWidget(self.bandpassAction1)
        toolbar[2].addWidget(self.passHighFreq)
        
        # Plot widgets for 1-9 sensors
        self.pw = [] # Plot widget array, index - sensor number
        self.p = [] # Raw data plot, index - sensor number
        self.pe = [] # Envelope data plot, index - sensor number
        
        for i in range(9):
            self.pw.append(pg.PlotWidget(background=(21 , 21, 21, 255)))
            self.pw[i].showGrid(x=True, y=True, alpha=0.7)            
            self.p.append(self.pw[i].plot())
            self.pe.append(self.pw[i].plot())
            self.p[i].setPen(color=(80, 255, 255), width=0.8)
            self.pe[i].setPen(color=(220, 0, 60), width=1)
            self.pw[i].getAxis('bottom').setStyle(showValues=False)
        self.pw[8].getAxis('bottom').setStyle(showValues=True)
        
        for i in range(8):
            self.pw[i+1].setXLink(self.pw[i])
        
        # Plot widget for spectral Plot
        self.pwFFT = pg.PlotWidget(background=(13, 13, 13, 255))
        self.pwFFT.showGrid(x=True, y=True, alpha=0.7) 
        self.pFFT = self.pwFFT.plot()
        self.pFFT.setPen(color=(100, 255, 255), width=1)
        self.pwFFT.setLabel('bottom', 'Frequency', 'Hz')
        
        # Histogram widget
        self.pb = [] # Histogram item array, index - sensor number
        self.pbar = pg.PlotWidget(background=(13 , 13, 13, 255))
        self.pbar.showGrid(x=True, y=True, alpha=0.7)            
        self.pb.append(pg.BarGraphItem(x=np.linspace(1, 2, num=1), height=np.linspace(1, 2, num=1), width=0.3, pen=QtGui.QColor(153, 0, 0), brush=QtGui.QColor(153, 0, 0)))
        self.pb.append(pg.BarGraphItem(x=np.linspace(2, 3, num=1), height=np.linspace(2, 3, num=1), width=0.3, pen=QtGui.QColor(229, 104, 19), brush=QtGui.QColor(229, 104, 19)))
        self.pb.append(pg.BarGraphItem(x=np.linspace(3, 4, num=1), height=np.linspace(3, 4, num=1), width=0.3, pen=QtGui.QColor(221, 180, 10), brush=QtGui.QColor(221, 180, 10)))
        self.pb.append(pg.BarGraphItem(x=np.linspace(4, 5, num=1), height=np.linspace(4, 5, num=1), width=0.3, pen=QtGui.QColor(30, 180, 30), brush=QtGui.QColor(30, 180, 30)))
        self.pb.append(pg.BarGraphItem(x=np.linspace(5, 6, num=1), height=np.linspace(5, 6, num=1), width=0.3, pen=QtGui.QColor(11, 50, 51), brush=QtGui.QColor(11, 50, 51)))
        self.pb.append(pg.BarGraphItem(x=np.linspace(6, 7, num=1), height=np.linspace(6, 7, num=1), width=0.3, pen=QtGui.QColor(29, 160, 191), brush=QtGui.QColor(29, 160, 191)))
        self.pb.append(pg.BarGraphItem(x=np.linspace(7, 8, num=1), height=np.linspace(7, 8, num=1), width=0.3, pen=QtGui.QColor(30, 30, 188), brush=QtGui.QColor(30, 30, 188)))
        self.pb.append(pg.BarGraphItem(x=np.linspace(8, 9, num=1), height=np.linspace(8, 9, num=1), width=0.3, pen=QtGui.QColor(75, 13, 98), brush=QtGui.QColor(75, 13, 98)))
        self.pb.append(pg.BarGraphItem(x=np.linspace(9, 10, num=1), height=np.linspace(9, 10, num=1), width=0.3, pen=QtGui.QColor(139, 0, 55), brush=QtGui.QColor(139, 0, 55)))
        for i in range(9):
            self.pbar.addItem(self.pb[i])  
        self.pbar.setLabel('bottom', 'Sensor number')
        
        # Style
        centralStyle = "color: rgb(255, 255, 255); background-color: rgb(13, 13, 13);"
        
        # Numbering of graphs
        backLabel = []
        for i in range(5):
            backLabel.append(QtWidgets.QLabel(""))
            backLabel[i].setStyleSheet("font-size: 25px; background-color: rgb(21, 21, 21);")
        
        numberLabel = []
        for i in range(9):
            numberLabel.append(QtWidgets.QLabel(" " + str(i+1) + " "))
        numberLabel[0].setStyleSheet("font-size: 25px; background-color: rgb(153, 0, 0); border-radius: 14px;")
        numberLabel[1].setStyleSheet("font-size: 25px; background-color: rgb(229, 104, 19); border-radius: 14px;") 
        numberLabel[2].setStyleSheet("font-size: 25px; background-color: rgb(221, 180, 10); border-radius: 14px;")
        numberLabel[3].setStyleSheet("font-size: 25px; background-color: rgb(30, 180, 30); border-radius: 14px;")
        numberLabel[4].setStyleSheet("font-size: 25px; background-color: rgb(11, 50, 51); border-radius: 14px;")
        numberLabel[5].setStyleSheet("font-size: 25px; background-color: rgb(29, 160, 191); border-radius: 14px;")
        numberLabel[6].setStyleSheet("font-size: 25px; background-color: rgb(30, 30, 188); border-radius: 14px;")
        numberLabel[7].setStyleSheet("font-size: 25px; background-color: rgb(75, 13, 98); border-radius: 14px;")
        numberLabel[8].setStyleSheet("font-size: 25px; background-color: rgb(139, 0, 55); border-radius: 14px;")
        
        self.gainLabel  = []
        self.gainBox  = []
        for i in range(9):
            self.gainLabel.append(QtWidgets.QLabel("GAIN: 1000 x"))
            self.gainBox.append(QtWidgets.QSpinBox())
            self.gainBox[i].setRange(1, 10)
            self.gainBox[i].setValue(1)
        
        # Main widget
        centralWidget = QtWidgets.QWidget()
        centralWidget.setStyleSheet(centralStyle)
        
        self.textWindow = QtWidgets.QPlainTextEdit()
        self.textWindow.setReadOnly(True)
        
        self.textWindow.insertPlainText(datetime.now().strftime("[%H:%M:%S] ") + "program launched\n")
        
        # Layout
        vbox = QtWidgets.QVBoxLayout()
        
        topleft = QtWidgets.QFrame()
        topleft.setFrameShape(QtWidgets.QFrame.StyledPanel)
        
        plotLayout = []
        self.row = []
        for i in range(9):
            plotLayout.append(QtWidgets.QGridLayout())
            plotLayout[i] = QtWidgets.QGridLayout()
            if i % 2 == 0: plotLayout[i].addWidget(backLabel[int(i/2)], 0, 0, 10, 1)
            plotLayout[i].addWidget(numberLabel[i], 0, 0, 10, 1, Qt.AlignVCenter)
            plotLayout[i].addWidget(self.pw[i], 0, 1, 10, 50) 
            plotLayout[i].addWidget(self.gainLabel[i], 0, 49) 
            plotLayout[i].addWidget(self.gainBox[i], 0, 50) 
            plotLayout[i].setContentsMargins(0, 0, 0, 0)
            
            self.row.append(QtWidgets.QWidget())
            self.row[i].setLayout(plotLayout[i])
            
        splitter = QtWidgets.QSplitter(Qt.Vertical)
        splitter.handle(100)
        for i in range(9): splitter.addWidget(self.row[i])
        
        layout = QtWidgets.QGridLayout()       
        layout.addWidget(splitter, 0, 0, 40, 4)
        layout.addWidget(self.pbar, 0, 4, 20, 11)
        layout.addWidget(self.pwFFT, 20, 4, 16, 11)
        layout.setColumnStretch(2, 2)

        layout.addWidget(self.sensorSelectedAction , 20, 13, 1, 1)
        layout.addWidget(self.sensorSelectedActionBox , 20, 14, 1, 1)  
        
        layout.addWidget(self.textWindow, 37, 4, 3, 12)   
        
        vbox.addLayout(layout)
        centralWidget.setLayout(vbox)
        self.setCentralWidget(centralWidget)  
        self.showMaximized()
        self.show()    
        
        # Serial monitor
        self.serialMonitor = SerialMonitor(self.delay)
        ports = [self.COMports.itemText(i) for i in range(self.COMports.count())]
        
        for i in range(len(self.serialMonitor.ports)):
                if self.serialMonitor.ports[i] not in ports:
                    self.COMports.addItem(self.serialMonitor.ports[i])
                    
        if self.serialMonitor.COM != '':
            self.serialMonitor.serialConnect()
            self.liveFromSerialAction.setChecked(True)
            self.dataRecordingAction.setDisabled(False)
            self.sensorsNumber.setDisabled(False)
            self.textWindow.insertPlainText(datetime.now().strftime("[%H:%M:%S] ") + "live from " + self.serialMonitor.COM +" \n")
            self.textWindow.verticalScrollBar().setValue(self.textWindow.verticalScrollBar().maximum()-2)
            self.COMports.setDisabled(True)
            self.refreshAction.setDisabled(False)  
            self.pauseAction.setDisabled(False)  
        
        self.sensorsNumber.valueChanged.connect(self.setSensorsNumber) 
        self.mainrun = MainRun(self.delay)
        self.mainrun.bufferUpdated.connect(self.updateListening, QtCore.Qt.QueuedConnection)  
        
    def liveFromSerial(self):
        if self.liveFromSerialAction.isChecked():
            self.refresh()
            self.serialMonitor.serialConnect()
            self.textWindow.insertPlainText(datetime.now().strftime("[%H:%M:%S] ") + "live from " + self.serialMonitor.COM +" \n")
            self.textWindow.verticalScrollBar().setValue(self.textWindow.verticalScrollBar().maximum()-2)
            self.PlaybackAction.setChecked(False)
            self.refreshAction.setDisabled(False)   
            self.pauseAction.setDisabled(False)
            self.dataRecordingAction.setDisabled(False)
            self.dataRecordingAction.setChecked(False) 
            self.COMports.setDisabled(True)
            self.slider.setDisabled(True)
            self.slider.setFixedWidth(40)
            self.sensorsNumber.setDisabled(False)

        else:
            self.refresh()
            self.serialMonitor.serialDisconnection()
            self.textWindow.insertPlainText(datetime.now().strftime("[%H:%M:%S] ") + "live stopped\n")
            self.textWindow.verticalScrollBar().setValue(self.textWindow.verticalScrollBar().maximum()-2)
            self.refreshAction.setDisabled(True)   
            self.pauseAction.setDisabled(True)
            self.dataRecordingAction.setDisabled(True)
            self.dataRecordingAction.setChecked(False)
            self.COMports.setDisabled(False)
            self.sensorsNumber.setDisabled(True)
           
    # Start working
    def start(self):
        self.mainrun.running = True
        self.mainrun.start()
    
    # Pause data plotting
    def pause(self):
        if self.pauseAction.isChecked():
            self.mainrun.running = False
            self.textWindow.insertPlainText(datetime.now().strftime("[%H:%M:%S] ") + "pause ON" + "\n")
            self.textWindow.verticalScrollBar().setValue(self.textWindow.verticalScrollBar().maximum()-2)
        else:
            self.mainrun.running = True
            self.mainrun.start()
            self.textWindow.insertPlainText(datetime.now().strftime("[%H:%M:%S] ") + "pause OFF" + "\n")
            self.textWindow.verticalScrollBar().setValue(self.textWindow.verticalScrollBar().maximum()-2)

    # Refresh data
    def refresh(self):
        self.l = 0
        self.Time = [0]*self.dataWidth
        self.Data = np.zeros((9, self.dataWidth))
        self.DataEnvelope = np.zeros((9, self.dataWidth))
        self.MovingAverage = MovingAverage(self.fs)
        self.FFT = np.zeros((9, 2000)) 
        self.msg_end = bytearray([0])     
        self.ms_len =  0
        self.slider.setValue(0)

    # Refresh screen
    def refreshForAction(self):
        self.refresh()
        self.textWindow.insertPlainText(datetime.now().strftime("[%H:%M:%S] ") + "refresh" + "\n")
        self.textWindow.verticalScrollBar().setValue(self.textWindow.verticalScrollBar().maximum()-2)      
         
    # Initialize recording data to a file
    def dataRecording(self):
        if (self.dataRecordingAction.isChecked()):
            self.sensorsNumber.setDisabled(True)
            self.refreshAction.setDisabled(True)   
            self.pauseAction.setDisabled(True)
            
            self.recordingFileName_TXT = datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".txt"
            self.recordingFileName_BIN = datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".bin"
            self.textWindow.insertPlainText(datetime.now().strftime("[%H:%M:%S] ") + "recording to \"" + os.getcwd() +"\\" + self.recordingFileName_BIN + "\"\n")
            self.textWindow.verticalScrollBar().setValue(self.textWindow.verticalScrollBar().maximum()-2)

            self.recordingFile_TXT = open(self.recordingFileName_TXT, "a") # Data file creation
            self.recordingFile_TXT.write(datetime.now().strftime("Date: %Y.%m.%d\rTime: %H:%M:%S") + "\r\n") # Data file name
            self.recordingFile_TXT.write("File format: \r\ntime in s | 9 sensor data points \r\n") # Data file format
            
            self.recordingFile_BIN = open(self.recordingFileName_BIN, 'ab')
        else:
            if not self.PlaybackAction.isChecked():
                self.refreshAction.setDisabled(False)
            self.recordingFile_TXT.close()
            self.recordingFile_BIN.close()
            self.pauseAction.setDisabled(False)
            self.sensorsNumber.setDisabled(False)
            self.textWindow.insertPlainText(datetime.now().strftime("[%H:%M:%S] ") + "recording stopped. Result file: \"" + os.getcwd() + self.recordingFileName_BIN + "\"\n")
            self.textWindow.verticalScrollBar().setValue(self.textWindow.verticalScrollBar().maximum()-2)
                
    # Selecting playback file
    def dataLoad(self):
        if self.liveFromSerialAction.isChecked():
            self.dataRecordingAction.setChecked(False)
            self.refreshAction.setDisabled(False)    
            self.pauseAction.setDisabled(False)
        self.recordingFileName_TXT = ''
        path = QtWidgets.QFileDialog.getOpenFileName(self, 'Open a file', '',
                                        'All Files (*.bin*)')
        if path != ('', ''):
            self.loadFileName = str(path[0])
            self.textWindow.insertPlainText(datetime.now().strftime("[%H:%M:%S] ") + "playback file selected: " + self.loadFileName + "\n")
            self.textWindow.verticalScrollBar().setValue(self.textWindow.verticalScrollBar().maximum()-2)
            self.PlaybackAction.setText("Start/Stop playback from file: \n" + self.loadFileName)
            self.PlaybackAction.setDisabled(False)
    
    # Playback initialization 
    def Playback(self):
        if self.PlaybackAction.isChecked():
            self.dataRecordingAction.setChecked(False)
            self.slider.setDisabled(False)
            self.slider.setFixedWidth(300)
            if self.liveFromSerialAction.isChecked():
                self.liveFromSerialAction.setChecked(False)
            self.refresh()
            self.liveFromSerialAction.setChecked(False)
            self.serialMonitor.serialDisconnection()
            self.dataRecordingAction.setDisabled(False)  
            self.refreshAction.setDisabled(True) 
            self.pauseAction.setDisabled(False)  
            self.COMports.setDisabled(False)
            self.sensorsNumber.setDisabled(False)
            
            self.loadFile = open(self.loadFileName, 'rb')
            
            self.textWindow.insertPlainText(datetime.now().strftime("[%H:%M:%S] ") + "playback from: " + self.loadFileName + "\n")
            self.textWindow.verticalScrollBar().setValue(self.textWindow.verticalScrollBar().maximum()-2)
            
            self.loadData = self.loadFile.read()
            self.loadDataLen = int(len(self.loadData)/9/2)
            self.loadFile.close()
            
        else:
            self.slider.setDisabled(True)
            self.slider.setFixedWidth(40)
            self.refresh()
            self.dataRecordingAction.setDisabled(True)
            self.textWindow.insertPlainText(datetime.now().strftime("[%H:%M:%S] ") + "playback stopped \n")
            self.textWindow.verticalScrollBar().setValue(self.textWindow.verticalScrollBar().maximum()-2)
            self.pauseAction.setDisabled(True)  

    # Update
    def updateListening(self): 
        if (not self.liveFromSerialAction.isChecked()):
            self.serialMonitor.updatePorts()
                   
            ports = [self.COMports.itemText(i) for i in range(self.COMports.count())]
            
            for i in range(self.COMports.count()):
                if self.COMports.itemText(i) not in self.serialMonitor.ports:
                    self.COMports.removeItem(i)
                    
            for i in range(len(self.serialMonitor.ports)):
                if self.serialMonitor.ports[i] not in ports:
                    self.COMports.addItem(self.serialMonitor.ports[i])
            
            if self.serialMonitor.COM != self.COMports.currentText():
                self.serialMonitor.COM = self.COMports.currentText()
                self.serialMonitor.connect = False
        
        if len(self.COMports) == 0: 
            self.COMports.addItem("NO PORTS")
            self.COMports.setDisabled(True)
            self.liveFromSerialAction.setDisabled(True)
        else:
            self.COMports.setDisabled(False)
            self.liveFromSerialAction.setDisabled(False)
            
        
        if self.passLowFreq.value() > self.passHighFreq.value(): self.passLowFreq.setValue(self.passHighFreq.value())
        self.passLowFrec = self.passLowFreq.value()
        self.passHighFrec = self.passHighFreq.value()
        
        if self.bandpassAction.isChecked():
            self.passLowFreq.setDisabled(False)
            self.passHighFreq.setDisabled(False)
        else:
            self.passLowFreq.setDisabled(True)
            self.passHighFreq.setDisabled(True)
        
        if self.bandstopAction.isChecked(): 
            self.notchActiontypeBox.setDisabled(False)
        else:
            self.notchActiontypeBox.setDisabled(True)
            
        if self.EnvelopeSignalAction.isChecked():
            self.envelopeSmoothingСoefficient.setDisabled(False)
            self.MovingAverage.MA_alpha = self.envelopeSmoothingСoefficient.value()
        else:
            self.envelopeSmoothingСoefficient.setDisabled(True)
        
        # Read data from File               
        if (self.PlaybackAction.isChecked() and self.loadFileName != ''):
            self.readFromFile()
        
        # Read data from serial          
        if (self.liveFromSerialAction.isChecked()):
            self.readFromSerial()
            
        while self.sensorSelectedActionBox.count() < int(self.sensorsNumber.value()): 
            self.sensorSelectedActionBox.addItem(str(self.sensorSelectedActionBox.count() + 1))
            
        while self.sensorSelectedActionBox.count() > int(self.sensorsNumber.value()): 
            self.sensorSelectedActionBox.removeItem(self.sensorSelectedActionBox.count()-1)
                    
        # Filtering
        if (self.PlaybackAction.isChecked() and self.loadFileName != '') or (self.liveFromSerialAction.isChecked()):
            Data = np.zeros((9, self.dataWidth))
            Time = np.zeros((9, self.dataWidth))
            
            for i in range( int(self.sensorsNumber.value()) ):
                Data[i] = np.concatenate((self.Data[i][self.l: self.dataWidth], self.Data[i][0: self.l]))
                Time = np.concatenate((self.Time[self.l: self.dataWidth], self.Time[0: self.l]))
            
                if self.bandstopAction.isChecked():
                    if (self.notchActiontypeBox.currentText() == "50 Hz"): 
                        for j in range(4): 
                            Data[i] = self.butter_bandstop_filter(Data[i], 45 + j*50, 55 + j*50, self.fs)
                    if (self.notchActiontypeBox.currentText() == "60 Hz"):
                        for j in range(4): Data[i] = self.butter_bandstop_filter(Data[i], 55 + j*60, 65 + j*60, self.fs)
                                
                if (self.bandpassAction.isChecked()) :
                    Data[i] = self.butter_bandpass_filter(Data[i], self.passLowFrec, self.passHighFrec, self.fs)
            
                # Shift the boundaries of the graph
                self.pw[i].setXRange(self.timeWidth*(self.Time[self.l - 1] // self.timeWidth), self.timeWidth*((self.Time[self.l - 1] // self.timeWidth + 1)))
                
                # Plot raw data
                if  self.rawSignalAction.isChecked(): self.p[i].setData(y=Data[i], x=Time)
                else: self.p[i].clear()
                
                self.DataEnvelope[i][0: self.dataWidth - self.ms_len] = self.DataEnvelope[i][self.ms_len:self.dataWidth]
                for j in range (self.dataWidth - self.ms_len, self.dataWidth):
                    self.DataEnvelope[i][j] = int(self.MovingAverage.movingAverage(i, Data[i][j]))
                
                # Plot envelope data
                if  self.EnvelopeSignalAction.isChecked(): self.pe[i].setData(y=self.DataEnvelope[i], x=Time)
                else: self.pe[i].clear()
                    
                # Plot histogram
                self.pb[i].setOpts(height=2*self.DataEnvelope[i][-1])
            
            for i in range( int(self.sensorsNumber.value()), 9):
                self.p[i].clear()
                self.pe[i].clear()
                self.pb[i].setOpts(height=0)
            
            self.ms_len = 0       
            
            # Plot FFT data
            Y = np.zeros((9, 2000))
            for i in range(int(self.sensorsNumber.value())):
                Y[i] = abs(fft(Data[i][-2001: -1]))/2000
                self.FFT[i] = (1-0.5)*Y[i] + 0.5*self.FFT[i]
            X = self.fs*np.linspace(0, 1, 2000)
            sensor = self.sensorSelectedActionBox.currentIndex()
            self.pFFT.setData(y=self.FFT[sensor][2: int(len(self.FFT[sensor])/2)], x=X[2: int(len(X)/2)])
        else:
            for i in range(int(self.sensorsNumber.value())):
                self.p[i].clear()
                self.pe[i].clear()
                self.pb[i].setOpts(height=0)
            self.pFFT.clear()

    # Read data from File   
    def readFromFile(self):
        if self.MYOstackVersionCheck.currentText() == "v1.0" or (self.MYOstackVersionCheck.currentText() == "v2.0"):
            coefficient = 3.25/4.094
            ref = 2048
        else:
            coefficient = 3.25/1.024
            ref = 1024
        
        j = 0
        while j < 100:
            j += 1
            
            if ( self.l == self.dataWidth):
                self.l = 0
            
            if ( self.sliderpos > self.loadDataLen - 2):
                self.refresh()
                self.sliderpos = 0
                self.slider.setValue(0)
                    
            unpeck_b = struct.unpack("H H H H H H H H H", self.loadData[self.sliderpos*9*2:(self.sliderpos+1)*9*2])
            for i in range(9): self.Data[i][self.l] = (unpeck_b[i] - ref)*coefficient/self.gainBox[i].value()
            
            if ((self.slider.value() != int(self.sliderpos/self.loadDataLen*100))):
                self.sliderpos += int(self.slider.value()*self.loadDataLen/100 - self.sliderpos)
                temp = self.l
                self.refresh()
                self.l = temp
                self.Time[self.l] = self.sliderpos*self.dt
            
            self.Time[self.l] = self.sliderpos*self.dt
            self.l = self.l + 1
            self.ms_len += 1
            
            self.sliderpos += 1
            self.slider.setValue(int(self.sliderpos/self.loadDataLen*100))
            
        
    # Read data from serial                  
    def readFromSerial(self): 
        
        coefficient = 1
        if self.MYOstackVersionCheck.currentText() == "v1.0" or (self.MYOstackVersionCheck.currentText() == "v2.0"):
            coefficient = 3.25/4.094
            ref = 2048
        else:
            coefficient = 3.25/1.024
            ref = 1024
        
        msg = self.serialMonitor.serialRead()
        # Parsing data from serial buffer
        msg = msg.decode(errors='ignore')
        if len(msg) >= 2:
            msg_end_n = msg.rfind("\r", 1)
            msg_begin = self.msg_end
            self.msg_end = msg[msg_end_n:len(msg)]
            if(self.l > 2):
                msg = msg_begin + msg[0:msg_end_n]
            for st in msg.split('\r\n'):
                s = st.split(';')
                if (len(s) == 9) :
                    data = [0]*9
                    if ( self.l == self.dataWidth):
                        self.l = 0
                    for i in range(9):
                        if s[i].isdigit():
                            data[i] = int(s[i])
                            self.Data[i][self.l] = (data[i] - ref)*coefficient/self.gainBox[i].value()
                        else:
                            self.Data[i][self.l] = 0;
                        
                    self.Time[self.l] = self.Time[self.l - 1] + self.dt
                    
                    if (self.dataRecordingAction.isChecked()):
                        bin_data = struct.pack("H H H H H H H H H", data[0], data[1], data[2],
                                                data[3], data[4], data[5], data[6], data[7], data[8])
                        
                        
                        self.recordingFile_BIN.write(bin_data)
                        
                        sensors_data = str(round(self.Time[self.l], 3))
                        for i in range(9): sensors_data += (" " + str(round(self.Data[i][self.l], 3)))
                        self.recordingFile_TXT.write(sensors_data + " \n")
                    
                    self.l = self.l + 1
                    self.ms_len += 1                                   
    
    # Butterworth bandpass filter
    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=4):
        nyq = 0.5*fs
        low = lowcut/nyq
        high = highcut/nyq
        b, a = butter(order, [low, high], btype='bandpass')
        y = lfilter(b, a, data)
        return y
    
    # Butterworth bandstop filter
    def butter_bandstop_filter(self, data, lowcut, highcut, fs, order=4):
        nyq = 0.5*fs
        low = lowcut/nyq
        high = highcut/nyq
        b, a = butter(order, [low, high], btype='bandstop')
        y = lfilter(b, a, data)
        return y
 
    def setSensorsNumber(self, num):
        if self.liveFromSerialAction.isChecked():
            self.refresh()
        
        for i in range(9):
            self.row[i].hide()
            self.pw[i].getAxis('bottom').setStyle(showValues=False)
            self.pw[i].showLabel('bottom', 0)
        self.pw[int(num)-1].getAxis('bottom').setStyle(showValues=True)
        
        self.pbar.clear()
        for i in range(int(num)):  
            self.pbar.addItem(self.pb[i])  
            self.row[i].show()
            
    # Exit event
    def closeEvent(self, event):
        self.mainrun.running = False
        self.serialMonitor.serialDisconnection()
        event.accept()

# Serial monitor class
class SerialMonitor:
    # Custom constructor
    def __init__(self, delay):
        self.running = False
        self.connect = False
        self.baudRate = 115200
        self.playFile = 0
        self.delay = delay      
        self.ports = [p[0] for p in serial.tools.list_ports.comports(include_links=False) ]
        self.COM = ''
        self.ser = serial.Serial()
        if len(self.ports) > 0:
            self.COM = self.ports[0]
        
    def updatePorts(self):
        self.ports = [p[0] for p in serial.tools.list_ports.comports(include_links=False) ]
    
    def serialConnect(self):
        self.updatePorts()
        if not self.connect:
            if self.COM != '':
                try:
                    self.ser = serial.Serial(self.COM, self.baudRate)
                    self.ser.setDTR(False)
                    self.ser.setRTS(False)
                    self.connect = True  
                except SerialException :
                    self.connect = False
                    
    def serialDisconnection(self):
        self.ser.close()
        self.connect = False
        
    def serialRead(self):  
        msg = bytes(0)
        try:
            msg = self.ser.read( self.ser.inWaiting() )
        except SerialException :
            try:
               self.ser.close()
               self.ser.open()
               msg = bytes(0)
            except SerialException :
                pass
            pass
        return msg

# Moving average class
class MovingAverage:
    # Custom constructor
    def __init__(self, fs):
        self.MA = np.zeros((9, 3)) 
        self.MA_alpha = 0.95
        self.Y0 = np.zeros(9)
        self.X0 = np.zeros(9)
        self.fs = fs
    
    def movingAverage(self, i, data):
        wa = 2.0*self.fs*np.tan(3.1416*1/self.fs)
        HPF = (2*self.fs*(data-self.X0[i]) - (wa-2*self.fs)*self.Y0[i])/(2*self.fs+wa)
        self.Y0[i] = HPF
        self.X0[i] = data
        data = HPF
        if data < 0:
            data = -data
        self.MA[i][0] = (1 - self.MA_alpha)*data + self.MA_alpha*self.MA[i][0];
        self.MA[i][1] = (1 - self.MA_alpha)*(self.MA[i][0]) + self.MA_alpha*self.MA[i][1];
        self.MA[i][2] = (1 - self.MA_alpha)*(self.MA[i][1]) + self.MA_alpha*self.MA[i][2];
        return self.MA[i][2]*2

# Serial monitor class
class MainRun(QtCore.QThread):
    bufferUpdated = QtCore.pyqtSignal()
    # Custom constructor
    def __init__(self, delay):
        QtCore.QThread.__init__(self)
        self.running = False
        self.playFile = 0
        self.delay = delay      

    # Listening port
    def run(self):
        while self.running is True:
            self.bufferUpdated.emit()
            time.sleep(self.delay) 
         
# Starting program       
if __name__ == '__main__':
    app = QtCore.QCoreApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    window = GUI()
    window.show()
    window.start()
    sys.exit(app.exec_())

