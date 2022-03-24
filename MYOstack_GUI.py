# Graphical interface for signal visualization and interaction with ELEMYO MYOstack sensors
# 2021-04-23 by ELEMYO (https://github.com/ELEMYO/ELEMYO GUI)
# 
# Changelog:
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

from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import Qt
import sys
import serial
import pyqtgraph as pg
import numpy as np
import time
from scipy.signal import butter, lfilter
import serial.tools.list_ports
from datetime import datetime
from scipy.fftpack import fft

# Main window
class GUI(QtWidgets.QMainWindow):
    # Initialize constructor
    def __init__(self):
          super(GUI, self).__init__()
          self.initUI()
    # Custom constructor
    def initUI(self): 
        # Values
        COM = '' # Example: COM='COM9'
        baudRate = 1000000 # Serial frequency
        self.delay = 0.06 # Delay for graphic update
        
        self.gain = [1, 1, 1, 1, 1, 1, 1, 1, 1] # Sensors gain, index is the sensor number
        
        self.setWindowTitle("MYOstack GUI v1.0.1 | ELEMYO" + "    ( COM Port not found )")
        self.setWindowIcon(QtGui.QIcon('img/icon.png'))
        self.f = open(datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".txt", "w") # Data file creation
        self.f.write(datetime.now().strftime("Date: %Y.%m.%d\rTime: %H:%M:%S") + "\r\n") # Data file head
        self.f.write("File format: \r\nseconds | data1 | data2 | data3 | data4 | data5 | data6 | data7| data8 | data9 \r\n") # Data file format
        self.l = 0 # Current data point
        self.dt = 0.002 # Time between two signal measurements in s
        self.fs = 1/self.dt # Signal discretization frequency in Hz
        self.passLowFrec = 10 # Low frequency for passband filter
        self.passHighFrec = 200 # Low frequency for passband filter
        self.dataWidth = int(6.2/self.dt) # Maximum count of ploting data points (6.2 secondes vindow)
        self.Time = [0]*self.dataWidth # Time array
        self.timeWidth = 5 # Time width of plot
        self.Data = np.zeros((9, self.dataWidth)) # Raw data matrix, first index - sensor number, second index - sensor data 
        self.DataEnvelope = np.zeros((9, self.dataWidth)) # Envelope of row data, first index - sensor number, second index - sensor data 
        
        # Accessory variables for envelope (for moving average method)
        self.MA = np.zeros((9, 3)) 
        self.MA_alpha = 0.95
        self.Y0 = np.zeros(9)
        self.X0 = np.zeros(9)
        
        # Accessory variables for data read from serial
        self.ms_len = 0;
        self.msg_end = np.array([0])
        
        self.loopNumber = 0; # Loop number
        self.FFT = 0 # Fast Fourier transform data
        
        self.selectedSensor = 1 # Sensor number selected from GUI
        self.selectedGain = 1 # Sensor gain selected from GUI

        # Menu panel
        stopAction = QtWidgets.QAction(QtGui.QIcon('img/pause.png'), 'Stop/Start (Space)', self)
        stopAction.setShortcut('Space')
        stopAction.triggered.connect(self.stop)
        refreshAction = QtWidgets.QAction(QtGui.QIcon('img/refresh.png'), 'Refresh (R)', self)
        refreshAction.setShortcut('r')
        refreshAction.triggered.connect(self.refresh)
        exitAction = QtWidgets.QAction(QtGui.QIcon('img/out.png'), 'Exit (Esc)', self)
        exitAction.setShortcut('Esc')
        exitAction.triggered.connect(self.close)
        
        # Toolbar
        toolbar = self.addToolBar('Tool')
        toolbar.addAction(stopAction)
        toolbar.addAction(refreshAction)
        toolbar.addAction(exitAction)
        
        # Plot widgets for 1-9 sensor
        self.pw = [] # Plot widget array, index - sensor number
        self.p = [] # Raw data plot, index - sensor number
        self.pe = [] # Envelope data plot, index - sensor number
        for i in range(9):
            self.pw.append(pg.PlotWidget(background=(21 , 21, 21, 255)))
            self.pw[i].showGrid(x=True, y=True, alpha=0.7) 
            self.p.append(self.pw[i].plot())
            self.pe.append(self.pw[i].plot())
            self.p[i].setPen(color=(100, 255, 255), width=0.8)
            self.pe[i].setPen(color=(255, 0, 0), width=1)
        
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
        
        # Styles
        centralStyle = "color: rgb(255, 255, 255); background-color: rgb(13, 13, 13);"
        editStyle = "border-style: solid; border-width: 1px;"
        
        # Settings zone
        filtersText = QtWidgets.QLabel("FILTERS:")
        self.passLowFreq = QtWidgets.QLineEdit(str(self.passLowFrec), self)
        self.passLowFreq.setMaximumWidth(100)
        self.passLowFreq.setStyleSheet(editStyle)
        self.passHighFreq = QtWidgets.QLineEdit(str(self.passHighFrec), self)
        self.passHighFreq.setMaximumWidth(100)
        self.passHighFreq.setStyleSheet(editStyle)
        self.bandpass = QtWidgets.QCheckBox("BANDPASS FILTER:")
        self.bandstop50 = QtWidgets.QCheckBox("NOTCH 50 Hz")
        self.bandstop60 = QtWidgets.QCheckBox("NOTCH 60 Hz")
        
        plotStyle = QtWidgets.QLabel("PLOT STYLE: ")
        self.signal = QtWidgets.QCheckBox("Signal")
        self.envelope = QtWidgets.QCheckBox("Envelope")
        self.signal.setChecked(True)
        
        self.envelopeSmoothing = QtWidgets.QLabel(" Envelope smoothing:")
        self.envelopeSmoothingСoefficient = QtWidgets.QLineEdit(str(self.MA_alpha), self)
        self.envelopeSmoothingСoefficient.setMaximumWidth(100)
        self.envelopeSmoothingСoefficient.setStyleSheet(editStyle)
        
        self.sensorNumberText = QtWidgets.QLabel("Sensor number:")
        self.sensorGainText = QtWidgets.QLabel("Sensor gain:")
        self.sensorNumber = QtWidgets.QLineEdit(str(self.selectedSensor), self)
        self.sensorNumber.setMaximumWidth(100)
        self.sensorNumber.setStyleSheet(editStyle)
        self.sensorGain = QtWidgets.QLineEdit(str(self.selectedGain), self)
        self.sensorGain.setMaximumWidth(100)
        self.sensorGain.setStyleSheet(editStyle)
        
        # Buttons for selecting sensor for FFT analysis
        fftButton = []
        for i in range(9):
            fftButton.append(QtWidgets.QRadioButton(str(i + 1)))
            fftButton[i].Value = i + 1
        fftButton[0].setChecked(True)
        self.button_group = QtWidgets.QButtonGroup()
        for i in range(9):
            self.button_group.addButton(fftButton[i], i + 1)
        self.button_group.buttonClicked.connect(self._on_radio_button_clicked)
        
        
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
        
        # Main widget
        centralWidget = QtWidgets.QWidget()
        centralWidget.setStyleSheet(centralStyle)
        
        # Layout
        vbox = QtWidgets.QVBoxLayout()
        
        layout = QtWidgets.QGridLayout()
        layout.addWidget(backLabel[0], 0, 1)
        layout.addWidget(numberLabel[0], 0, 1, Qt.AlignVCenter)
        layout.addWidget(numberLabel[1], 1, 1, Qt.AlignVCenter)
        layout.addWidget(backLabel[1], 2, 1)
        layout.addWidget(numberLabel[2], 2, 1, Qt.AlignVCenter)
        layout.addWidget(numberLabel[3], 3, 1, Qt.AlignVCenter)
        layout.addWidget(backLabel[2], 4, 1, 4, 1)
        layout.addWidget(numberLabel[4], 4, 1, 4, 1, Qt.AlignVCenter)
        layout.addWidget(numberLabel[5], 8, 1, Qt.AlignVCenter)
        layout.addWidget(backLabel[3], 9, 1)
        layout.addWidget(numberLabel[6], 9, 1, Qt.AlignVCenter)
        layout.addWidget(numberLabel[7], 10, 1, Qt.AlignVCenter)
        layout.addWidget(backLabel[4], 11, 1, 4, 1)
        layout.addWidget(numberLabel[8], 11, 1, 4, 1, Qt.AlignVCenter)
        
        layout.addWidget(self.pw[0], 0, 2, 1, 2)
        layout.addWidget(self.pw[1], 1, 2, 1, 2)
        layout.addWidget(self.pw[2], 2, 2, 1, 2)
        layout.addWidget(self.pw[3], 3, 2, 1, 2)
        layout.addWidget(self.pw[4], 4, 2, 4, 2)
        layout.addWidget(self.pw[5], 8, 2, 1, 2)
        layout.addWidget(self.pw[6], 9, 2, 1, 2)
        layout.addWidget(self.pw[7], 10, 2, 1, 2)
        layout.addWidget(self.pw[8], 11, 2, 4, 2)
        layout.addWidget(self.pbar, 0, 4, 4, 10)
        layout.addWidget(self.pwFFT, 4, 4, 7, 10)
        layout.setColumnStretch(2, 2)
        

        layout.addWidget(fftButton[0], 4, 10)
        layout.addWidget(fftButton[1], 4, 11) 
        layout.addWidget(fftButton[2], 4, 12) 
        layout.addWidget(fftButton[3], 5, 10)
        layout.addWidget(fftButton[4], 5, 11)  
        layout.addWidget(fftButton[5], 5, 12)
        layout.addWidget(fftButton[6], 6, 10)
        layout.addWidget(fftButton[7], 6, 11)  
        layout.addWidget(fftButton[8], 6, 12)      
        layout.addWidget(filtersText, 11, 4) 
        layout.addWidget(self.bandstop50, 11, 5) 
        layout.addWidget(self.bandstop60, 11, 6)
        layout.addWidget(self.bandpass, 11, 7) 
        layout.addWidget(self.passLowFreq, 11, 8) 
        layout.addWidget(self.passHighFreq, 11, 9)
        layout.addWidget(plotStyle, 12, 4)
        layout.addWidget(self.signal, 12, 5) 
        layout.addWidget(self.envelope, 12, 6)  
        layout.addWidget(self.envelopeSmoothing, 12, 7)      
        layout.addWidget(self.envelopeSmoothingСoefficient, 12, 8)
        layout.addWidget(self.sensorNumberText, 13, 4)
        layout.addWidget(self.sensorNumber, 13, 5)
        layout.addWidget(self.sensorGainText, 13, 6)
        layout.addWidget(self.sensorGain, 13, 7)
        
        vbox.addLayout(layout)
        centralWidget.setLayout(vbox)
        self.setCentralWidget(centralWidget)  
        self.showMaximized()
        self.show()
        # Serial monitor
        self.monitor = SerialMonitor(COM, baudRate, self.delay)
        self.monitor.bufferUpdated.connect(self.updateListening, QtCore.Qt.QueuedConnection)
    # Start working
    def start(self):
        self.monitor.running = True
        self.monitor.start()
    # Pause
    def stop(self):
        if self.monitor.running == False:
            self.monitor.running = True
            self.monitor.start()
        else:
            self.monitor.running = False
    # Refresh
    def refresh(self):
        self.l = 0 #Current point
        self.Time = [0]*self.dataWidth #Tine array
        self.Data = np.zeros((9, self.dataWidth))
        self.DataEnvelope = np.zeros((9, self.dataWidth))
        self.Time = [0]*self.dataWidth
        self.msg_end = 0        
        self.loopNumber = 0;
    # Update
    def updateListening(self, msg):
        # Update variables
        self.setWindowTitle("MYOstack GUI v1.0.1 | ELEMYO " + 
                            "    ( " + self.monitor.COM + " , " + str(self.monitor.baudRate) + " baud )")
        s = self.passLowFreq.text()
        if s.isdigit():
            self.passLowFrec = float(s)
        s = self.passHighFreq.text()
        if s.isdigit():
            self.passHighFrec = float(self.passHighFreq.text())
        
        s = self.envelopeSmoothingСoefficient.text()
        try:
            if (float(s) >= 0) and (float(s) <= 1):
                self.MA_alpha= float(s)
        except ValueError:
            pass
        
        s = self.sensorNumber.text()
        if s.isdigit():
            self.selectedSensor = int(s)
        
        s = self.sensorGain.text()
        try:
            self.selectedGain = float(s)
            if (self.selectedSensor in range(1, 10)) and (self.selectedGain in range(1, 12)):
                self.gain[self.selectedSensor - 1] = self.selectedGain
        except ValueError:
            pass
        
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
                    if ( self.l == self.dataWidth):
                        self.l = 0
                    for i in range(9):
                        if s[i].isdigit():
                            self.Data[i][self.l] = int(s[i])/4.094*3.3/self.gain[i]
                        else:
                            self.Data[i][self.l] = 0;
                        
                    self.Time[self.l] = self.Time[self.l - 1] + self.dt
                    self.f.write(str(round(self.Time[self.l], 3)) + " " + str(self.Data[0][self.l]) + " " + str(self.Data[1][self.l]) + " "
                                 + str(self.Data[2][self.l]) + " " + str(self.Data[3][self.l]) + " "
                                 + str(self.Data[4][self.l]) + " " + str(self.Data[5][self.l]) + " "
                                 + str(self.Data[6][self.l]) + " " + str(self.Data[7][self.l]) + " "
                                 + str(self.Data[8][self.l]) + "\r\n")
                    
                    self.l = self.l + 1
                    self.loopNumber += 1
                    self.ms_len += 1
                
        # Filtering
        Data = np.zeros((9, self.dataWidth))
        for i in range(9):
            Data[i] = np.concatenate((self.Data[i][self.l: self.dataWidth], self.Data[i][0: self.l]))
         
        Time = self.Time[self.l: self.dataWidth] + self.Time[0: self.l]
        
        self.monitor.delay = self.delay
        if self.bandstop50.isChecked() == 1:
            if self.fs > 110: 
                for i in range(9): Data[i] = self.butter_bandstop_filter(Data[i], 48, 52, self.fs)
            if self.fs > 210: 
                for i in range(9): Data[i] = self.butter_bandstop_filter(Data[i], 98, 102, self.fs)
            if self.fs > 310: 
                for i in range(9): Data[i] = self.butter_bandstop_filter(Data[i], 148, 152, self.fs)
            if self.fs > 410: 
                for i in range(9): Data[i] = self.butter_bandstop_filter(Data[i], 195, 205, self.fs)
            self.monitor.delay = self.delay + 0.03
        if self.bandstop60.isChecked() == 1:
            if self.fs > 130:
                for i in range(9): Data[i] = self.butter_bandstop_filter(Data[i], 58, 62, self.fs)
            if self.fs > 230:
                for i in range(9): Data[i] = self.butter_bandstop_filter(Data[i], 118, 122, self.fs)
            if self.fs > 330:
                for i in range(9): Data[i] = self.butter_bandstop_filter(Data[i], 158, 162, self.fs)
            self.monitor.delay = self.delay + 0.03
        if ((self.bandpass.isChecked() == 1 or (self.signal.isChecked() == 1 and self.envelope.isChecked() == 1)) and self.passLowFrec < self.passHighFrec 
            and self.passLowFrec > 0 and self.fs > 2*self.passHighFrec):
            for i in range(9):
                Data[i] = self.butter_bandpass_filter(Data[i], self.passLowFrec, self.passHighFrec, self.fs)
            self.monitor.delay = self.delay + 0.04
        
        for i in range(9):
            self.DataEnvelope[i][0: self.dataWidth - self.ms_len] = self.DataEnvelope[i][self.ms_len:self.dataWidth]
        for j in range (self.dataWidth - self.ms_len, self.dataWidth):
            for i in range(9):
                self.DataEnvelope[i][j] = self.movingAverage(i, Data[i][j], self.MA_alpha)
        self.ms_len = 0
               
        l = 0 # length of filter fluctuation tail
        # Removing filter fluctuation tail from plot
        if self.bandpass.isChecked() == 1 or self.bandstop50.isChecked() == 1 or self.bandstop60.isChecked() == 1 or self.envelope.isChecked() == 1:
            l = int(1/self.dt)
            if self.loopNumber < self.dataWidth:
                for i in range(9): 
                    if self.l >= l: 
                        Data[i][0: self.dataWidth - self.l + l] = Data[i][self.dataWidth - self.l + l]
                        self.DataEnvelope[i][0: self.dataWidth - self.l + l]=self.DataEnvelope[i][self.dataWidth - self.l + l]
                    else:
                        Data[i] = Data[i][self.dataWidth - self.l]
                        self.DataEnvelope[i] = self.DataEnvelope[i][self.dataWidth - self.l]
                l = 0
        if self.loopNumber < self.dataWidth:
            l = self.dataWidth - self.l
        
        # Shift the boundaries of the graph
        timeCount = self.Time[self.l - 1] // self.timeWidth
        for i in range(9):
            self.pw[i].setXRange(self.timeWidth*timeCount, self.timeWidth*(timeCount + 1))            
        
        # Plot raw and envelope data
        if  self.signal.isChecked() == 1 and self.envelope.isChecked() == 1:
            for i in range(9):
                self.p[i].setData(y=Data[i][l: self.dataWidth], x=Time[l: self.dataWidth])
                self.pe[i].setData(y=self.DataEnvelope[i][l: self.dataWidth], x=Time[l: self.dataWidth])
            self.monitor.delay += 0.02
        
        # Plot envelope data            
        if self.signal.isChecked() == 0 and self.envelope.isChecked() == 1:
            for i in range(9):
                self.pe[i].setData(y=self.DataEnvelope[i][l: self.dataWidth], x=Time[l: self.dataWidth])
                self.p[i].clear()
                
        # Plot raw data 
        if self.signal.isChecked() == 1 and self.envelope.isChecked() == 0:
            for i in range(9):
                self.p[i].setData(y=Data[i][l: self.dataWidth], x=Time[l: self.dataWidth])
                self.pe[i].clear()
                        
        # Plot histogram
        for i in range(9):
            self.pb[i].setOpts(height=2*self.DataEnvelope[i][-1])
        
        # Plot FFT data
        Y = abs(fft(Data[self.button_group.checkedId() - 1][-501: -1]))/500
        X = 1/self.dt*np.linspace(0, 1, 500)
        self.FFT = (1-0.85)*Y + 0.85*self.FFT
        self.pFFT.setData(y=self.FFT[2: int(len(self.FFT)/2)], x=X[2: int(len(X)/2)]) 
                    
    # Values for butterworth bandpass filter
    def butter_bandpass(self, lowcut, highcut, fs, order=4):
        nyq = 0.5*fs
        low = lowcut/nyq
        high = highcut/nyq
        b, a = butter(order, [low, high], btype='bandpass')
        return b, a
    # Butterworth bandpass filter
    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=4):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y
    # Values for butterworth bandstop filter
    def butter_bandstop(self, lowcut, highcut, fs, order=2):
        nyq = 0.5*fs
        low = lowcut/nyq
        high = highcut/nyq
        b, a = butter(order, [low, high], btype='bandstop')
        return b, a
    # Butterworth bandstop filter
    def butter_bandstop_filter(self, data, lowcut, highcut, fs, order=4):
        b, a = self.butter_bandstop(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y
    def movingAverage(self, i, data, alpha):
        wa = 2.0*self.fs*np.tan(3.1416*1/self.fs)
        HPF = (2*self.fs*(data-self.X0[i]) - (wa-2*self.fs)*self.Y0[i])/(2*self.fs+wa)
        self.Y0[i] = HPF
        self.X0[i] = data
        data = HPF
        if data < 0:
            data = -data
        self.MA[i][0] = (1 - alpha)*data + alpha*self.MA[i][0];
        self.MA[i][1] = (1 - alpha)*(self.MA[i][0]) + alpha*self.MA[i][1];
        self.MA[i][2] = (1 - alpha)*(self.MA[i][1]) + alpha*self.MA[i][2];
        return self.MA[i][2]*2
    # Change gain
    def _on_radio_button_clicked(self, button):
        if self.monitor.COM != '':
            self.monitor.ser.write(bytearray([button.Value]))
    # Exit event
    def closeEvent(self, event):
        self.f.close()
        self.monitor.ser.close()
        event.accept()

# Serial monitor class
class SerialMonitor(QtCore.QThread):
    bufferUpdated = QtCore.pyqtSignal(bytes)
    # Custom constructor
    def __init__(self, COM, baudRate, delay):
        QtCore.QThread.__init__(self)
        self.running = False
        self.filter = False
        self.COM = COM
        self.baudRate = baudRate
        self.baudRate = baudRate
        self.checkPort = 1
        self.delay = delay

    # Listening port
    def run(self):
        while self.running is True:
            while self.COM == '': 
                ports = serial.tools.list_ports.comports(include_links=False)
                for port in ports :
                    self.COM = port.device
                if self.COM != '':
                    time.sleep(0.5)
                    self.ser = serial.Serial(self.COM, self.baudRate)
                    self.checkPort = 0
            while self.checkPort:
                ports = serial.tools.list_ports.comports(include_links=False)
                for port in ports :
                    if self.COM == port.device:
                        time.sleep(0.5)
                        self.ser = serial.Serial(self.COM, self.baudRate)
                        self.checkPort = 0
                   
            # Waiting for data
            while (self.ser.inWaiting() == 0):
                pass
            # Reading data
            msg = self.ser.read( self.ser.inWaiting() )
            if msg:
                #Parsing data
                self.bufferUpdated.emit(msg)
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
