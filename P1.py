import sys
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import music21
import wave
import pyaudio
import queue
import threading
import time
import struct
import math
import scipy

from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QPushButton, QLabel, QSizePolicy
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot, Qt, QTimer

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from scipy import signal
from scipy.signal import fftconvolve


CHUNK_UNIT = 1024
TIME = 4
CHUNK = CHUNK_UNIT * TIME
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = 60
PLOT_XLIM = 5
WAVE_OUTPUT_FILENAME = 'data/wav/output.wav'
WAVE_INPUT_FILENAME = 'data/wav/happy.wav'
SCORE_FILENAME = 'data/166951-Happy_Birthday_To_You_C_Major.mxl'

WINDOW = signal.hamming(CHUNK)
WAVE_DATA = np.array(([]))
FFT_DATA = np.array(([]))
FFT_DATA2 = np.array(([]))
Q = queue.Queue()

IS_SYSTEM_RUN = True

IS_LOG = False  # Log is very slow
logFile = 'log/' + str(int(time.time())) + '.log'
LOGGER = open(logFile, 'w')
LOGID = 0

SCORE_MEASURE_DATA = np.array([])
SCORE_BEAT_DATA = np.array([])
SCORE_NOTE_DATA = np.array([])
SCORE_MATCH_TIME_DATA = np.array([])

LIVE_MATCH_TIME = 0
LIVE_CURRENT_TIME = 0
SCORE_MATCH_INDEX = -1
LIVE_DATA = np.array([])

INFO_MEASURE = ''
INFO_BEAT = ''
INFO_PITCH_NAME = ''


class App(QMainWindow):

    def __init__(self):
        super().__init__()
        self.title = 'PyQt5 simple window'
        self.left = 200
        self.top = 0
        self.width = 800
        self.height = 800
        self.initUI()
        self.initTimer()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.statusBar().showMessage('Message in statusbar.')

        self.m = PlotCanvas(self, width=8, height=6)
        self.m.move(0, 0)

        # button = QPushButton('PyQt5 button', self)
        # button.setToolTip('This is an example button')
        # button.move(100, 400)
        # button.clicked.connect(self.on_click)

        self.label1 = QLabel('Measure:', self)
        self.label1.setTextFormat(Qt.RichText)
        self.label1.move(50, 630)
        self.label1.setMinimumWidth(200)
        self.label1.setText('<font color="red" size="20">Measure: 0</font>')

        self.label2 = QLabel('Beat:', self)
        self.label2.setTextFormat(Qt.RichText)
        self.label2.move(300, 630)
        self.label2.setMinimumWidth(200)
        self.label2.setText('<font color="green" size="20">Beat: 0</font>')

        self.label3 = QLabel('Pitch:', self)
        self.label3.setTextFormat(Qt.RichText)
        self.label3.move(550, 630)
        self.label3.setMinimumWidth(200)
        self.label3.setText('<font color="blue" size="20">Pitch: </font>')

        self.show()
        print('App show()')

        # while True:
        #     print('App')

    def initTimer(self):
        self.timer = QTimer(self)  # 初始化一个定时器
        self.timer.timeout.connect(self.on_timer)  # 计时结束调用operate()方法
        self.timer.start(0.1)  # 设置计时间隔并启动

    def refreshPitch(self):
        # print('Refresh Pitch:', INFO_PITCH_NAME)
        # self.label3.setText(INFO_PITCH_NAME)
        self.label1.setText(
            '<font color="red" size="20">Measure: ' + INFO_MEASURE + '</font>')
        self.label2.setText(
            '<font color="green" size="20">Beat: ' + INFO_BEAT + '</font>')
        self.label3.setText(
            '<font color="blue" size="20">Pitch: ' + INFO_PITCH_NAME + '</font>')

    @pyqtSlot()
    def on_click(self):
        print('PyQt5 button click')
        self.m.refresh()

    @pyqtSlot()
    def on_timer(self):
        # print('Timer')
        self.refreshPitch()
        self.m.refresh()


class PlotCanvas(FigureCanvas):

    def __init__(self, parent=None, width=8, height=6, dpi=100):
        # fig = Figure(figsize=(width, height), dpi=dpi)
        fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(
            3, figsize=(width, height), dpi=dpi)

        # self.axes = fig.add_subplot(111)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.plot()

    def plot(self):
        self.plotChunk = 1024
        self.plotYlim = 10000

        x = np.arange(0, self.plotChunk, 1)
        x_fft2 = np.linspace(0, RATE, CHUNK_UNIT / 2)
        x_fft = np.arange(0, RATE, 2)

        self.ax1.set_title('Wave')
        self.ax1.set_xlim(0, self.plotChunk)
        self.ax1.set_ylim(-self.plotYlim, self.plotYlim)
        self.line1, = self.ax1.plot(x, np.zeros_like(x))

        self.ax2.set_title('FFT2')
        self.ax2.set_xlim(20, RATE / 2)
        self.line2, = self.ax2.plot(x_fft2, np.zeros_like(x_fft2))

        self.ax3.set_title('FFT')
        self.ax3.set_xlim(20, RATE / 2)
        self.line3, = self.ax3.plot(x_fft, np.zeros_like(x_fft))

        # x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # data1 = [random.random() for i in range(10)]
        # data2 = [random.random() for i in range(10)]
        # data3 = [random.random() for i in range(10)]
        # self.line1, = self.ax1.plot(x, data1)
        # self.line2, = self.ax2.plot(x, data2)
        # self.line3, = self.ax3.plot(x, data3)
        self.draw()
        # print('Plot draw()')

    def refresh(self):
        global WAVE_DATA, FFT_DATA, FFT_DATA2
        waveLength = len(WAVE_DATA)

        plotData = np.zeros(self.plotChunk)
        if waveLength < self.plotChunk:
            plotData[0:waveLength] = WAVE_DATA[0:waveLength]
        else:
            plotData[0:self.plotChunk] = WAVE_DATA[-self.plotChunk:]

        ylim = np.int32(np.max(np.abs(plotData)) * 1.1)
        if ylim <= 2000:
            ylim = 2000
        self.ax1.set_ylim(-ylim, ylim)
        self.line1.set_ydata(plotData)

        if len(FFT_DATA2 > 0):
            ylim = np.int32(np.max(FFT_DATA2) * 1.1)
            if ylim <= 10:
                ylim = 10
            self.ax2.set_ylim(0, ylim)
            self.line2.set_ydata(FFT_DATA2)

        if len(FFT_DATA > 0):
            ylim = np.int32(np.max(FFT_DATA) * 1.1)
            if ylim <= 1:
                ylim = 1
            self.ax3.set_ylim(0, ylim)
            self.line3.set_ydata(FFT_DATA)

        # data1 = [random.random() for i in range(10)]
        # data2 = [random.random() for i in range(10)]
        # data3 = [random.random() for i in range(10)]
        # self.line1.set_ydata(data1)
        # self.line2.set_ydata(data2)
        # self.line3.set_ydata(data3)
        self.draw()


class ProducerThread(threading.Thread):
    def __init__(self, arg):
        super(ProducerThread, self).__init__()  # 注意：一定要显式的调用父类的初始化函数。
        self.arg = arg

    def run(self):  # 定义每个线程要运行的函数
        print('Run Producer:%s\r' % self.arg)

        #
        # Read wave from Microphone
        #
        p = pyaudio.PyAudio()

        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        # input_device_index = 0,
                        frames_per_buffer=CHUNK)

        print("* recording")

        frames = []
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK, exception_on_overflow=False)
            # frames.append(data)
            Q.put(data)
            print('Producer:', i, len(data))
            # print(data)

        print("* done recording")

        stream.stop_stream()
        stream.close()
        p.terminate()

        # #
        # # Read wave from file
        # #
        # f = wave.open(WAVE_INPUT_FILENAME, 'rb')
        # params = f.getparams()
        # nchannels, sampwidth, framerate, nframes = params[:4]
        # print(params)

        # # data = f.readframes(30960)
        # # data = f.readframes(CHUNK)
        # # Q.put(data)
        # for i in range(0, int(nframes / CHUNK)):
        #     data = f.readframes(CHUNK)
        #     Q.put(data)
        #     print('Producer:', i, len(data))
        #     # print(data)


class ConsumerThread(threading.Thread):
    def __init__(self, arg):
        super(ConsumerThread, self).__init__()  # 注意：一定要显式的调用父类的初始化函数。
        self.arg = arg
        self.pitchNameList = None
        self.freqHarmonicArray = None

    def run(self):  # 定义每个线程要运行的函数
        print('Run Consumer:%s\r' % self.arg)

        while True:
            while not Q.empty():
                data = Q.get()
                self.pipeline(data)
        print("* Close Consumer")

    def pipeline(self, data):
        print('Consumer Pipeline:', len(data))
        # waveData = np.array(struct.unpack(str(4 * CHUNK) + 'B', data),
        #                     dtype='b')
        # waveData = np.array(struct.unpack('%dh' % (len(data)), data))
        waveData = np.fromstring(data, dtype=np.int16)
        # print(waveData.shape, np.max(waveData), np.min(waveData))
        waveData.shape = -1, 2
        waveData = waveData.T

        global TIME
        for i in range(TIME):
            start = CHUNK_UNIT * i
            end = start + CHUNK_UNIT
            waveDataUnit = waveData[0][start:end]

            global FFT_DATA, FFT_DATA2
            # FFT_DATA, _, frequency = self.fft(waveDataUnit)
            FFT_DATA2, _, frequency2 = self.fft2(waveDataUnit)
            # # LOGGER.write('\n\n' + str(frequency) + '\n')
            # # LOGGER.write('\n' + str(frequency2) + '\n\n')
            # # corr = self.autocorrelationFunction(waveDataUnit)

            global WAVE_DATA
            WAVE_DATA = np.concatenate((WAVE_DATA, waveDataUnit))
            if len(WAVE_DATA) > RATE * PLOT_XLIM:
                WAVE_DATA = WAVE_DATA[CHUNK_UNIT:]

            pitchName, pitchConfidence, pitchLoss = self.pitchDetect(
                frequency2, numPitch=1)

            if pitchConfidence < 6:
                pitchName = 'No'

            global INFO_PITCH_NAME
            INFO_PITCH_NAME = pitchName

            global LIVE_DATA
            LIVE_DATA = np.append(LIVE_DATA, pitchName)

        self.pitchScoreMatch()

        # # print('WAVE_DATA:', len(WAVE_DATA))
        # print('Frequency:', pitchName, pitchConfidence,
        #       pitchLoss, frequency, frequency2, corr, '\n')
        # # print('Frequency:', frequency, frequency2, corr, '\n')

    def fft(self, recordedSignal):
        n = len(recordedSignal)
        signal = np.zeros(RATE)
        signal[0:n] = recordedSignal[0:n]
        recordedSignal = signal

        n = RATE
        # nUniquePts = int(np.ceil((n+1)/2.0))
        nUniquePts = n // 2

        # recordedSignal = recordedSignal * hanning(n, sym=0)
        fft = np.fft.fft(recordedSignal)
        fftOriginal = fft = fft[0:nUniquePts]
        fft = np.abs(fft) / float(n)
        fft = fft**2

        # odd nfft excludes Nyquist point
        if n % 2 > 0:  # we've got odd number of points fft
            fft[1:len(fft)] = fft[1:len(fft)] * 2
        else:
            fft[1:len(fft) - 1] = fft[1:len(fft) - 1] * \
                2  # we've got even number of points fft

        rms = np.sqrt(np.mean(np.int32(recordedSignal)**2)) - \
            np.sqrt(np.sum(fft))
        if rms > 1e-5:
            print('rms:', np.sqrt(np.mean(np.int32(recordedSignal)**2)),
                  np.sqrt(np.sum(fft)))

        # frequency = np.argmax(fft)
        # frequency = np.argmax(fft) / n * RATE

        freqs = np.arange(0, RATE // 2, 1)
        # frequency = freqs[self.peakSearch(fft)]
        peak = self.peakSearch(fft)
        frequency = self.peakSort(fft[peak], freqs[peak], numPeak=10)

        return fft, fftOriginal, frequency

    def fft2(self, recordedSignal):
        n = len(recordedSignal)
        # fft = scipy.fftpack.fft(recordedSignal)
        fft = np.fft.fft(recordedSignal)
        fftOriginal = fft = fft[0:(n // 2)]
        fft = np.abs(fft) * 2 / float(n)
        fft = fft**2

        if n % 2 > 0:
            fft[1:len(fft)] = fft[1:len(fft)] * 2
        else:
            fft[1:len(fft) - 1] = fft[1:len(fft) - 1] * \
                2

        freqs = np.arange(0, len(fft), 1) / n * RATE
        # frequency = np.argmax(fft) / n * RATE
        peak = self.peakSearch(fft)
        frequency = self.peakSort(fft[peak], freqs[peak], numPeak=10)

        return fft, fftOriginal, frequency

    def autocorrelationFunction(self, recordedSignal):
        correlation = fftconvolve(
            recordedSignal, recordedSignal[::-1], mode='full')
        lengthCorrelation = len(correlation) // 2
        correlation = correlation[lengthCorrelation:]

        # Calculates the difference between slots
        difference = np.diff(correlation)
        positiveDifferences = matplotlib.mlab.find(difference > 0)
        if len(positiveDifferences) == 0:  # pylint: disable=len-as-condition
            finalResult = 10  # Rest
        else:
            beginning = positiveDifferences[0]
            peak = np.argmax(correlation[beginning:]) + beginning
            finalResult = RATE / peak

        return finalResult

    def peakSearch(self, data, threshold=0):
        dataPrefix = np.insert(data, 0, data[0])
        dataSuffix = np.append(data, data[-1])

        diffPrefix = np.diff(dataPrefix)
        diffSuffix = np.diff(dataSuffix)

        return (diffPrefix > 0) & (diffSuffix <= 0)

    def peakSort(self, fft, freq, numPeak=5):
        freqArray = np.array([])
        fftArray = np.copy(fft)
        for i in range(numPeak):
            idx = np.argmax(fftArray)
            freqArray = np.append(freqArray, freq[idx])
            fftArray[idx] = 0
        return freqArray

    def preparePitchHarmonicFrequency(self, numHarmonic=10):
        # numHarmonic = 5
        pitchNameList = []
        freqHarmonicArray = np.array([])

        useScale = music21.scale.ChromaticScale('C3')
        for p in useScale.pitches:
            if p.nameWithOctave == 'C4':
                continue

            pitchNameList.append(p.nameWithOctave)
            freqFundamental = p.frequency
            freqsHarmonic = list()
            for i in range(numHarmonic):
                freqsHarmonic.append(freqFundamental * (i + 1))
            freqHarmonicArray = np.append(
                freqHarmonicArray, np.array(freqsHarmonic), axis=0)

        useScale = music21.scale.ChromaticScale('C4')
        for p in useScale.pitches:
            if p.nameWithOctave == 'C5':
                continue

            pitchNameList.append(p.nameWithOctave)
            freqFundamental = p.frequency
            freqsHarmonic = list()
            for i in range(numHarmonic):
                freqsHarmonic.append(freqFundamental * (i + 1))
            freqHarmonicArray = np.append(
                freqHarmonicArray, np.array(freqsHarmonic), axis=0)

        useScale = music21.scale.ChromaticScale('C5')
        for p in useScale.pitches:
            if p.nameWithOctave == 'C6':
                continue

            pitchNameList.append(p.nameWithOctave)
            freqFundamental = p.frequency
            freqsHarmonic = list()
            for i in range(numHarmonic):
                freqsHarmonic.append(freqFundamental * (i + 1))
            freqHarmonicArray = np.append(
                freqHarmonicArray, np.array(freqsHarmonic), axis=0)

        freqHarmonicArray.shape = (-1, numHarmonic)
        # print(freqDict.shape, freqDict)
        return pitchNameList, freqHarmonicArray

    def pitchDetect(self, freqData, numPitch=1):
        # step 2: prepare harmonic matrix
        if (self.pitchNameList == None) | (self.freqHarmonicArray == None):
            self.pitchNameList, self.freqHarmonicArray = self.preparePitchHarmonicFrequency()

        # freqData = np.array([2652, 1580, 3194, 4307, 1540, 2627, 2112])
        lossThreshold = 0.02
        weight = 1
        resultSum = np.zeros_like(self.freqHarmonicArray)

        # step 2: calculate harmonic weight
        for freq in freqData:
            result = self.freqHarmonicArray / freq
            result = np.abs(result - 1)
            result[result > lossThreshold] = 0
            result[result > 0] += weight
            resultSum += result

        # step 3: calculate harmonic sum
        pitchConfidenceSumArray = np.sum(resultSum, axis=1)

        # step 4: calculate confidence and loss
        pitchCostArray, pitchConfidenceArray = np.modf(pitchConfidenceSumArray)
        pitchLossArray = np.zeros_like(pitchCostArray)
        pitchLossArray[pitchCostArray > 0] = pitchCostArray[pitchCostArray >
                                                            0] / pitchConfidenceArray[pitchCostArray > 0] * 100.0

        # step 5: calculate harmonic attention
        pitchConfidenceAttention = list()
        numPitch, numHaronic = resultSum.shape
        for i in range(numPitch):
            attention = 0
            attentionMax = 0
            for j in range(numHaronic):
                if resultSum[i][j] > 0:
                    attention += 1
                else:
                    if attention > attentionMax:
                        attentionMax = attention
                    attention = 0

            if attention > attentionMax:
                attentionMax = attention
            pitchConfidenceAttention.append(attentionMax)
        pitchConfidenceArray = pitchConfidenceArray * \
            np.array(pitchConfidenceAttention)

        # TODO: numPitch
        idx = np.argmax(pitchConfidenceArray)

        pitchName = self.pitchNameList[idx]
        pitchConfidence = pitchConfidenceArray[idx]
        pitchLoss = pitchLossArray[idx]

        if IS_LOG:
            global LOGID, LOGGER
            LOGGER.write('pipeline:\t' + str(LOGID) + '\n')
            LOGGER.write(str(len(freqData)) + '\t' +
                         pitchName + '\t' +
                         str(pitchConfidence) + '\n')
            LOGGER.write(str(pitchConfidenceSumArray[0:12]) + '\n' +
                         str(pitchConfidenceSumArray[12:24]) + '\n' +
                         str(pitchConfidenceSumArray[24:36]) + '\n')
            LOGGER.write(str(pitchConfidenceArray[0:12]) + '\n' +
                         str(pitchConfidenceArray[12:24]) + '\n' +
                         str(pitchConfidenceArray[24:36]) + '\n')
            LOGGER.write('\n')
            LOGID += 1

        return pitchName, pitchConfidence, pitchLoss

    def preparePitchScore(self):
        s = music21.converter.parse(SCORE_FILENAME)

        measureList = list()
        beatList = list()
        noteList = list()
        notePartList = list()

        for p in s.getElementsByClass('Part'):
            notePartList.append(list())

        for n in range(len(notePartList)):
            for m in s.parts[n].getElementsByClass('Measure'):
                for e in m.elements:
                    if type(e) is music21.note.Rest:
                        for i in range(int(e.quarterLength / 0.25)):
                            notePartList[n].append('')
                            if n == 0:
                                measureList.append(m.number)
                                beatList.append(e.beat)
                    if type(e) is music21.note.Note:
                        for i in range(int(e.quarterLength / 0.25)):
                            notePartList[n].append(e.pitch.nameWithOctave)
                            if n == 0:
                                measureList.append(m.number)
                                beatList.append(e.beat)
                    if type(e) is music21.chord.Chord:
                        for i in range(int(e.quarterLength / 0.25)):
                            notePartList[n].append(
                                ''.join(p.nameWithOctave for p in e.pitches))
                            if n == 0:
                                measureList.append(m.number)
                                beatList.append(e.beat)

        # print(notePartList)
        for j in range(len(notePartList[0])):
            note = ''.join(notePartList[i][j]
                           for i in range(len(notePartList)))
            noteList.append(note)

        # print('\n')
        # print(measureList)
        # print(beatList)
        # print(noteList)
        global SCORE_MEASURE_DATA, SCORE_BEAT_DATA, SCORE_NOTE_DATA, SCORE_MATCH_TIME_DATA
        SCORE_MEASURE_DATA = np.array(measureList)
        SCORE_BEAT_DATA = np.array(beatList)
        SCORE_NOTE_DATA = np.array(noteList)
        SCORE_MATCH_TIME_DATA = np.zeros(len(SCORE_NOTE_DATA))

    def pitchScoreMatch(self):
        global SCORE_MEASURE_DATA, SCORE_BEAT_DATA, SCORE_NOTE_DATA
        if len(SCORE_MEASURE_DATA) == 0 | len(SCORE_BEAT_DATA) == 0 | len(SCORE_NOTE_DATA) == 0:
            self.preparePitchScore()

        global SCORE_MATCH_TIME_DATA, SCORE_MATCH_INDEX, LIVE_DATA, LIVE_MATCH_TIME, LIVE_CURRENT_TIME
        LIVE_CURRENT_TIME += 1

        global TIME
        # if LIVE_CURRENT_TIME % 5 == 0:
        # if LIVE_CURRENT_TIME % 1 == 0:
        start = LIVE_MATCH_TIME
        end = LIVE_CURRENT_TIME * TIME
        notesToMatch = np.unique(LIVE_DATA[start:end])
        # ''.join(n for n in np.unique(LIVE_DATA[start:end]))

        start = SCORE_MATCH_INDEX + 1
        end = start + 4
        notesInScore = SCORE_NOTE_DATA[start:end]

        # Match algorithm
        isMatch = False
        for i in range(len(notesInScore)):
            for j in range(len(notesToMatch)):
                found = self.pitchStandard(notesInScore[i]).find(
                    self.pitchStandard(notesToMatch[j]))
                if found != -1:
                    isMatch = True
                    SCORE_MATCH_INDEX = SCORE_MATCH_INDEX + 1 + i
                    LIVE_MATCH_TIME = LIVE_CURRENT_TIME * TIME
                    SCORE_MATCH_TIME_DATA[SCORE_MATCH_INDEX] = LIVE_MATCH_TIME

                    global INFO_MEASURE, INFO_BEAT
                    INFO_MEASURE = str(
                        SCORE_MEASURE_DATA[SCORE_MATCH_INDEX])
                    INFO_BEAT = str(SCORE_BEAT_DATA[SCORE_MATCH_INDEX])

                    print('Match: ', SCORE_MEASURE_DATA[SCORE_MATCH_INDEX],
                          SCORE_BEAT_DATA[SCORE_MATCH_INDEX], SCORE_NOTE_DATA[SCORE_MATCH_INDEX])
                    print('\n')
                    if IS_LOG:
                        LOGGER.write('notesInScore: ' +
                                     str(notesInScore) + '\n')
                        LOGGER.write('notesToMatch: ' +
                                     str(notesToMatch) + '\n')
                        LOGGER.write('Match: ' + str(LIVE_MATCH_TIME) + '/' + str(LIVE_CURRENT_TIME) + '\t' +
                                     str(SCORE_MATCH_INDEX) + '\t' +
                                     str(SCORE_MEASURE_DATA[SCORE_MATCH_INDEX]) + '\t' +
                                     str(SCORE_BEAT_DATA[SCORE_MATCH_INDEX]) + '\t' +
                                     str(SCORE_NOTE_DATA[SCORE_MATCH_INDEX]) + '\n\n')
                    break
            if isMatch:
                break

        # TODO: No Match
        if isMatch == False:
            print('No Match!\n\n')
            if IS_LOG:
                LOGGER.write('No Match: ' + str(LIVE_MATCH_TIME) +
                             '/' + str(LIVE_CURRENT_TIME) + '\n\n')

        self.pitchScoreMatchStat()

    def pitchScoreMatchStat(self):
        print(SCORE_MATCH_TIME_DATA)
        if SCORE_MATCH_TIME_DATA[-1] != 0:
            nomatch = len(SCORE_MATCH_TIME_DATA[SCORE_MATCH_TIME_DATA == 0])
            match = len(SCORE_MATCH_TIME_DATA[SCORE_MATCH_TIME_DATA > 0])
            if IS_LOG:
                LOGGER.write('All: ' + str(len(SCORE_MATCH_TIME_DATA)) + '\n')
                LOGGER.write('No Match: ' + str(nomatch) + '\n')
                LOGGER.write('Match: ' + str(match) + '\n')
                LOGGER.write(str(SCORE_MATCH_TIME_DATA) + '\n\n')

    def pitchStandard(self, pitchName):
        if pitchName.find('D-') != -1:
            pitchName = pitchName.replace('D-', 'C#')
        if pitchName.find('E-') != -1:
            pitchName = pitchName.replace('E-', 'D#')
        if pitchName.find('G-') != -1:
            pitchName = pitchName.replace('G-', 'F#')
        if pitchName.find('A-') != -1:
            pitchName = pitchName.replace('A-', 'G#')
        if pitchName.find('B-') != -1:
            pitchName = pitchName.replace('B-', 'A#')

        return pitchName


if __name__ == '__main__':

    producer = ProducerThread(arg='')
    producer.start()

    consumer = ConsumerThread(arg='')
    consumer.start()

    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
