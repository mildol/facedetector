from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np
from skimage.measure import block_reduce as blrd
import sys
from PyQt5 import QtCore, QtGui, QtWidgets

# класс интерфейса
class Ui_Interface(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(400, 250)
        MainWindow.setMinimumSize(QtCore.QSize(400, 250))
        MainWindow.setMaximumSize(QtCore.QSize(400, 250))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(10, 10, 381, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(10, 40, 141, 20))
        self.lineEdit.setObjectName("lineEdit")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(10, 60, 381, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.lineEdit_2 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_2.setGeometry(QtCore.QRect(10, 90, 141, 20))
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(10, 110, 381, 31))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(10, 140, 141, 21))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.clicked_1)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "Введите количество эталонов:"))
        self.label_2.setText(_translate("MainWindow", "Введите количество людей:"))
        self.label_3.setText(_translate("MainWindow", "Вывести результат детекции:"))
        self.pushButton.setText(_translate("MainWindow", "Вывод"))
        
    def clicked_1(self, Interface):
        Detection(int(self.lineEdit.text()), int(self.lineEdit_2.text())).show_results()
        
# класс изображения
class Image():
    
    #считывание изображения
    def __init__(self, i, j, tp):
        self.tp = tp
        self.image = cv.imread('ORL_Faces/s' + str(i + 1) +'/' + \
                            str(j + 1) + '.pgm')
        
        self.image_gray = cv.imread('ORL_Faces/s' + str(i + 1) +'/' + \
                            str(j + 1) + '.pgm', 0)
    
    #возвращение изображения
    #нужного типа
    def get_image(self):
        if self.tp == 'image':
            return self.image
        
        elif self.tp == 'image_gray':
            return self.image_gray
        
        elif self.tp == 'image_float':
            return np.float32(self.image_gray)

# класс распознавания
class Detection():
    
    #считывание массива для эталона
    #n - кол-во эталонов
    #kk - кол-во классов изображений
    #tp - тип метода
    def __init__(self, n, kk):
        self.n = n
        self.m = kk
        
    #выбор метода
    #x, y - индексация для изображения
    def function(self, x, y, tp):
        if tp == 'bright_hist':
            image = Image(x, y, 'image').get_image()
            return np.histogram(image.ravel(), 256, [0, 256])
        
        elif tp == 'dft':
            image = Image(x, y, 'image_gray').get_image()
            dft = np.fft.fftshift(np.fft.fft2(image))
            return np.log(np.abs(dft))
        
        elif tp == 'dct':
            image = Image(x, y, 'image_float').get_image()
            return np.uint8(cv.dct(image, cv.DCT_ROWS))
        
        elif tp == 'gradient':
            image = Image(x, y, 'image_gray').get_image()
            lap = cv.Laplacian(image, cv.CV_64F, ksize=5)
            return np.uint8(np.absolute(lap))
        
        elif tp == 'scale':
            image = Image(x, y, 'image_gray').get_image()
            return np.array(blrd(image, (2, 1), np.max))

    #вычисление среднего при нескольких эталонах
    def get_arrays(self, tp):
        self.standards = [[] for x in range(self.m)]
        for i in range(self.m):
            func = self.function(i, self.n, tp)[0]
            hist = np.zeros(func.shape)
            
            for j in range(self.n):
                hist += func
            
            self.standards[i] = hist / self.n
    
    #сравнение эталонов с подаваемыми изображениями
    #вывод результата
    def compare_arrays(self, tp):
        self.result = [[] for x in range(self.m)]
        self.get_arrays(tp)
        
        #разница между эталоном и изображением.
        for i in range(self.m):
            for j in range(self.m):
                distance = 0.0
                
                for k in range(self.n, 10):
                    func = self.function(j, k, tp)[0]
                    hist = func
                    
                    for l in range(len(func)):
                        distance += (self.standards[i][l] - hist[l]) ** 2
                    
                distance = np.sqrt(distance) / 10 - self.n
                self.result[i].append(distance)
            
            if np.argmin(self.result[i]) == i:
                self.result[i] = i + 1
            else:
                self.result[i] = np.argmin(self.result[i]) + 1

        return self.result
    
    #вывод результатов
    def get_results(self):
        results = []
        
        for method in 'bright_hist', 'dft', 'dct', 'gradient', 'scale':
            results.append(self.compare_arrays(method))

        return results

    def show_results(self):
        results = self.get_results()
        accuracy = [[] for x in range(len(results))]

        for i in range(len(results)):
            trues = 0

            for j in range(len(results[0])):

                if results[i][j] == j + 1:
                    trues += 1

                accuracy[i].append((trues / (j + 1)) * 100)

        plt.ion()
        plt.figure(figsize=(20, 10))

        for i in range(self.m):
            plt.clf()
            ax1 = plt.subplot(2, 6, 1)
            ax2 = plt.subplot(2, 6, 2)
            ax3 = plt.subplot(2, 6, 3)
            ax4 = plt.subplot(2, 6, 7)
            ax5 = plt.subplot(2, 6, 8)
            ax6 = plt.subplot(2, 6, 9)
            ax7 = plt.subplot(2, 6, 4)
            ax8 = plt.subplot(2, 6, 5)
            ax9 = plt.subplot(2, 6, 6)
            ax10 = plt.subplot(2, 6, 10)
            ax11 = plt.subplot(2, 6, 11)

            ax1.imshow(Image(i, 0, 'image').get_image())
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax1.set_xlabel('Изображение = ' + str(i + 1))

            ax2.imshow(Image(results[0][i] - 1, 0, 'image').get_image())
            ax2.set_xticks([])
            ax2.set_yticks([])
            ax2.set_xlabel('Гистограмма яркости = ' + str(results[0][i]))

            ax3.imshow(Image(results[1][i] - 1, 0, 'image').get_image())
            ax3.set_xticks([])
            ax3.set_yticks([])
            ax3.set_xlabel('DFT = ' + str(results[1][i]))

            ax4.imshow(Image(results[2][i] - 1, 0, 'image').get_image())
            ax4.set_xticks([])
            ax4.set_yticks([])
            ax4.set_xlabel('DCT = ' + str(results[2][i]))

            ax5.imshow(Image(results[3][i] - 1, 0, 'image').get_image())
            ax5.set_xticks([])
            ax5.set_yticks([])
            ax5.set_xlabel('Градиент = ' + str(results[3][i]))

            ax6.imshow(Image(results[4][i] - 1, 0, 'image').get_image())
            ax6.set_xticks([])
            ax6.set_yticks([])
            ax6.set_xlabel('Scale = ' + str(results[4][i]))

            ax7.plot(np.arange(self.m), accuracy[0])
            ax7.set_xlabel('Гистограмма яркости')

            ax8.plot(np.arange(self.m), accuracy[1])
            ax8.set_xlabel('DFT')

            ax9.plot(np.arange(self.m), accuracy[2])
            ax9.set_xlabel('DCT')

            ax10.plot(np.arange(self.m), accuracy[3])
            ax10.set_xlabel('Градиент')

            ax11.plot(np.arange(self.m), accuracy[4])
            ax11.set_xlabel('Scale')

            plt.draw()
            plt.pause(3)

        plt.ioff()
        plt.show()

app = QtWidgets.QApplication(sys.argv)
Interface = QtWidgets.QMainWindow()
ui = Ui_Interface()
ui.setupUi(Interface)
Interface.show()
sys.exit(app.exec_())
