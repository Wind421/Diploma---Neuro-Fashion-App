import sys

from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt,QTimer
from PyQt5.QtMultimedia import QCamera, QCameraImageCapture
from PyQt5.QtMultimediaWidgets import QCameraViewfinder

import pygame
import gtts

import torch.serialization
import fastai.callback.progress
import fastai.callback.fp16
import fastai.vision.learner
from fastai.learner import load_learner

from PIL import Image
import pandas as pd
import os

#import resources

def save_voice(text,name):
    if os.path.exists(name+'.mp3'):
        os.remove(name+'.mp3')

    language = 'ru'
    speech = gtts.gTTS(text=text, lang=language, slow=False)
    speech.save(name+'.mp3')

def play_text(text):
    pygame.mixer.init()
    pygame.mixer.music.load(text+'.mp3')
    pygame.mixer.music.play()

def output_to_test(category,attributs):
    attr_cloth = pd.read_csv('attr_cloth.csv', sep=';')
    category_cloth = pd.read_csv('category_cloth.csv',sep=';')
    
    decription_cloth = 'Это '
    decription_cloth += str(category_cloth.loc[category, 'russian'])
    
    decription_cloth+=' имеет, '
    for attr in attributs:
        if int(attr)==1000:
            continue
        decription_cloth+=str(attr_cloth.loc[int(attr), 'russian'])
        decription_cloth+=', '

    phrases = decription_cloth.split(', ')
    unique_phrases = list(set(phrases))
    unique_phrases.sort(key=phrases.index)
    decription_cloth = ', '.join(unique_phrases)

    decription_cloth = decription_cloth[:-2] + '.'   
    print(decription_cloth)    
    return decription_cloth

def neuro_output():

    img=Image.open(r'captured_image.jpg')
    img.thumbnail((224,224),Image.LANCZOS)

    cat_model = load_learner(r'cat_resnet34_export.pkl')
    attr_model = load_learner(r'attr_resnet50_export.pkl')

    category = int(cat_model.predict(img)[0])-1
    attributs = attr_model.predict(img)[0]

    decription = output_to_test(category,attributs)

    save_voice(decription,'last_des')

class CameraApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.clickCount = 0  
        self.clickExitCount = 0 
        self.clickDesCount=0
        self.currentView = None

    def initUI(self):
        self.setWindowTitle('Приложение для захвата фотографии')
        self.setGeometry(100, 100, 1100, 800)

        self.captureButton = QPushButton('Сделать \nснимок', self)
        self.captureButton.clicked.connect(self.captureImage)
        self.captureButton.setFixedSize(270,100) 
        self.captureButton.move(820,260)

        self.describeButton = QPushButton('Описать \nснимок', self)
        self.describeButton.clicked.connect(self.describeCloth)
        self.describeButton.setFixedSize(270,100) 
        self.describeButton.move(820,370)
        
        self.exitButton = QPushButton("Выход", self)
        self.exitButton.clicked.connect(self.exitDef) 
        self.exitButton.setFixedSize(270,100) 
        self.exitButton.move(820,480)

        self.cameraLabel = QLabel(self)
        self.cameraLabel.setGeometry(800, 780, 800, 780)
        self.cameraLabel.move(10,10)
        self.cameraLabel.setAlignment(Qt.AlignCenter)

        self.photoLabel = QLabel(self)
        self.photoLabel.setGeometry(800, 780, 800, 780)
        self.photoLabel.move(10,10)
        self.photoLabel.setAlignment(Qt.AlignCenter) 

        # Установка цвета рамки
        self.setStyleSheet("""QPushButton 
                           {background-color: #4080FF;
                           color: #ffffff; 
                           font-size: 40px;
                           font-weight: bold;
                           border-radius: 10px;
                           }""")

        self.camera = QCamera()
        self.imageCapture = QCameraImageCapture(self.camera)
        self.imageCapture.imageCaptured.connect(self.displayImage)

        self.viewfinder = QCameraViewfinder(self.cameraLabel)
        self.viewfinder.setFixedSize(self.cameraLabel.size())
        self.camera.setViewfinder(self.viewfinder)
        self.camera.start() 

    def displayImage(self, id, preview):
        if self.clickCount == 0:
            play_text('voice')
            self.clickCount = 1
        else:
            image_jpg = QPixmap(preview)
            image_jpg.save('captured_image.jpg')
            self.photoLabel.setPixmap(image_jpg)

            play_text('itog')

            QTimer.singleShot(2000, lambda: self.photoLabel.clear())
            self.clickCount=0
            self.camera.start()

    def captureImage(self):
        self.imageCapture.capture()

    def exitDef(self):
        if self.clickExitCount == 0:
            play_text('exit_but')
            self.clickExitCount=1
        else:
            play_text('you_exit')
            self.clickExitCount=0
            self.close()

    def describeCloth(self):
        if self.clickDesCount == 0:
            play_text('des_but')
            self.clickDesCount=1
        else:
            play_text('neuro_start')
            neuro_output()
            play_text('last_des')
            self.clickDesCount=0

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = CameraApp()
    ex.show()
    sys.exit(app.exec_())