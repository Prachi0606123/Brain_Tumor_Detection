from PyQt4 import QtGui
from PyQt4 import QtCore
from PyQt4.QtGui import QFileDialog
from PyQt4.QtCore import *
from PyQt4.QtGui import *

import login
import home
import error_log
import err_img

import MySQLdb

import numpy as np
import cv2
import os
import sys

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.callbacks import TensorBoard

import tensorflow as tf
from tensorflow.python.keras.models import load_model

import pickle
import time


from matplotlib import pyplot as plt
from skimage.morphology import extrema
from skimage.morphology import watershed as skwater
from sklearn import svm, datasets 



fname=""
folder=""
flag=0

class Login(QtGui.QMainWindow, login.Ui_UserLogin):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.setupUi(self) 
        self.pushButton.clicked.connect(self.log)
        self.pushButton_2.clicked.connect(self.can)
        
    def log(self):
        i=0
        db = MySQLdb.connect("localhost","root","root","user")
        cursor = db.cursor()
        a=self.lineEdit.text()
        b=self.lineEdit_2.text()
        sql = "SELECT * FROM user WHERE username='%s' and pass='%s'" % (a,b)
        try:
            cursor.execute(sql)
            results = cursor.fetchall()
            for row in results:
                i=i+1
        except Exception as e:
           print e
        if i>=0:
            print "login success"
            self.hide()
            self.home=home()
            self.home.show()
            
        else:
            print "login failed"
            self.errlog=errlog()
            self.errlog.show()
                    
        db.close()
        
    def can(self):
        sys.exit()
        
   

        

class home(QtGui.QMainWindow, home.Ui_Home):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.selimg)
        self.pushButton_2.clicked.connect(self.seldir)
        self.pushButton_3.clicked.connect(self.train_cnn)
        self.pushButton_4.clicked.connect(self.severity)
        self.pushButton_7.clicked.connect(self.pred)
        self.pushButton_5.clicked.connect(self.ex)
        self.pushButton_6.clicked.connect(self.preproc)
  

    def selimg(self):
        global fname
        self.QFileDialog = QtGui.QFileDialog(self)
        #self.QFileDialog.show()
        fname = QFileDialog.getOpenFileName(self, 'Open file','c:\\',"Image files (*.jpg *.png)")
        print fname
        label = QLabel(self.label_5)
        pixmap = QPixmap(fname)
        label.setPixmap(pixmap)
        label.resize(pixmap.width(),pixmap.height())
        label.show()

    
    def seldir(self):
        global folder
        self.QFileDialog = QtGui.QFileDialog(self)
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        print folder
        DATADIR = "Datasets" #folder

        CATEGORIES = ["normal", "tumor"]

        training_data = []

        IMG_SIZE = 256

        def create_training_data():
            for category in CATEGORIES:  

                path = os.path.join(DATADIR,category)  
                class_num = CATEGORIES.index(category)  

                for img in tqdm(os.listdir(path)):  
                    try:
                        img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
                        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                        training_data.append([new_array, class_num])  # add this to our training_data
                    except Exception as e:
                        print("general exception", e, os.path.join(path,img))

        create_training_data()

        random.shuffle(training_data)

        X = []
        y = []

        for features,label in training_data:
            X.append(features)
            y.append(label)

        X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

        pickle_out = open("X.pickle","wb")
        pickle.dump(X, pickle_out)
        pickle_out.close()

        pickle_out = open("y.pickle","wb")
        pickle.dump(y, pickle_out)
        pickle_out.close()


    def train_cnn(self):
        pickle_in = open("X.pickle","rb")
        X = pickle.load(pickle_in)

        pickle_in = open("y.pickle","rb")
        y = pickle.load(pickle_in)

        X = X/255.0

        dense_layers = [0]
        layer_sizes = [64]
        conv_layers = [3]

        for dense_layer in dense_layers:
            for layer_size in layer_sizes:
                for conv_layer in conv_layers:
                    NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
                    print(NAME)

                    model = Sequential()

                    model.add(Conv2D(layer_size, (3, 3), input_shape=X.shape[1:]))
                    model.add(Activation('relu'))
                    model.add(MaxPooling2D(pool_size=(2, 2)))

                    for l in range(conv_layer-1):
                        model.add(Conv2D(layer_size, (3, 3)))
                        model.add(Activation('relu'))
                        model.add(MaxPooling2D(pool_size=(2, 2)))

                    model.add(Flatten())

                    for _ in range(dense_layer):
                        model.add(Dense(layer_size))
                        model.add(Activation('relu'))

                    model.add(Dense(1))
                    model.add(Activation('sigmoid'))

                    tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

                    model.compile(loss='binary_crossentropy',
                                  optimizer='adam',
                                  metrics=['accuracy'],
                                  )

                    hiss=model.fit(X, y,
                              batch_size=32,
                              epochs=4,
                              validation_split=0.3,
                              callbacks=[tensorboard])
                    
        model.summary()
        plt.figure(figsize=(15,7))

        plt.subplot(1,2,1)
        plt.plot(hiss.history['acc'], label='train')
        plt.plot(hiss.history['val_acc'], label='validation')
        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1,2,2)
        plt.plot(hiss.history['loss'], label='train')
        plt.plot(hiss.history['val_loss'], label='validation')
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.show()
        model.save('6CNN.model')

    def pred(self):
         global fname
         if fname=="":
             print "please select file first"
             self.errimg=errimg()
             self.errimg.show()

         else:
            global flag
            CATEGORIES = ["normal", "tumor"]
            def prepare(filepath):
                IMG_SIZE = 256  
                img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


            model = load_model("6CNN.model")

            prediction = model.predict([prepare(str(fname))])

            print(prediction)
            print(CATEGORIES[int(prediction[0][0])])
            cat=CATEGORIES[int(prediction[0][0])]
            
            if cat=="normal":
                flag=0
		self.lineEdit.setText("No Tumour Detected")
            if cat=="tumor":
                flag=1
		self.lineEdit.setText("Tumour Detected")	
             
        
    def severity(self):
        global fname
        global flag
        if fname=="":
            self.errimg=errimg()
            self.errimg.show()
        
        if flag==0:
            print "No tumour detected"
        else:
            pickle_in = open("X.pickle","rb")
            X = pickle.load(pickle_in)
            dataset_size = len(X)
            X1 = X.reshape(dataset_size,-1)
            X=X1

            pickle_in = open("y.pickle","rb")
            y = pickle.load(pickle_in)


            C = 1.0 
              
            svc = svm.SVC(kernel ='linear', C = 1).fit(X, y) 
              

            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            h = (x_max / x_min)/100
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), 
                     np.arange(y_min, y_max, h))

            def ShowImage(title,img,ctype):
              plt.figure(figsize=(10, 10))
              if ctype=='bgr':
                b,g,r = cv2.split(img)       # get b,g,r
                rgb_img = cv2.merge([r,g,b])     # switch it to rgb
                plt.imshow(rgb_img)
              elif ctype=='hsv':
                rgb = cv2.cvtColor(img,cv2.COLOR_HSV2RGB)
                plt.imshow(rgb)
              elif ctype=='gray':
                plt.imshow(img,cmap='gray')
              elif ctype=='rgb':
                plt.imshow(img)
              else:
                raise Exception("Unknown colour type")
              plt.axis('off')
              plt.title(title)
              plt.show()
            img           = cv2.imread(str(fname))
            gray          = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            #ShowImage('Brain MRI',gray,'gray')

            ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_OTSU)
            #ShowImage('Thresholding image',thresh,'gray')

            ret, markers = cv2.connectedComponents(thresh)

            #Get the area taken by each component. Ignore label 0 since this is the background.
            marker_area = [np.sum(markers==m) for m in range(np.max(markers)) if m!=0] 
            per=int(marker_area[0])/10
	    per1=str(per)[:2]+"."+str(per)[2:4]+"%"
	    print "Percentage=",str(per)[:2]+"."+str(per)[2:4],"%"
	    self.lineEdit_4.setText(per1)
	    

            
            #Get label of largest component by area
            largest_component = np.argmax(marker_area)+1 #Add 1 since we dropped zero above                        
            
            
            #Get pixels which correspond to the brain
            brain_mask = markers==largest_component

            brain_out = img.copy()
            #In a copy of the original image, clear those pixels that don't correspond to the brain
            brain_out[brain_mask==False] = (0,0,0)

            img = cv2.imread(str(fname))
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

                
            # noise removal
            kernel = np.ones((3,3),np.uint8)
            opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
                
            # sure background area
            sure_bg = cv2.dilate(opening,kernel,iterations=3)
                
            # Finding sure foreground area
            dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
            ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
                
            # Finding unknown region
            sure_fg = np.uint8(sure_fg)
            unknown = cv2.subtract(sure_bg,sure_fg)
               
            # Marker labelling
            ret, markers = cv2.connectedComponents(sure_fg)
                 
            # Add one to all labels so that sure background is not 0, but 1
            markers = markers+1

                 
            # Now, mark the region of unknown with zero
            markers[unknown==255] = 0
            markers = cv2.watershed(img,markers)
            img[markers == -1] = [255,0,0]

            im1 = cv2.cvtColor(img,cv2.COLOR_HSV2RGB)
            ShowImage('Watershed segmented image',im1,'gray')

            brain_mask = np.uint8(brain_mask)
            kernel = np.ones((8,8),np.uint8)
            closing = cv2.morphologyEx(brain_mask, cv2.MORPH_CLOSE, kernel)

            # Plot the data for Proper Visual Representation 
            plt.subplot(1, 1, 1) 
              

              
            plt.scatter(X[:, 0], X[:, 1])
            plt.xlabel('length') 
            plt.ylabel('width') 

            plt.title('SVM with linear kernel') 
              
            # Output the Plot 
            plt.show() 
            

    def preproc(self):
        global fname
        if fname=="":
            self.errimg=errimg()
            self.errimg.show()
        else:
            filename = fname
            print "file for processing",filename
            image =cv2.imread(str(filename))
            #print type(image)
            cv2.imshow("Original Image", image)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cv2.imshow("1 - Grayscale Conversion", gray)
            gray = cv2.bilateralFilter(gray, 11, 17, 17)
            cv2.imshow("2 - Bilateral Filter", gray)
            edged = cv2.Canny(gray, 27, 40)
            cv2.imshow("4 - Canny Edges", edged)

   
            
    def ex(self):
        sys.exit()
        

class errlog(QtGui.QMainWindow, error_log.Ui_Error):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.ok1)
    def ok1(self):
        self.hide()

class errimg(QtGui.QMainWindow, err_img.Ui_Error):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.ok1)
    def ok1(self):
        self.hide()



def main():
    app = QtGui.QApplication(sys.argv)  
    form = Login()                 
    form.show()                         
    app.exec_()                         


if __name__ == '__main__':              
    main()                             
