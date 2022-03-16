import matplotlib.pyplot as plt
import string
import cv2
import os
from google_drive_downloader import GoogleDriveDownloader as gdd
import numpy as np

ORB = 'orb'
SIFT = 'sift'

def getKeyPoints(img,TYPE):
    # https://docs.opencv.org/4.x/db/d27/tutorial_py_table_of_contents_feature2d.html
    if TYPE==ORB:
        # Initiate ORB detector
        orb = cv2.ORB_create()
        # find the keypoints with ORB
        kp = orb.detect(img,None)
        # compute the descriptors with ORB
        kp, des = orb.compute(img, kp)
        return kp, des

def trimImage(frame):
    #crop top
    if not np.sum(frame[0]):
        return trimImage(frame[1:])
    #crop top
    if not np.sum(frame[-1]):
        return trimImage(frame[:-2])
    #crop top
    if not np.sum(frame[:,0]):
        return trimImage(frame[:,1:])
    #crop top
    if not np.sum(frame[:,-1]):
        return trimImage(frame[:,:-2])
    return frame

class Plot:
    def __init__(self,rows,columns):
        self.i=0
        self.j=0
        self.rows=rows
        self.columns=columns
        self.fig, self.plts =  plt.subplots(rows,columns,figsize=(13,13),squeeze=False)
        alphabet_string = string.ascii_lowercase
        self.alphabet_list = list(alphabet_string)

    def set(self,image,title,cmap='gray'):
        self.plts[self.i][self.j].imshow(image,cmap=cmap)
        self.plts[self.i][self.j].axis('off')
        self.plts[self.i][self.j].title.set_text(self.alphabet_list[self.i*self.columns+self.j] + ") " + title)
        if(self.j<self.columns-1):
            self.j+=1
        elif(self.i<self.rows):
            self.i+=1
            self.j=0
        # print("Setting plot at x: "self.i, " y: ",self.j)

    def plot(self,image,title):
        self.plts[self.i][self.j].plot(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        # self.plts[self.i][self.j].axis('off')
        self.plts[self.i][self.j].title.set_text(title)
        if(self.j<self.columns-1):
            self.j+=1
        elif(self.i<self.rows):
            self.i+=1
            self.j=0

    def show(self):
        # self.plts.tight_layout()
        self.fig.show()

    def save(self,savepath,filename):
        if(not (os.path.isdir(savepath))):
            os.makedirs(savepath)
            
        self.fig.savefig(savepath+filename, bbox_inches='tight')
        print("Plots saved on ", savepath+filename)

def createFolder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def drive_downloader(file_id, image_file):
    if not os.path.isfile(image_file): 
        gdd.download_file_from_google_drive(file_id=file_id,
                                            dest_path=image_file,
                                            unzip=False)