from cmath import isnan
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.fft import fft, ifft
import math
import os
import string
# import moviepy.editor as mpy


class ImageGrid():
    def __init__(self,rows,columns):
        self.rows=rows
        self.columns=columns
        self.images={}
        self.titles={}
        pass

    def set(self,x,y,image,title=''):
        self.images[(x,y)] = image.copy()
        self.titles[x,y] = title

    def generate(self,scale=300):
        finalimage=[]
        for row in range(self.rows):
            colimages=[]
            for col in range(self.columns):
                image=self.images[(row,col)]
                aspect_ratio=image.shape[1]//image.shape[0] # Height/Width
                col_img = cv2.resize(image, (scale*aspect_ratio,scale))
                cv2.putText(col_img,self.titles[row,col], (0,30), \
                    cv2.FONT_HERSHEY_COMPLEX,fontScale=0.7,color=(80,100,255),thickness=2,lineType=cv2.LINE_8 )
                if len(colimages)<1:
                    colimages = col_img
                else:
                    colimages = np.hstack((colimages, col_img))
            if len(finalimage)<1:
                finalimage = colimages
            else:
                finalimage = np.vstack((finalimage, colimages))

        self.gridimage=finalimage
        return self.gridimage
