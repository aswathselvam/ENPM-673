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
                    cv2.FONT_HERSHEY_COMPLEX,fontScale=0.71,color=(0,0,0),thickness=3,lineType=cv2.LINE_8 )
                cv2.putText(col_img,self.titles[row,col], (0,30), \
                    cv2.FONT_HERSHEY_COMPLEX,fontScale=0.7,color=(80,80,255),thickness=2,lineType=cv2.LINE_8 )
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

class Plotter():
    def __init__(self,title,x_label,y_label,x_lim,y_lim):
        self.fig, ax = plt.subplots()
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_xlim(0, x_lim)
        ax.set_ylim(0, y_lim)
        ax.legend()
        lw=3
        alpha = 0.5
        self.line1, = ax.plot(np.arange(100), '-b', lw=lw, alpha=alpha, label='line1')
        self.line2, = ax.plot(np.arange(100), 'oy', lw=lw, alpha=alpha, label='line2')
        plt.ion()
        plt.show()

    def plot(self, line, x,y):
        line.set_data(x,y)
        self.fig.canvas.draw()
