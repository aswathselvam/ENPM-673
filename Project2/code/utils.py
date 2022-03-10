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
    def set(self,x,y,image,title=None):
        self.images[(x,y)] = image.copy()
        self.titles[x,y] = title

    def generate(self):
        finalimage=[]
        for row in range(self.rows):
            colimages=[]
            for col in range(self.columns):
                image=self.images[(row,col)]
                col_img = cv2.resize(image, (image.shape[1]//(self.columns+1),image.shape[0]//(self.rows+1)) )
                if len(colimages)<1:
                    colimages = col_img
                else:
                    colimages = np.hstack((colimages, col_img))
            if len(finalimage)<1:
                finalimage = colimages
            else:
                finalimage = np.vstack((finalimage, colimages))
                
        self.gridimage=finalimage
        cv2.imshow('Final Image', finalimage)
        cv2.waitKey(0)
        return
