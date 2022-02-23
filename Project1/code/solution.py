import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
import math
import argparse

def imRead(path = '../data/1tagvideo.mp4'):
    cap = cv2.VideoCapture(path)    
    ret, frame = cap.read()
    if ret:
        frame  = cv2.resize(frame, (512,512))
    cap.release()   
    return frame

def main():
    
    
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--videopath', default='../data/1tagvideo.mp4', help='Video path , Default:../data/1tagvideo.mp4')
    Parser.add_argument('--savepath', default='../outputs/Q1/', help='Template Path , Default: ../outputs/template.png')
    
    args = Parser.parse_args()
    VideoPath = args.videopath
    SavePath = args.savepath
    
    if(not (os.path.isdir(SavePath))):
        os.makedirs(SavePath)
        
    image = imRead(VideoPath)
 
    
if __name__ == '__main__':
    main()
    
    