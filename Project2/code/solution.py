import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
import argparse
from utils import *
from lanepredictor import LanePredictor
import glob

def main():
    
    
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--data_histogramEQ', default='../data/adaptive_hist_data/', help='Video path , Default:../data/adaptive_hist_data/')
    Parser.add_argument('--video_whiteline', default='../data/whiteline.mp4', help='Video path , Default:../data/whiteline.mp4')
    Parser.add_argument('--video_challenge', default='../data/challenge.mp4', help='Video path , Default:../data/challenge.mp4')
    Parser.add_argument('--savepath', default='../outputs/', help='Template Path , Default: ../outputs/')
    
    args = Parser.parse_args()
    data_histogramEQ_path = args.data_histogramEQ
    video_whiteline_path = args.video_whiteline
    video_challenge_path = args.video_challenge
    savepath = args.savepath

    if(not (os.path.isdir(savepath))):
        os.makedirs(savepath)
    
    data_histogramEQ = []
    for image_file in sorted(glob.glob(data_histogramEQ_path+"*.png")):
        image = cv2.imread(image_file)
        image = cv2.resize(image, (image.shape[1]//2,image.shape[0]//2 ) )
        # cv2.imshow("Histogram equalization data", image)
        # cv2.waitKey(100)
        data_histogramEQ.append(image)

    #---------Problem 1: Histogram equalization-------------#
    lanePredictor = LanePredictor()
    lanePredictor.histogramEqualization(data_histogramEQ, mode=lanePredictor.HISTOGRAM_EQUALIZATION)
    lanePredictor.histogramEqualization(data_histogramEQ, mode=lanePredictor.ADAPTIVE_HISTOGRAM_EQUALIZATION)

    out = cv2.VideoWriter('../outputs/'+'whiteline.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 15, (800,640))    
    cap = cv2.VideoCapture(video_whiteline_path)

    while(True):
        ret, frame = cap.read()
        if ret:
            #-----------Problem 2: Straight Lane Detection----------#
            frame  = cv2.resize(frame, (512,512))
            lanePredictor.detectStraightLane(frame)

        else:
            break


    out = cv2.VideoWriter('../outputs/'+'challenge.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 15, (800,640))    
    cap = cv2.VideoCapture(video_challenge_path)

    while(True):
        ret, frame = cap.read()
        if ret:
            #-----------Problem 2: Straight Lane Detection----------#
            frame  = cv2.resize(frame, (512,512))
            lanePredictor.detectCurvedLane(frame)

        else:
            break
    #---------------Problem 3: Predict Turn-----------------#
    
    
    cap.release()
    out.release()

if __name__ == '__main__':
    main()
    