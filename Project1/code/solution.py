import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
import argparse
from utils2 import *

def main():
    
    
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--videopath', default='../data/1tagvideo.mp4', help='Video path , Default:../data/1tagvideo.mp4')
    Parser.add_argument('--savepath', default='../outputs/Q1/', help='Template Path , Default: ../outputs/template.png')
    
    args = Parser.parse_args()
    VideoPath = args.videopath
    SavePath = args.savepath

    testudoPath = "../data/testudo.png" 

    if(not (os.path.isdir(SavePath))):
        os.makedirs(SavePath)

    out = cv2.VideoWriter('../outputs/Q2/'+'cube_output.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 15, (800,640))

    cap = cv2.VideoCapture(VideoPath)
    size=128
    testudoBlock = cv2.imread(testudoPath)
    testudoBlock  = cv2.resize(testudoBlock, (size,size))
    # testudoBlock = cv2.cvtColor(testudoBlock, cv2.COLOR_RGB2BGR)
    
    while(True):
        ret, frame = cap.read()
        if ret:
            frame  = cv2.resize(frame, (512,512))
            detectARTag(frame)
            cv2.waitKey(1)

            #-----------------Problem 1 A----------------#
            success, H, tag = detectARTag(frame)
            if success:
                cv2.imshow("Warped Tag", tag)
            else:
                continue 

            # #-------------Problem 1 B-----------------#
            orientation, decodedValue,_ = getOrientation(tag)

            # #-------------Problem 2 A-----------------#
            imOut=projectTestudo(frame, testudoBlock, H, orientation, decodedValue)
            cv2.imshow("Testudo Image Projection",imOut)

            # #-------------Problem 2 B-----------------#
            success, cubeAROut = processAR(frame,H,size)
            cv2.imshow("Cube AR",cubeAROut)
            out.write(cubeAROut)

        else:
            break
    cap.release()
    out.release()

if __name__ == '__main__':
    main()
    