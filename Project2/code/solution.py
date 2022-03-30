import cv2
import os
from cv2 import VideoWriter
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
    Parser.add_argument('--problem', default='2', help='The problem numer ex: 1, 2 or 3, default is 1')
    
    args = Parser.parse_args()
    data_histogramEQ_path = args.data_histogramEQ
    video_whiteline_path = args.video_whiteline
    video_challenge_path = args.video_challenge
    savepath = args.savepath
    problem = int(args.problem)

    if(not (os.path.isdir(savepath))):
        os.makedirs(savepath)

    lanePredictor = LanePredictor()

    if problem==1:
        images_histogramEQ = []
        for image_file in sorted(glob.glob(data_histogramEQ_path+"*.png")):
            image = cv2.imread(image_file)
            image = cv2.resize(image, (image.shape[1]//2,image.shape[0]//2 ) )
            # cv2.imshow("Histogram equalization data", image)
            # cv2.waitKey(100)
            images_histogramEQ.append(image)

    #---------Problem 1: Histogram equalization-------------#
        BOTH=True
        outimage = lanePredictor.histogramEqualization([images_histogramEQ[0]], BOTH=BOTH)
        histogramWriter = cv2.VideoWriter(savepath+"histogram_and_adaptive.mp4",cv2.VideoWriter_fourcc(*'mp4v'), 5, (outimage.shape[1],outimage.shape[0]))
        for image in images_histogramEQ:
            outimage = lanePredictor.histogramEqualization([image], BOTH=BOTH)
            histogramWriter.write(outimage)
        histogramWriter.release()
        cv2.imwrite(savepath+"histogram_and_adaptive.jpg",outimage)

        return

    #-----------Problem 2: Straight Lane Detection----------#
    if problem==2:
        cap = cv2.VideoCapture(video_whiteline_path)
        ret, frame=cap.read()
        out=lanePredictor.detectStraightLane(cv2.resize(frame, (800,512)))

        #Create a new instance of lanePredictor for detecting flipped image.
        flippedLanePredictor = LanePredictor()
        videoWriter=None

        imagegrid=ImageGrid(2,3)
        while(True):
            ret, frame = cap.read()
            if ret:
                frame  = cv2.resize(frame, (800,512))
                out,hist=lanePredictor.detectStraightLane(frame)
                outflipped,histflipped=flippedLanePredictor.detectStraightLane(cv2.flip(frame, 1))
                imagegrid.set(0,0,frame,"Input image")
                imagegrid.set(0,1,out,"Lane-Type detection")
                imagegrid.set(1,1,hist,"")
                imagegrid.set(0,2,outflipped,"Vertically flipped")
                imagegrid.set(1,2,histflipped,"")
                # imagegrid.set(1,2,imgf,"Lane-Type histogram")

                grid_img = imagegrid.generate(scale=400)
                if isinstance(videoWriter, type(None)):
                    videoWriter = cv2.VideoWriter(savepath+'whiteline.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 15, (grid_img.shape[1],grid_img.shape[0]))  
                videoWriter.write(grid_img)

                cv2.imshow('grid_img',  grid_img)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else: 
                videoWriter.release()
                break

        return 

    #---------------Problem 3: Predict Turn--------------#
    if problem==3:
        cap = cv2.VideoCapture(video_challenge_path)
        ret, frame = cap.read()
        videoWriter=None
        imagegrid=ImageGrid(2,3)
        while(True):
            ret, frame = cap.read()
            if not ret:
                videoWriter.release()
                break  
            
            frame  = cv2.resize(frame, (800,512))
            # lanePredictor.calibrateColor(frame)
            if isinstance(lanePredictor.H, type(None)):
                detections_img = lanePredictor.getHomography(frame)
                continue
            lanemask,lanemasktd,lanemasktd_marking,back_proj_img,roc_img  = lanePredictor.detectwithTopDownView(frame)
            imagegrid.set(0,0,frame,"1) Input image")
            imagegrid.set(0,1,lanemask,"2) Lanemask")
            imagegrid.set(0,2,lanemasktd,"3)Top down")
            imagegrid.set(1,0,lanemasktd_marking,"4) Top down marking")
            imagegrid.set(1,1,back_proj_img,"5) Back projection")
            imagegrid.set(1,2,roc_img,"6) Radius of Curvature")

            grid_img = imagegrid.generate(scale=400)
            if isinstance(videoWriter, type(None)):
                videoWriter = cv2.VideoWriter(savepath+'challenge.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 10, (grid_img.shape[1],grid_img.shape[0]))  
            videoWriter.write(grid_img)
            cv2.imshow("detections_img", detections_img)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                videoWriter.release()
                break
            
        return
    # cap.release()

if __name__ == '__main__':
    main()
    