import cv2
import os
from cv2 import imshow
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
import math
import argparse
import timeit

from sqlalchemy import VARBINARY
from utils import *

def imRead(path = '../data/1tagvideo.mp4'):
    cap = cv2.VideoCapture(path)    
    ret, frame = cap.read(20)
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
        
    read_image = imRead(VideoPath)


    image = cv2.cvtColor(read_image, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image,(3,3),0.2)#sigmaX=0.1,sigmaY=0.1)

    #Using DFT from cv2 library
    dft = cv2.dft(np.float32(image),flags = cv2.DFT_COMPLEX_OUTPUT) #dft is h,w,2 

    #Using FFT from np library
    fft = np.fft.fft2(image) # fft is hxwhx1 - imaginary component is added with real component

    #Bring the low frequency to center
    dft_shift = np.fft.fftshift(dft)
    fft_shift = np.fft.fftshift(fft)
    # print(dft_shift.shape)

    # find the magnitude spectrum to visualise it.
    magnitude_spectrum_dft = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
    magnitude_spectrum_fft = 20*np.log(fft_shift)

    #choose fft or dft
    spectrum = dft_shift
    magnitude_spectrum = magnitude_spectrum_dft

    # cv2.imshow("FFT Magnitude Spectrum", np.uint8(magnitude_spectrum))
    # cv2.imshow("DFT Magnitude Spectrum",  np.uint8(magnitude_spectrum_dft))

    #Cut a hole at the center of frequency specturm, AKA Make a High pass filter mask
    mask = np.ones(spectrum.shape,dtype=np.float32)
    
    #center of mask 
    centre = (mask.shape[0]//2,mask.shape[1]//2)
    radius = 10

    #Fill the circle with zeros
    mask_value = 0
    thickness = -1
    cv2.circle(mask, centre, radius, mask_value, thickness)
    mask = mask/255.0

    #Filter the image spectrum
    filtered_spectrum = spectrum*mask

    magnitude_spectrum_hp = 20*np.log(cv2.magnitude(filtered_spectrum[:,:,0],filtered_spectrum[:,:,1]))

    ifft = np.fft.ifftshift(filtered_spectrum)
    inverse_fft = cv2.idft(ifft)
    print(inverse_fft.shape)
    filltered_image = cv2.magnitude(inverse_fft[:,:,0], inverse_fft[:,:,1])
    filltered_image = cv2.normalize(filltered_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # perform thresholding and obtain binary image
    _, binary = cv2.threshold(filltered_image, 20, 255, cv2.THRESH_BINARY) 
    kernel = np.ones((3,3),np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    cv2.imshow("Binary image ",filltered_image)

    # find the contour of the largest area in the image and extract the region where White space of Tag is present. 
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print("Number of contours found: ", len(contours))
    contours_img = cv2.cvtColor(binary,cv2.COLOR_GRAY2RGB)
    for i in range(len(contours)):
        cv2.drawContours(contours_img, contours, i, (255*(1-i/len(contours)),i*255/len(contours),0), 1)
    cv2.imshow("Locate AR TAG - Contours", contours_img)

    Area_List = []
    for c in contours:
        Area_List.append(cv2.contourArea(c))
    i = np.argmax(np.array(Area_List))
    rect = cv2.boundingRect(contours[i])
    print("Area List: ", i)
    x,y,w,h = rect
    margin = 20
    # get the AR Tag with white space with 10pixel pad margin
    AR_tag = read_image[y-margin:y+h+margin,x-margin:x+w+margin]  
    
    # find the contour of the AR block inside the Tag with  white space 
    _, binary = cv2.threshold(cv2.cvtColor(AR_tag, cv2.COLOR_BGR2GRAY), 240, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_AR_code_img = cv2.cvtColor(binary,cv2.COLOR_GRAY2RGB)
    for i in range(len(contours)):
        cv2.drawContours(contours_AR_code_img, contours, i, (255*(1-i/len(contours)),i*255/len(contours),0), 1)
    cv2.imshow("Contours of AR code", contours_AR_code_img)

    print("Number of contours found: ", len(contours))

    AR_contour = contours[-1] # get the AR's contour
    # print("Contours detected are: ",contours)
    # get the corresponding corner.
    AR_corners = cv2.approxPolyDP(AR_contour,0.07*cv2.arcLength(AR_contour,True),True)
    # AR_corners = cv2.goodFeaturesToTrack(binary, 4, 0.5, 50)
    AR_corners_img = cv2.cvtColor(binary,cv2.COLOR_GRAY2RGB)
    for corner in AR_corners:
        x,y = corner.ravel()
        cv2.circle(AR_corners_img,(int(x),int(y)),5,(36,255,12),-1)

    print("AR Corners: ",AR_corners)
    cv2.drawContours(AR_corners_img, AR_corners, -1, (0, 0, 255), 3)
    cv2.imshow("Corners from image",AR_corners_img)
    size = 10  #read_image.shape[0]
    
    destination_points = np.array([[0,0],[size,0],[size,size],[0,size]]).reshape(-1,1,2)
    H = computeHomography(np.float32(AR_corners), np.float32(destination_points))
    
    # warp the AR from the image plane to custom square plane like Bird's eye view representation. 
    imOut = np.zeros((size,size,3),dtype = np.uint8)
    AR_Tag_focused = warp(AR_tag, H, imOut)
    
    #Enlarge image for viewing 
    AR_Tag_focused  = cv2.resize(AR_Tag_focused, (128,128)) 
    
    AR_Tag_focused = AR_Tag_focused[margin:-margin,margin:-margin]
    
    cv2.imshow("Warped Tag", AR_Tag_focused)

    ## Print the images:
    fig,plts = plt.subplots(2,3,figsize = (15,10))
    plts[0][0].imshow(read_image,'gray')
    plts[0][0].axis('off')
    plts[0][0].title.set_text('a) Input image')
    
    plts[0][1].imshow(magnitude_spectrum,'gray')
    plts[0][1].axis('off')
    plts[0][1].title.set_text('b) FFT magnitude')

    plts[0][2].imshow(magnitude_spectrum_hp,'gray')
    plts[0][2].axis('off')
    plts[0][2].title.set_text('c) FFT magnitude after applying High pass filter ')

    plts[1][0].imshow(filltered_image,'gray')
    plts[1][0].axis('off')
    plts[1][0].title.set_text('d) Detected Edges')
    
    plts[1][1].imshow(AR_tag,'gray')
    plts[1][1].axis('off')
    plts[1][1].title.set_text('e) AR Tag located')
    
    plts[1][2].imshow(AR_Tag_focused)
    plts[1][2].axis('off')
    plts[1][2].title.set_text('f) AR Tag in Top down view ')
    
    if(not (os.path.isdir(SavePath))):
        os.makedirs(SavePath)
        
    fig.savefig(SavePath+'ARDetectionUsingFFt.png')
    print("Check ", SavePath," for Problem 1 Results")
   

    cv2.waitKey(5000)    
    
if __name__ == '__main__':
    main()
    
    