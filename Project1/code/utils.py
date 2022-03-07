from cmath import isnan
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.fft import fft, ifft
import math
import os
import string

from sqlalchemy import column
# import moviepy.editor as mpy

prev_AR_block = None



savePlotFFT = True
savePlotTestudo = True

def detectARTag(read_image):
    global savePlotFFT
    success = False
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
    radius = 30

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
    filltered_image = cv2.magnitude(inverse_fft[:,:,0], inverse_fft[:,:,1])
    filltered_image = cv2.normalize(filltered_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # perform thresholding and obtain binary image
    _, binary = cv2.threshold(filltered_image, 20, 255, cv2.THRESH_BINARY) 
    kernel = np.ones((3,3),np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    # cv2.imshow("Binary image ",filltered_image)

    # find the contour of the largest area in the image and extract the region where White space of Tag is present. 
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print("Number of contours found: ", len(contours))
    contours_img = cv2.cvtColor(binary,cv2.COLOR_GRAY2RGB)
    for i in range(len(contours)):
        cv2.drawContours(contours_img, contours, i, (255*(1-i/len(contours)),i*255/len(contours),0), 1)
    # cv2.imshow("Locate AR TAG - Contours", contours_img)

    Area_List = []
    if(len(contours)<3):
        print("[Info] Very few contours found")
    for c in contours:
        Area_List.append(cv2.contourArea(c))
    i = np.argmax(np.array(Area_List))
    rect = cv2.boundingRect(contours[i])
    x,y,w,h = rect
    margin = 20

    # get the AR Tag with white space with some padding
    AR_sheet = read_image[y-margin:y+h+margin,x-margin:x+w+margin]
    # cv2.imshow("AR Tag Located", AR_sheet)

    # find the contour of the AR block inside the Tag with  white space 
    bgr2gray = cv2.cvtColor(AR_sheet, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(bgr2gray, 240, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print("Number of contours found: ", len(contours))

    contours_AR_code_img = cv2.cvtColor(binary,cv2.COLOR_GRAY2RGB)
    for i in range(len(contours)):
        cv2.drawContours(contours_AR_code_img, contours, i, (255*(1-i/len(contours)),i*255/len(contours),0), 2+i)
    # cv2.imshow("Contours of AR code", contours_AR_code_img)

    #Get AR Tags black border, -1 index has the corners of A4 size sheet paper. 
    AR_contour = contours[-2] 
    # print("Contours detected are: ",contours)
    # get the corresponding corner.
    AR_corners = cv2.approxPolyDP(AR_contour,0.07*cv2.arcLength(AR_contour,True),True)
    # AR_corners = cv2.goodFeaturesToTrack(binary, 4, 0.5, 50)
    AR_corners_img = cv2.cvtColor(binary,cv2.COLOR_GRAY2RGB)
        
    #4 Corner points for identifying 4 unique corner points: Dark Red, Blue, Green, Yellow
    cornerpts_color_map=[(86, 50, 168), (15, 165, 219), (29, 219, 15), (219, 216, 15)]
    count=0
    for corner in AR_corners:
        x,y = corner.ravel()
        cv2.circle(AR_corners_img,(int(x),int(y)),6,cornerpts_color_map[count],5)
        count+=1
        if count>=4:
            # print("[Info] More than 4 corner points detected!!")
            break
    if count<3:
        print("[Error] Less than 4 corner points detected!!")
        return success, None, None
    AR_corners_margin = np.squeeze(AR_corners)
    AR_Tag = AR_sheet[AR_corners_margin[0][1]:AR_corners_margin[2][1], AR_corners_margin[0][0]:AR_corners_margin[2][0]]

    # print("AR Corners coordinates: ",AR_corners)
    cv2.drawContours(AR_corners_img, AR_corners, -1, (0, 0, 255), 3)
    # cv2.imshow("Corners of AR Code",AR_corners_img)
    
    # Compute Homography
    size = 128  #read_image.shape[0]
    destination_points = np.array([[0,0],[size,0],[size,size],[0,size]]).reshape(-1,1,2)
    AR_corners = np.squeeze(np.float32(AR_corners))
    AR_corners[:,0]+=y-margin
    AR_corners[:,1]+=x-margin
    # print(AR_corners)
    # print(AR_corners[:,1])
    H = computeHomography(AR_corners, np.float32(destination_points))
    # H,mask = cv2.findHomography(np.float32(AR_corners), np.float32(destination_points))

    
    # warp the AR from the image plane to custom square plane like Bird's eye view representation. 
    imOut = np.zeros((size,size,3),dtype = np.uint8)
    AR_Tag_focused = warp(read_image, H, imOut)
    # AR_Tag_focused = cv2.warpPerspective(AR_tag, H,(50, 50))
    
    #Enlarge image for viewing 
    AR_Tag_focused  = cv2.resize(AR_Tag_focused, (128,128)) 
        
    # cv2.imshow("Warped Tag", AR_Tag_focused)
    # cv2.waitKey(1000)    

    ## Print the images:
    if savePlotFFT:
        plots = Plot(3,3)
        plots.set(read_image,"Input image")
        plots.set(magnitude_spectrum,"FFT magnitude")
        plots.set(magnitude_spectrum_hp,'FFT magnitude after applying \nHigh pass filter ')
        plots.set(filltered_image,'Detected Edges')
        plots.set(cv2.resize(contours_img,(200,200)), "Contours detected in the whole image")
        plots.set(contours_AR_code_img, "Contours of AR code,\nprocessed from (d)")
        plots.set( cv2.cvtColor(AR_corners_img,cv2.COLOR_BGR2RGB), "4 Corners of AR code")
        plots.set(AR_Tag,'AR Tag in image space')
        plots.set(AR_Tag_focused,'AR Tag after \nHomography Transformation')   
        plots.save('../outputs/Q1/','ARDetectionUsingFFt.png')
        savePlotFFT=False
    success = True
    return success, H, AR_Tag_focused



def projectTestudo(im_org,testudoBlock, H, orientation, decodedValue):
    global savePlotTestudo
    size = 128
    H_inv=np.linalg.inv(H)
    if np.isnan(H_inv).any():
        return im_org
    warpedTestudo = warp(testudoBlock,H_inv, np.zeros_like(im_org))      # custom Warp
    _, warpedTestudo_mask = cv2.threshold(cv2.cvtColor(warpedTestudo, cv2.COLOR_BGR2GRAY), 20, 255, cv2.THRESH_BINARY_INV)
    imOut = im_org.copy()
    warpedTestudo_mask = np.dstack((warpedTestudo_mask,warpedTestudo_mask,warpedTestudo_mask))
    imOut = cv2.bitwise_and(imOut, warpedTestudo_mask)
    imOut = cv2.addWeighted(imOut, 1.0, warpedTestudo, 1.0, 0)
    text_location = (10,10)
    text = "Tag: "+str(round(decodedValue,2))
    imOut = cv2.putText(imOut, text, text_location, cv2.FONT_HERSHEY_SIMPLEX, 1, (120, 120, 120) ,3, cv2.LINE_AA, False) 
                
    if savePlotTestudo:
        plot = Plot(2,1)
        plot.set(im_org,"Original image") 
        plot.set(imOut,"Testudo Image Projection\n on AR Tag") 
        plot.save("../outputs/Q2/","AR_Testudo_Projection.png")
        savePlotTestudo=False
    
    return imOut


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

def imRead(path = '../data/1tagvideo.mp4',i=0):
    cap = cv2.VideoCapture(path)    
    ret, frame = cap.read(i)
    if ret:
        frame  = cv2.resize(frame, (512,512))
    cap.release()   
    return frame

def videoRead(path):
    imgs = []
    cap = cv2.VideoCapture(path)
    
    while(True):
        ret, frame = cap.read()
        if ret:
            frame  = cv2.resize(frame, (800,640))
            imgs.append(frame)
        else:
            break
    cap.release()
    print("Number of frames in the video",len(imgs))
    return imgs

def warp(src, H , dst):
    """ To perform warping,
    Apply H on every pixel location from src (a,b ,c=1) to find destination location x,y,z
    since z doesnt exist, do x = x/z, y = y/z , so z = 1
    
    if x and y locations are within the boundaries of dst image shape,
    paste the value from src [a,b] at dst [x,y]
    """
    # im=src
    im = cv2.transpose(src)
    height,width = im.shape[:2]
    h_limit, w_limit = dst.shape[:2]
    
    for a in range(height-1):
        for b in range(width-1):
            ab1 = np.array([a ,b, 1])
            x,y,z = H.dot(ab1)
            x,y = int(x/z), int(y/z)
            if (x >= 0 and x < w_limit) and (y >= 0 and y < h_limit) :
                    val = im[a ,b]
                    dst[int(y),int(x)] = val
    
    return dst

def computeHomography(pts1,pts2):
    """ Compute Homography matrix:
    1) Compute the A matrix with 8x9 dimensions. Need to solve for Ax = 0 
    2) perform Singular Value decomposition on A matrix to find the solution x 
    3) reshape x into a 3x3 homography matrix
    """
    pts1,pts2 = pts1.squeeze(),pts2.squeeze()
    
    X,Y = pts1[:,0],pts1[:,1]
    Xp,Yp = pts2[:,0],pts2[:,1]

    startFlag=1

    for (x,y,xp,yp) in zip(X,Y,Xp,Yp):

        if (startFlag == 1) :
            A = np.array([[-x,-y,-1,0,0,0, x*xp, y*xp,xp], [0,0,0,-x,-y,-1, x*yp, y*yp, yp]])
        else:
            tmp = np.array([[-x,-y,-1,0,0,0, x*xp, y*xp,xp], [0,0,0,-x,-y,-1, x*yp, y*yp, yp]])
            A = np.vstack((A, tmp))

        startFlag+=1    

    U,S,Vt = np.linalg.svd(A.astype(np.float32))

    H_ = Vt[8,:]/Vt[8][8]
    H_ = H_.reshape(3,3)
    
    return H_

def flip(im):
    return cv2.flip(im,-1)


def crop_AR(AR_block):
    """
    For a given AR tag, crop the black region
    
    """
    global prev_AR_block
    Xdistribution = np.sum(AR_block,axis=0)
    Ydistribution = np.sum(AR_block,axis=1)
    
    mdpt = len(Xdistribution)//2
    left_Xdistribution = Xdistribution[:mdpt]
    right_Xdistribution = Xdistribution[mdpt:]
    
    leftx,rightx,topx,topy = -1,-1,-1,-1
    for i in range(len(left_Xdistribution)):
        if left_Xdistribution[i] > 2000:
            leftx = i
            break

    for i in range(len(right_Xdistribution)):
        if right_Xdistribution[i] < 2000:
            rightx = i
            rightx+=mdpt
            break
    

    top_Ydistribution = Ydistribution[:mdpt]
    bottom_Ydistribution = Ydistribution[mdpt:]

    for i in range(len(top_Ydistribution)):
        if top_Ydistribution[i] > 2000:
            topy = i
            break

    for i in range(len(bottom_Ydistribution)):
        if bottom_Ydistribution[i] < 2000:
            bottomy = i
            bottomy+=mdpt
            break

    cropped_AR_block = AR_block[topy:bottomy,leftx:rightx]
    
    if (leftx < 0 )or(rightx < 0)or(topy < 0 )or(bottomy < 0):
        cropped_AR_block  = prev_AR_block
        print('bad tag found')
    else:
        prev_AR_block = cropped_AR_block        
        
    return cropped_AR_block

def getOrientation(AR_block):
    margin=10
    AR_block = AR_block[margin:-margin,margin:-margin]
    _, AR_block = cv2.threshold(cv2.cvtColor(AR_block, cv2.COLOR_BGR2GRAY), 240, 255, cv2.THRESH_BINARY) # only threshold
    cropped_AR_block = crop_AR(AR_block)
    cropped_AR_block  = cv2.resize(cropped_AR_block, (64,64))

    lowerright = cropped_AR_block[48:64,48:64]
    lowerleft = cropped_AR_block[48:64,0:16]

    upperright = cropped_AR_block[0:16,48:64]
    upperleft = cropped_AR_block[0:16,0:16]

    UL,UR,LL,LR = np.int(np.median(upperleft)), np.int(np.median(upperright)), np.int(np.median(lowerleft)), np.int(np.median(lowerright))

    AR_orientationPattern = [UL,UR,LL,LR]
    orientations = [180,-90,90,0]

    index = np.argmax(AR_orientationPattern)

    orientation = orientations[index]
    decode=True
    if decode ==  True:
        rotated_AR_block = RotatebyOrientation(cropped_AR_block, orientation)
        
        block1 = rotated_AR_block[16:32,16:32]
        block2 = rotated_AR_block[16:32,32:48]
        block3 = rotated_AR_block[32:48,32:48]
        block4 = rotated_AR_block[32:48, 16:32]

        bit1 = np.median(block1)/255
        bit2 = np.median(block2)/255
        bit3 = np.median(block3)/255
        bit4 = np.median(block4)/255

        # print("Bit Value: ",bit1,bit2,bit3,bit4 )
        decodedValue = bit1*1 + bit2*2 + bit3*4 + bit4*8

    else:
        decodedValue = None
    return orientation, decodedValue, rotated_AR_block

def RotatebyOrientation(Block, orientation):
    
    # rotateBlock by orientation degree
    if orientation == 90:
        #print("Rotated anticlckwise 90")
        Block = cv2.rotate(Block, cv2.cv2.ROTATE_90_CLOCKWISE) 

    elif orientation == -90:
        # print("Rotated clckwise 90")
        Block = cv2.rotate(Block, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)

    elif orientation == 180:
        # print("Rotated 180")
        Block = cv2.rotate(Block, cv2.cv2.ROTATE_180) 
    return Block



## global variables
prev_AR_contours = None

K = np.array([[1346.100595, 0, 932.1633975],
            [0.0000000000, 1355.933136, 654.8986796],
            [0.0000000000, 0.0000000000,1.0000000000]])

def ProjectionMatrix(H):
    global K
    h1, h2, h3 = H[:,0], H[:,1], H[:,2]
    K_inv = np.linalg.inv(K) 
    lamda = 2/(np.linalg.norm(K_inv.dot(h1)) + np.linalg.norm(K_inv.dot(h2)) )
    
    B_ = lamda*K_inv.dot(H)

    if np.linalg.det(B_) > 0 :
        B = B_
    else:
        B = - B_

    r1, r2, r3 = B[:,0], B[:,1], np.cross(B[:,0], B[:,1])
    t = B[:,2]

    RTmatrix = np.dstack((r1,r2,r3,t)).squeeze()
    P = K.dot(RTmatrix)
    return P


def getCubeCoordinates(P,cube_size = 128):

    x1,y1,z1 = P.dot([0,0,0,1])
    x2,y2,z2 = P.dot([0,cube_size,0,1])
    x3,y3,z3 = P.dot([cube_size,0,0,1])
    x4,y4,z4 = P.dot([cube_size,cube_size,0,1])

    x5,y5,z5 = P.dot([0,0,-cube_size,1])
    x6,y6,z6 = P.dot([0,cube_size,-cube_size,1])
    x7,y7,z7 = P.dot([cube_size,0,-cube_size,1])
    x8,y8,z8 = P.dot([cube_size,cube_size,-cube_size,1])

    X = [x1/z1 ,x2/z2 ,x3/z3 ,x4/z4 ,x5/z5 ,x6/z6 ,x7/z7 ,x8/z8] 
    Y = [y1/z1 ,y2/z2 ,y3/z3 ,y4/z4 ,y5/z5 ,y6/z6 ,y7/z7 ,y8/z8] 
    XY = np.dstack((X,Y)).squeeze().astype(np.int32)
    
    return XY

def processAR(im_org,H,size = 128):
    global prev_AR_contours

    # find the projection Matrix
    P = ProjectionMatrix(np.linalg.inv(H))
    # find cube coordinates
    XY = getCubeCoordinates(P,cube_size = size)
    # draw the cube
    imOut = drawCube(im_org, XY)

    success = True

    return success, imOut


def ProcessVideo(Videopath = '../data/1tagvideo.mp4'):
    imgs = []
    cap = cv2.VideoCapture(Videopath)
    outputs = []
    startFlag = True
    counter = 0
    successcounter=0
    plots = Plot(2,2)
    while(True):
        ret, frame = cap.read()
        if ret:
            im_org  = cv2.resize(frame, (800,640))
            success, imOut = processAR(im_org,size = 128)
            outputs.append(imOut)
            counter +=1
            successcounter += 1 if success else 0
            if counter%40 == 0 and success and plots.i<2 and plots.j<2:
                plots.set(cv2.cvtColor(imOut,cv2.COLOR_BGR2RGB)," ")
                               
        else:
            print("Done Processing.....")
            print("Projected on ",round((successcounter/counter)*100, 2),"% of the video")
            break        
    cap.release()
    plots.save("../outputs/Q2/",'AR_Cube_Projection.png')
    return outputs




def getCorners(c):
    
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    return np.array([[np.array(extLeft), np.array(extRight) , np.array(extTop) , np.array(extBot)]]).reshape(-1,1,2)


def ProjectionMatrix(H):
    global K
    h1, h2, h3 = H[:,0], H[:,1], H[:,2]
    K_inv = np.linalg.inv(K) 
    lamda = ((np.linalg.norm(K_inv.dot(h1)) + np.linalg.norm(K_inv.dot(h2)) )/2)**(-1)
    B_ = lamda*K_inv.dot(H)

    if np.linalg.det(B_) > 0 :
        B = B_
    else:
        B = - B_

    r1, r2, r3 = B[:,0], B[:,1], np.cross(B[:,0], B[:,1])
    t = B[:,2]

    RTmatrix = np.dstack((r1,r2,r3,t)).squeeze()
    P = K.dot(RTmatrix)
    return P


def getCubeCoordinates(P,cube_size = 128):

    x1,y1,z1 = P.dot([0,0,0,1])
    x2,y2,z2 = P.dot([0,cube_size,0,1])
    x3,y3,z3 = P.dot([cube_size,0,0,1])
    x4,y4,z4 = P.dot([cube_size,cube_size,0,1])

    x5,y5,z5 = P.dot([0,0,-cube_size,1])
    x6,y6,z6 = P.dot([0,cube_size,-cube_size,1])
    x7,y7,z7 = P.dot([cube_size,0,-cube_size,1])
    x8,y8,z8 = P.dot([cube_size,cube_size,-cube_size,1])

    X = [x1/z1 ,x2/z2 ,x3/z3 ,x4/z4 ,x5/z5 ,x6/z6 ,x7/z7 ,x8/z8] 
    Y = [y1/z1 ,y2/z2 ,y3/z3 ,y4/z4 ,y5/z5 ,y6/z6 ,y7/z7 ,y8/z8] 
    XY = np.dstack((X,Y)).squeeze().astype(np.int32)
    
    return XY

def drawCube(im_org, XY):
    im_print = im_org
    for xy_pts in XY:
        x,y = xy_pts
        cv2.circle(im_print,(x,y), 5, (0,0,255), -1)
    thickness = 15
    linetype = cv2.LINE_8
    im_print = cv2.line(im_print,tuple(XY[0]),tuple(XY[1]), (0,255,10), thickness, linetype)
    im_print = cv2.line(im_print,tuple(XY[0]),tuple(XY[2]), (0,255,10), thickness, linetype)
    im_print = cv2.line(im_print,tuple(XY[0]),tuple(XY[4]), (0,255,10), thickness, linetype)
    im_print = cv2.line(im_print,tuple(XY[1]),tuple(XY[3]), (0,225,10), thickness, linetype)
    im_print = cv2.line(im_print,tuple(XY[1]),tuple(XY[5]), (0,225,10), thickness, linetype)
    im_print = cv2.line(im_print,tuple(XY[2]),tuple(XY[6]), (0,200,10), thickness, linetype)
    im_print = cv2.line(im_print,tuple(XY[2]),tuple(XY[3]), (0,200,10), thickness, linetype)
    im_print = cv2.line(im_print,tuple(XY[3]),tuple(XY[7]), (0,175,10), thickness, linetype)
    im_print = cv2.line(im_print,tuple(XY[4]),tuple(XY[5]), (0,150,10), thickness, linetype)
    im_print = cv2.line(im_print,tuple(XY[4]),tuple(XY[6]), (0,150,10), thickness, linetype)
    im_print = cv2.line(im_print,tuple(XY[5]),tuple(XY[7]), (0,125,10), thickness, linetype)
    im_print = cv2.line(im_print,tuple(XY[6]),tuple(XY[7]), (0,100,10), thickness, linetype)

    return im_print
