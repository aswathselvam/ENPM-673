from turtle import right
from attr import NOTHING
import cv2
import numpy as np
from sympy import continued_fraction
from utils import ImageGrid, Plotter
from preprocess import *
import math 
import matplotlib.pyplot as plt
from collections import defaultdict


from sympy import Matrix
from sympy import *


RED = (0,0,255)
GREEN = (0,255,0)
YELLOW = (0,255,255)
OLIVE = (0,128,128)
ORCHID = (204,50,153)
WHITE= (200,200,200)
class LanePredictor:

    def __init__(self,data=None,video=None):
        self.data=data
        self.video=video
        self.HISTOGRAM_EQUALIZATION = "Histogram Equalization"
        self.ADAPTIVE_HISTOGRAM_EQUALIZATION = "Adaptive Histogram Equalization"
        self.histogram_profile=[None, None, None]
        self.USE_HISTOGRAM_MEMORY=False
        self.histplotter = None
        self.lineposPlotter = None
        self.colorCalibrationInitialized = False
        self.setupLinevariabless()
        self.mvl = MovingAverage(10)
        self.mvr = MovingAverage(10)
        self.ld = defaultdict(list)
        self.rd = defaultdict(list)

    def histogram(self, im_flat):
        # RANGE=256
        # bins = 100
        # TARGET_BIN_RATIO = (bins//RANGE)
        bins = 256
        h = np.zeros(bins)
        for i in im_flat:
            h[i] +=1
        return h

    def cumulate_distribution(self, sequence):
        bins = []
        sum=0
        for i in range(0,len(sequence)):
            sum+=sequence[i]
            bins.append(sum)
        return np.array(bins)
    
    def normalize(self, c):
        c = np.array(c)
        return ((c - c.min()) * 255) / (c.max() - c.min())
        

    def adjustGamma(self, image, gamma=2.0):
        """
        https://stackoverflow.com/questions/33322488/how-to-change-image-illumination-in-opencv-python
        
        """
        gamma_inv = 1.0 / gamma
        gammaTable = []
        for i in np.arange(0, 256):
            i = (i / 255.0) ** gamma_inv
            gammaTable.append(i*255)
        
        gammaTable = np.array(gammaTable).astype("uint8")
        
        # Apply gamma correction using the lookup table
        return cv2.LUT(image, gammaTable)

    def histogramEqualization(self,data,mode="Histogram Equalization",BOTH=True):
        """
        Computes Histogram Equalization
        """
        gridimages={}
        titles=["Original Image","Histogram Equalized", "Adaptive Equalization"]
        for image in data:
            # cv2.imshow("Original Image", (image))
            gridimages[titles[0]]=image

            if mode==self.HISTOGRAM_EQUALIZATION or BOTH:

                for i in range(image.shape[2]):
                    im_channel = image[:,:,i]
                    flat_image =  im_channel.flatten()
                    contrast_normalized  = None
                    if all([x is not None for x in self.histogram_profile]) and self.USE_HISTOGRAM_MEMORY:
                        contrast_normalized = self.histogram_profile[i]
                    else:
                        h = self.histogram(flat_image)
                        c = self.cumulate_distribution(h)
                        # c_norm = np.int32(cv2.normalize(c,None, 0,255,cv2.NORM_MINMAX))
                        contrast_normalized  = np.int32(self.normalize(c))
                        self.histogram_profile[i]=contrast_normalized                        

                    im_eq = contrast_normalized[flat_image]
                    im_eq = im_eq.reshape(-1,image.shape[1])
                    
                    if i==0:
                        im_eqs = np.array(im_eq)
                    else:
                        im_eqs = np.dstack((im_eqs,im_eq))

                # cv2.imshow("Equalilzed Image", np.uint8(im_eqs))
                # cv2.waitKey(50)
                gridimages[titles[1]]=np.uint8(im_eqs)

            if mode==self.ADAPTIVE_HISTOGRAM_EQUALIZATION or BOTH:
                im_eqs = self.adjustGamma(image, gamma = 2.0)
                # cv2.imshow("Adaptive Equalilzed Image", np.uint8(im_eqs))
                # cv2.waitKey(50)
                gridimages[titles[2]]=np.uint8(im_eqs)
            
            break

        imagegrid=ImageGrid(1,len(gridimages))
        counter=0
        for i in range(len(titles)):
            image = gridimages.get(titles[i])
            if image is not None:
                imagegrid.set(0,counter,image,titles[i])
                counter+=1
        return imagegrid.generate(scale=200)    

    def nothing(self,x):
        pass

    def createCalibrationSliders(self):
        cv2.namedWindow('image')

        # create trackbars for color change
        cv2.createTrackbar('L','image',0,255,self.nothing)
        cv2.createTrackbar('A','image',0,255,self.nothing)
        cv2.createTrackbar('B','image',0,255,self.nothing)
        cv2.createTrackbar('L_High','image',0,255,self.nothing)
        cv2.createTrackbar('A_High','image',0,255,self.nothing)
        cv2.createTrackbar('B_High','image',0,255,self.nothing)
        
        self.switch = '0 : RGB \n1 : LAB'
        cv2.createTrackbar(self.switch, 'image',0,1,self.nothing)

        self.colorCalibrationInitialized = True


    def calibrateColor(self, frame):
        
        if not self.colorCalibrationInitialized:
            self.createCalibrationSliders()

        # create switch for ON/OFF functionality
        original_frame=frame.copy()
        while True:
            switch = cv2.getTrackbarPos(self.switch,'image')

            if switch:
                frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2LAB)
            else:
                frame = original_frame
            
            # get current positions of four trackbars
            self.lpos = cv2.getTrackbarPos('L','image')
            self.apos = cv2.getTrackbarPos('A','image')
            self.bpos = cv2.getTrackbarPos('B','image')
            
            # get trackbar positions
            self.lposH = cv2.getTrackbarPos('L_High', 'image')
            self.aposH = cv2.getTrackbarPos('A_High', 'image')
            self.bposH = cv2.getTrackbarPos('B_High', 'image')

            # Reject colors outside the range of low and high sliders.
            lower_hsv = np.array([self.lpos, self.apos, self.bpos])
            higher_hsv = np.array([self.lposH, self.aposH, self.bposH])
            mask = cv2.inRange(frame, lower_hsv, higher_hsv)
            output_img = cv2.bitwise_and(frame, frame, mask=mask)

            # Try thresholding values
            L, a, b = cv2.split(frame)
            ret,thresh_l = cv2.threshold(L,self.lpos,255,cv2.THRESH_BINARY)
            ret,thresh_a = cv2.threshold(a,self.apos,255,cv2.THRESH_BINARY)
            ret,thresh_b = cv2.threshold(b,self.bpos,255,cv2.THRESH_BINARY)
            # ret,lab_thresh_img = cv2.threshold(lab_img[:-1],200,255,cv2.THRESH_BINARY)
            # output_img = cv2.merge([thresh_l,thresh_a,thresh_b])
            # ret,lab_thresh_img = cv2.threshold(lab_img[:-2],20,255,cv2.THRESH_BINARY)
            # lab_img[:-2] = lab_thresh_img
            # lab_img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2BGR)
            # lab_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # lab_img = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            if switch:
                output_img = cv2.cvtColor(output_img, cv2.COLOR_LAB2BGR)
            cv2.imshow("image", output_img)
            
            break

        return output_img

    def getLaneMask(self, frame):
        #From calibrated values: 

        #Use LAB Colorspace s=1, Use RGB colorspace s=0:
        s = 1

        if s:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

        lpos, apos, bpos, lposH, aposH, bposH = 215, 123, 119, 255, 130, 140

        lower_hsv = np.array([lpos, apos, bpos])
        higher_hsv = np.array([lposH, aposH, bposH])
        mask = cv2.inRange(frame, lower_hsv, higher_hsv)
        output_img = cv2.bitwise_and(frame, frame, mask=mask)

        if s:
            output_img = cv2.cvtColor(output_img, cv2.COLOR_LAB2BGR)
        
        ret,output_bw = cv2.threshold(cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY), 200, 255, cv2.THRESH_BINARY)
        return output_bw

    def getCurvedLaneMask(self,frame,lower_val=224):
        #From calibrated values: 

        if not self.colorCalibrationInitialized:
            self.lpos, self.apos, self.bpos, self.lposH, self.aposH, self.bposH = 0, 0, lower_val, 255, 255, 255
            
            #Use LAB Colorspace s=1, Use RGB colorspace s=0:
            self.switch = 0            
            

        if self.switch:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

        lower_hsv = np.array([self.lpos, self.apos, self.bpos])
        higher_hsv = np.array([self.lposH, self.aposH, self.bposH])
        mask = cv2.inRange(frame, lower_hsv, higher_hsv)
        output_img = cv2.bitwise_and(frame, frame, mask=mask)
        if self.switch:
            output_img = cv2.cvtColor(output_img, cv2.COLOR_LAB2BGR)

        output_bw = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)#cv2.threshold(, lower_val, 255, cv2.THRESH_BINARY)
        # cv2.imshow("image", output_bw)
        # cv2.waitKey(0)
        return output_bw


    def setupLinevariabless(self):
        self.max_line_length = -float('inf')
        self.long_line = None

        self.min_line_length = float('inf')
        self.short_line = None
        self.short_line_bin = None
        self.long_line_bin = None
        self.H=None

    def detectStraightLane(self, frame):
        """
        Detects straight Lanes in a given frame
        """

        lanemask = self.getLaneMask(frame)
        height = lanemask.shape[0]
        width = lanemask.shape[1]

        # Get white pixel distribution along the columns of the images
        distribution = np.sum(lanemask, axis=0)
        if not self.histplotter:
            self.histplotter = Plotter('Distribution of white pixels','columns of Image','Frequency',distribution.shape[0],np.max(distribution))
        self.histplotter.plot(self.histplotter.line1, np.arange(len(distribution)),distribution)

        #Get all the peaks:
        distribution = distribution
        peaks = np.argsort(distribution)
        indx = peaks[-len(peaks)//10:]
        peaks = distribution[indx] 
        # self.histplotter.plot(self.histplotter.line2, indx, peaks)
        
        min_mean_of_offset = float('inf')
        mean_offset_idx=0
        no_bins = 6
        bins = np.linspace(0,len(distribution),no_bins)
        bin_idx = np.digitize(indx, bins, right=False)
        self.histplotter.plot(self.histplotter.line2, bin_idx*width/no_bins, peaks)

        bin_idx_hist, bin_edges= np.histogram(bin_idx,bins)
        bin_idx = np.argsort(bin_idx_hist)
        highest_bin_idx = bin_idx[0]
        for bin in bin_idx:
            if bin != highest_bin_idx:
                second_highest_bin_idx = bin 
                break
        # print(highest_bin_idx,second_highest_bin_idx)

        # seperation = np.mean(indx)
        highest_bin_val = (highest_bin_idx+1)*width/no_bins
        second_highest_bin_val = (highest_bin_idx+1)*width/no_bins
        # self.histplotter.l2.set_xdata(second_highest_bin_val)

        
        lanemask = cv2.cvtColor(lanemask, cv2.COLOR_GRAY2BGR)
        dst = cv2.Canny(lanemask, 2, 100, None, 3)
    
        # Copy edges to the images that will display the results in BGR
        cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
        cdstP = np.copy(frame)
        
        lines = cv2.HoughLines(dst, 1, np.pi / 180, 150, None, 0, 0)
        
        if lines is not None:
            for i in range(0, len(lines)):
                rho = lines[i][0][0]
                theta = lines[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
                pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
                cv2.line(cdst, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
            
        linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 100, None, 50, 10)
        
        if linesP is not None:
            for i in range(0, len(linesP)):
                l = linesP[i][0]
                length = np.linalg.norm(l[:2]-l[2:])

                #Draw circle at start position of line:
                cv2.circle(cdstP, (l[0], l[1]), 5, YELLOW, 3, cv2.LINE_AA)

                #Draw circle at end position of line:
                cv2.circle(cdstP, (l[2], l[3]), 5, OLIVE, 3, cv2.LINE_AA)

                current_line_bin = round((np.digitize(l[0], bins, right=False)+np.digitize(l[2], bins, right=False))/2)

                if length > self.max_line_length:
                    self.long_line = l
                    self.max_line_length = length
                    self.long_line_bin = current_line_bin
                    self.histplotter.l1.set_xdata((self.long_line_bin)*width/no_bins)
                
                proposed_short_line_bin = current_line_bin
                if proposed_short_line_bin != self.long_line_bin:
                    self.short_line = l
                    self.min_line_length = length
                    self.short_line_bin = proposed_short_line_bin
                    # print(self.short_line_bin, self.long_line_bin)
                    self.histplotter.l2.set_xdata((self.short_line_bin)*width/no_bins)

                                
                if not isinstance(self.long_line, type(None)) and not isinstance(self.short_line, type(None)):
                    (leftline, rightline, color) =  (self.long_line, self.short_line, GREEN) if self.long_line_bin < self.short_line_bin else (self.short_line, self.long_line, RED)
                    
                    # Find the line intersections with the top and bottom of the image:
                    # x is opencv image columns, y is opencv image rows. 
                    # Line format [x1, y1, x2, y2]
                    m_ll = (leftline[3] - leftline[1])/(leftline[2] - leftline[0])
                    y_ll = leftline[1]
                    x_ll = leftline[0]
                    m_rl = (rightline[3] - rightline[1])/(rightline[2] - rightline[0])
                    y_rl = rightline[1]
                    x_rl = rightline[0]

                    # cv2.line(cdstP, (leftline[0], leftline[1]), (leftline[2], leftline[3]), RED, 3, cv2.LINE_AA)
                    # cv2.line(cdstP, (rightline[0], rightline[1]), (rightline[2], rightline[3]), GREEN, 3, cv2.LINE_AA)

                    y=0
                    top_left = (round((m_ll)*(y-y_ll)+x_ll), y)
                    top_right = (round((m_rl)*(y-y_rl)+x_rl), y)

                    y=frame.shape[0]
                    bottom_left = (round((1/m_ll)*(y-y_ll)+x_ll), y)
                    bottom_right = (round((1/m_rl)*(y-y_rl)+x_rl), y)
                    # print(tuple(top_left), top_right, bottom_left, bottom_right)

                    # Points chosen for Homography:
                    #Draw circle at start position of line:
                    # cv2.circle(cdstP, tuple(top_left), 20, YELLOW, 3, cv2.LINE_AA)

                    # # #Draw circle at end position of line:
                    # cv2.circle(cdstP, tuple(top_right), 20, OLIVE, 3, cv2.LINE_AA)
                    
                    # # #Draw circle at start position of line:
                    # cv2.circle(cdstP, bottom_left, 20, RED, 3, cv2.LINE_AA)

                    # # #Draw circle at end position of line:
                    # cv2.circle(cdstP, bottom_right, 20, GREEN, 3, cv2.LINE_AA)


                    # Find vanishing point:
                    x_intersection = (m_ll*x_ll - m_rl*x_rl + y_rl - y_ll)/(m_ll-m_rl) 
                    y_intersection = m_ll*(x_intersection - x_ll) + y_ll

                    x_intersection = round(x_intersection)
                    y_intersection = round(y_intersection)

                    cv2.line(cdstP, bottom_right, (x_intersection,y_intersection), color, 3, cv2.LINE_AA)
                    color = GREEN if color==RED else RED
                    cv2.line(cdstP, bottom_left, (x_intersection,y_intersection), color, 3, cv2.LINE_AA)


                    cv2.circle(cdstP, (x_intersection, y_intersection), 20, ORCHID, 10, cv2.LINE_AA)
                    # print(x_intersection,y_intersection)

                    offset = 20
                    src = np.array([[x_intersection-offset,y_intersection], [x_intersection+offset, y_intersection], [bottom_right[0], bottom_right[1]], [bottom_left[0], bottom_left[1]] ], dtype=np.float32)
                    # src = np.array([[355,310], [442, 310], [750, 510], [70, 510] ], dtype=np.float32)
                    # src = np.roll(src, 1)
                    tw = height
                    th = width
                    dst = np.array([[0,0],[tw,0],[tw,th],[0,th]],dtype=np.float32)

                    self.H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
                    # warped_img = cv2.warpPerspective(frame, self.H,(height,width))
                    # cv2.imshow("warped_img",warped_img)


                color = RED
                # print(l,highest_bin_val,(long_line_bin+1)*width/no_bins)
                if current_line_bin==self.short_line_bin:
                    # if l[0] < highest_bin_val or l[2] < highest_bin_val or l[1] < highest_bin_val:
                        color = GREEN

                cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), color, 3, cv2.LINE_AA)
                

        # cv2.imshow("Hough Lines", cdstP)

        # convert canvas to image
        img = np.fromstring(self.histplotter.fig.canvas.tostring_rgb(), dtype=np.uint8,
                sep='')
        img  = img.reshape(self.histplotter.fig.canvas.get_width_height()[::-1] + (3,))

        # img is rgb, convert to opencv's default bgr
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

        return cdstP, img

    def get_end_pnts(self, pnts, img):
        extremes = []    
        for p in pnts:
            x = p[0]
            y = p[1]
            n = 0
            try:       
                n += img[y - 1,x]
                n += img[y - 1,x - 1]
                n += img[y - 1,x + 1]
                n += img[y,x - 1]    
                n += img[y,x + 1]
                n += img[y,x]
                n += img[y + 1,x]    
                n += img[y + 1,x - 1]
                n += img[y + 1,x + 1]
                n /= 255        
                if n == 1:
                    extremes.append(p)
            except Exception:
                continue
        return extremes

    def getHomography(self,frame):
        """
        Detects Curvatures of Lanes in a given frame
        """

        # if not isinstance(self.H, type(None)):
        #     lanemask,lanemasktd,lanemasktd_marking,back_proj_img,roc_img = self.detectCurvedLane(frame)
        #     return lanemask,lanemasktd,lanemasktd_marking,back_proj_img,roc_img

        lanemask = self.getCurvedLaneMask(frame)
        element = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
        lanemask = cv2.dilate(lanemask,element,iterations=1)

        (contours,h) = cv2.findContours(lanemask,0, cv2.CHAIN_APPROX_NONE)
        lanemask_color = cv2.cvtColor(lanemask,cv2.COLOR_GRAY2BGR)

        out_mask = np.zeros_like(lanemask)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 50 or area > 1700:
                continue
            approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
            # if len(approx)!=4:
            #     continue
            cv2.drawContours(lanemask_color,[cnt],0,(0,255,0),2)
            cv2.drawContours(out_mask,[cnt], -1, 255, cv2.FILLED, 1)

        out=lanemask.copy()
        out[out_mask == 0] = 0
        lanemask = out

        # cv2.imshow("out",out)
        # cv2.waitKey(0)
        # return frame

        height = lanemask.shape[0]
        width = lanemask.shape[1]

        distribution = np.sum(lanemask, axis=0)
        if not self.lineposPlotter:
            self.lineposPlotter = Plotter('Distribution of white pixels','columns of Image','Frequency',width,width)
        self.lineposPlotter.plot(self.lineposPlotter.line1, np.arange(len(distribution)),distribution)


        #Create default parametrization LSD
        lsd = cv2.createLineSegmentDetector(0)

        #Detect lines in the image
        linesP = lsd.detect(lanemask)[0] #Position 0 of the returned tuple are the detected lines

        #Draw detected lines in the image
        lsd_img = lsd.drawSegments(lanemask,linesP)

        cdstP = cv2.cvtColor(lanemask,cv2.COLOR_GRAY2BGR)

        no_bins = 6
        bins = np.linspace(0,width,no_bins)

        if linesP is not None:
            for i in range(0, len(linesP)):
                l = np.array(linesP[i][0],dtype=np.float32)
                length = np.linalg.norm(l[:2]-l[2:])

                if length < 20:
                    continue

                current_line_bin = round((np.digitize(l[0], bins, right=False)+np.digitize(l[2], bins, right=False))/2)

                if length > self.max_line_length:
                    self.long_line = l
                    self.max_line_length = length
                    self.long_line_bin = current_line_bin
                    self.lineposPlotter.l1.set_xdata((self.long_line_bin)*width/no_bins)
                
                proposed_short_line_bin = current_line_bin
                if proposed_short_line_bin != self.long_line_bin:
                    self.short_line = l
                    self.min_line_length = length
                    self.short_line_bin = proposed_short_line_bin
                    # print(self.short_line_bin, self.long_line_bin)
                    self.lineposPlotter.l2.set_xdata((self.short_line_bin)*width/no_bins)

                                
                if not isinstance(self.long_line, type(None)) and not isinstance(self.short_line, type(None)):
                    (leftline, rightline, color) =  (self.long_line, self.short_line, GREEN) if self.long_line_bin < self.short_line_bin else (self.short_line, self.long_line, RED)
                    
                    # Find the line intersections with the top and bottom of the image:
                    # x is opencv image columns, y is opencv image rows. 
                    # Line format [x1, y1, x2, y2]
                    m_ll = (leftline[3] - leftline[1])/(leftline[2] - leftline[0])
                    y_ll = leftline[1]
                    x_ll = leftline[0]
                    m_rl = (rightline[3] - rightline[1])/(rightline[2] - rightline[0])
                    y_rl = rightline[1]
                    x_rl = rightline[0]

                    # cv2.line(cdstP, (leftline[0], leftline[1]), (leftline[2], leftline[3]), RED, 3, cv2.LINE_AA)
                    # cv2.line(cdstP, (rightline[0], rightline[1]), (rightline[2], rightline[3]), GREEN, 3, cv2.LINE_AA)

                    try:
                        y=0
                        top_left = (round((m_ll)*(y-y_ll)+x_ll), y)
                        top_right = (round((m_rl)*(y-y_rl)+x_rl), y)

                        y=frame.shape[0]
                        bottom_left = (round((1/m_ll)*(y-y_ll)+x_ll), y)
                        bottom_right = (round((1/m_rl)*(y-y_rl)+x_rl), y)
                        # print(tuple(top_left), top_right, bottom_left, bottom_right)
                    except Exception:
                        continue

                    # Points chosen for Homography:
                    #Draw circle at start position of line:
                    # cv2.circle(cdstP, tuple(top_left), 20, YELLOW, 3, cv2.LINE_AA)

                    # # #Draw circle at end position of line:
                    # cv2.circle(cdstP, tuple(top_right), 20, OLIVE, 3, cv2.LINE_AA)
                    
                    # # #Draw circle at start position of line:
                    cv2.circle(cdstP, bottom_left, 20, RED, 3, cv2.LINE_AA)

                    # # #Draw circle at end position of line:
                    cv2.circle(cdstP, bottom_right, 20, GREEN, 3, cv2.LINE_AA)


                    # Find vanishing point:
                    x_intersection = (m_ll*x_ll - m_rl*x_rl + y_rl - y_ll)/(m_ll-m_rl) 
                    y_intersection = m_ll*(x_intersection - x_ll) + y_ll

                    x_intersection = round(x_intersection)
                    y_intersection = round(y_intersection)


                    l = np.asarray(l,dtype=np.int)
                    #Draw circle at start position of line:
                    cv2.circle(cdstP, (l[0], l[1]), 5, YELLOW, 3, cv2.LINE_AA)

                    #Draw circle at end position of line:
                    cv2.circle(cdstP, (l[2], l[3]), 5, OLIVE, 3, cv2.LINE_AA)

                    cv2.line(cdstP, bottom_right, (x_intersection,y_intersection), color, 3, cv2.LINE_AA)
                    color = GREEN if color==RED else RED
                    cv2.line(cdstP, bottom_left, (x_intersection,y_intersection), color, 3, cv2.LINE_AA)


                    cv2.circle(cdstP, (x_intersection, y_intersection), 20, ORCHID, 10, cv2.LINE_AA)
                    # print(x_intersection,y_intersection)

                    offset = 30
                    src = np.array([[x_intersection-offset,y_intersection], [x_intersection+offset, y_intersection], [bottom_right[0], bottom_right[1]], [bottom_left[0], bottom_left[1]] ], dtype=np.float32)
                    # src[:,[0,1]] = src[:,[1,0]]
                    # src = np.array([[355,310], [442, 310], [750, 510], [70, 510] ], dtype=np.float32)
                    # src = np.roll(src, 1)
                    tw = height
                    th = width
                    dst = np.array([[0,0],[tw,0],[tw,th],[0,th]],dtype=np.float32)
                    
                    self.H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
                    # warped_img = cv2.warpPerspective(frame, self.H,(height,width))
                    # cv2.imshow("warped_img",warped_img)


                color = RED
                # print(l,highest_bin_val,(long_line_bin+1)*width/no_bins)
                if current_line_bin==self.short_line_bin:
                    # if l[0] < highest_bin_val or l[2] < highest_bin_val or l[1] < highest_bin_val:
                        color = GREEN

                # cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), color, 3, cv2.LINE_AA)

        # cv2.imshow("lanemask",cdstP)


        return frame 

    def updatefromHistory(self,ex,ey):
        temp=[]
        thresh = 100
        for x,y in zip(ex,ey):
            if len(self.ld[x])>30:
                self.ld[x].pop(0)
            avg = np.average(self.ld[x])
            if (abs(avg - y) > thresh):
                temp.append(np.average(self.ld[x]))
                continue
            self.ld[x].append(y)
            temp.append(np.average(self.ld[x]))
        return np.array(temp,dtype=int)

    def drawCircles(self,image,x_coords,y_coords,color=OLIVE):
        for x,y in zip(x_coords,y_coords):
            cv2.circle(image, (int(x),int(y)), 5, color, 3, cv2.LINE_AA)
        return image

    def getPlanePoints(self,exl,eyl):
        H_inv = np.linalg.inv(self.H)
        plane_pts=[]
        for x,y in zip(exl,eyl):
            [x,y,w]= H_inv@np.array([x,y,1])
            x=x/w
            y=y/w
            plane_pts.append([x,y])
        plane_pts = np.array([plane_pts],dtype=int)
        plane_pts = np.squeeze(plane_pts)
        return plane_pts

    def findROC(self,x_coords,y_coords):
        idx, delta = len(y_coords)//2,len(y_coords)//4

        dy = y_coords[idx+delta] - y_coords[idx]
        dx = x_coords[idx+delta] - x_coords[idx]

        # Sample point distance: sp
        sp = len(y_coords)//5
        dy_ = y_coords[idx+delta+sp] - y_coords[idx+sp]
        dy2 = dy_ - dy
        roc= ((1+(dy/dx)**2)**1.5)/(dy2/dx)
        return roc

    def detectwithTopDownView(self, frame):

        # self.calibrateColor(frame)
        original_frame = frame.copy()
        height = frame.shape[0]
        width = frame.shape[1]

        lanemask = self.getCurvedLaneMask(frame,220)
        lanemasktd = cv2.warpPerspective(lanemask, self.H,(height,width))
        
        #'''
        # _,lanemasktd = cv2.threshold(lanemasktd, 10, 255, cv2.THRESH_BINARY)
        # skel = lanemasktd
        #'''

        frame = lanemasktd
        element = cv2.getStructuringElement(cv2.MORPH_CROSS,(2,2))
        done = False
        skel = np.zeros(frame.shape,np.uint8)
        size = np.size(frame)
        while( not done):
            eroded = cv2.erode(frame,element)
            temp = cv2.dilate(eroded,element)
            temp = cv2.subtract(frame,temp)
            skel = cv2.bitwise_or(skel,temp)
            frame = eroded.copy()

            zeros = size - cv2.countNonZero(frame)
            if zeros==size:
                done = True


        # element = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))

        # (contours,h) = cv2.findContours(lanemask,0, cv2.CHAIN_APPROX_NONE)
        # lanemask_color = cv2.cvtColor(lanemask,cv2.COLOR_GRAY2BGR)

        # out_mask = np.zeros_like(lanemask)

        # for cnt in contours:
        #     area = cv2.contourArea(cnt)
        #     cntpts = np.squeeze(cnt)
        #     if len(cntpts)<3:
        #         continue
        #     minx = min(cntpts[:,0])
        #     maxx = max(cntpts[:,0])
        #     miny = min(cntpts[:,1])
        #     maxy = max(cntpts[:,1])
        #     sqarea = (abs(minx-maxx)*abs(miny-maxy))
        #     # print(area, sqarea)
        #     if area/sqarea>math.pi/5:
        #         continue
        #     # approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
        #     # if len(approx)!=4:
        #     #     continue
        #     cv2.drawContours(lanemask_color,[cnt],0,(0,255,0),2)
        #     cv2.drawContours(out_mask,[cnt], -1, 255, cv2.FILLED, 1)

        # # out=lanemask.copy()
        # # out[out_mask == 0] = 0 
        
        # Get indices of x and y from left and right lanes
        
        def lsqr(lx,ly):
            a,b,c = symbols('a b c')
            sqx = [i**2 for i in lx]
            X = Matrix([[a],[b],[c]])
            A = np.zeros(shape=(len(lx),3))
            B = Matrix(ly)
            for i in range(1,len(lx)):
                A[i][0] = sqx[i]
                A[i][1] = lx[i]
                A[i][2] = 1 
            A = Matrix(A)
            # print (A)
            temp = transpose(A)*A
            inA = temp.inv()
            X = (inA)*transpose(A)*B
            return (X)

        # Left side:
        left_half = skel[:,:skel.shape[1]//2]
        right_half = skel[:,skel.shape[1]//2:]
        # print(left_half.shape)
        # cv2.imshow("left_half",left_half)
        # cv2.imshow("right_half",right_half)

        left_half_coords = np.squeeze(np.dstack(np.where(left_half>80)))
        left_half_coords = left_half_coords[left_half_coords[:, 1].argsort()]
        right_half_coords = np.squeeze(np.dstack(np.where(right_half>80)))
        right_half_coords = right_half_coords[right_half_coords[:, 1].argsort()]
        right_half_coords[:,1] = right_half_coords[:,1]+left_half.shape[1]
       
        NUM_PTS = 10

        # Equation x points, left lane
        exl = np.linspace(0, left_half.shape[1], NUM_PTS)

        # Equation x points, right lane
        exr = np.linspace(left_half.shape[1], int(left_half.shape[1]*2.2), NUM_PTS)
        # exr = np.linspace(3*left_half.shape[1], left_half.shape[1], NUM_PTS)

        Xl = lsqr(left_half_coords[:,1], left_half_coords[:,0])
        Xr = lsqr(right_half_coords[:,1], right_half_coords[:,0])
        
        # Equation y points, left lane
        eyl = Xl[0]*exl**2 + Xl[1]*exl + Xl[2]        
        
        # Equation y points, right lane
        eyr = Xr[0]*exr**2 + Xr[1]*exr + Xr[2]

        # cl = np.polynomial.polynomial.polyfit(left_half_coords[:,1], left_half_coords[:,0],1)       
        # cr = np.polynomial.polynomial.polyfit(right_half_coords[:,1], right_half_coords[:,0],1)       
        # fl = np.polynomial.polynomial.Polynomial(cl)
        # fr = np.polynomial.polynomial.Polynomial(cr)
        # eyl = fl(exl)
        # eyr = fr(exr)

        eyl=np.array(eyl,dtype=int)
        eyr=np.array(eyr,dtype=int)

        #both left and right lanes:
        bx = np.concatenate((exr,exl))
        by = np.concatenate((eyr,eyl))

        # Keeping track of history of predictions
        # no_bins = 6
        # bins = np.linspace(0,left_half.shape[0],no_bins)
        # l_bin_idx = np.digitize(eyl, bins, right=False)
        # l_bin_history  = []

        # Create a dictionary with key as x axis and y as array of values
        # at timesteps t
        
        
        eyl = self.updatefromHistory(exl,eyl)
        eyr = self.updatefromHistory(exr,eyr)

        #Lane centers
        # For each point in y, get the average of x 
        cx = (exl+eyl)//2#np.ones_like(exl)*exl.shape
        cy = (eyl)//1
        
        skel_color = cv2.cvtColor(skel,cv2.COLOR_GRAY2BGR)
        skel_color = self.drawCircles(skel_color,bx,by,color=ORCHID)
        skel_color = self.drawCircles(skel_color,exl,eyl,color=RED)
        skel_color = self.drawCircles(skel_color,exr,eyr,color=GREEN)
        # skel_color = self.drawCircles(skel_color,cx,cy,color=YELLOW)
        polylinesl = np.squeeze(np.dstack((exl,eyl)))
        polylinesr = np.squeeze(np.dstack((exr,eyr)))
        skel_color = cv2.polylines(skel_color,  np.int32([polylinesl]),isClosed = False, color=RED,thickness =5)
        skel_color = cv2.polylines(skel_color,  np.int32([polylinesr]),isClosed = False, color=GREEN,thickness =5)
        # skel_color = cv2.polylines(skel_color, np.array(np.squeeze(np.dstack((exr,eyr))),dtype=int),isClosed = False, color=GREEN,thickness =5)

        cv2.imshow("skel_color",skel_color)
        lanemasktd_marking = skel_color

        plane_pts_l=self.getPlanePoints(exl,eyl)

        original_frame = self.drawCircles(original_frame,plane_pts_l[:,0],plane_pts_l[:,1],color=RED)

        plane_pts_r=self.getPlanePoints(exr,eyr)
        
        original_frame = self.drawCircles(original_frame,plane_pts_r[:,0],plane_pts_r[:,1],color=GREEN)

        plane_pts_c=self.getPlanePoints(cx,cy)
        
        # original_frame = self.drawCircles(original_frame,plane_pts_c[:,0],plane_pts_c[:,1],color=YELLOW)

        mask = np.zeros_like(original_frame)

        plane_pts = np.concatenate((plane_pts_l,plane_pts_r))
        # plane_pts= np.array([plane_pts],dtype=np.int32)
        imColorLane = cv2.fillPoly(mask, [plane_pts], (80,227, 227))
        imColorLane = cv2.polylines(imColorLane, [plane_pts_l],isClosed = False, color=RED,thickness =5)
        imColorLane = cv2.polylines(imColorLane, [plane_pts_r],isClosed = False, color=GREEN,thickness =5)

        imColorLane = cv2.addWeighted(original_frame,1,imColorLane,0.5,0)

        # cv2.fillConvexPoly(mask, plane_pts, 1)
        # mask = mask.astype(np.bool)

        # out_frame = np.zeros_like(original_frame)
        # out_frame[mask] = original_frame[mask]
        cv2.imshow("original_frame",imColorLane)

        back_proj_img = imColorLane

        ROCleft = self.findROC(exl,eyl)
        ROCright = self.findROC(exr,eyr)

        roc_left_text = "ROC Left: " + str(round(ROCleft,2))
        roc_right_text ="ROC Right: " + str(round(ROCright,2))
        roc_img = np.zeros_like(original_frame)
        cv2.putText(roc_img, roc_left_text, (roc_img.shape[0]//2-200,roc_img.shape[1]//3), \
                        cv2.FONT_HERSHEY_COMPLEX,fontScale=0.71,color=WHITE,thickness=2,lineType=cv2.LINE_8 )

        cv2.putText(roc_img, roc_right_text,(roc_img.shape[0]//2+200,roc_img.shape[1]//3), \
                        cv2.FONT_HERSHEY_COMPLEX,fontScale=0.71,color=WHITE,thickness=2,lineType=cv2.LINE_8 )

        lanemask = cv2.cvtColor(lanemask,cv2.COLOR_GRAY2BGR)
        lanemasktd = cv2.cvtColor(lanemasktd,cv2.COLOR_GRAY2BGR)
        # lanemask = cv2.cvtColor(lanemask,cv2.COLOR_GRAY2BGR)
        return lanemask,lanemasktd,lanemasktd_marking,back_proj_img,roc_img

