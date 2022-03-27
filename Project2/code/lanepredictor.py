from attr import NOTHING
import cv2
import numpy as np
from utils import ImageGrid, Plotter
from preprocess import *
import math 
import matplotlib.pyplot as plt

RED = (0,0,255)
GREEN = (0,255,0)
YELLOW = (0,255,255)
OLIVE = (0,128,128)
class LanePredictor:

    def __init__(self,data=None,video=None):
        self.data=data
        self.video=video
        self.HISTOGRAM_EQUALIZATION = "Histogram Equalization"
        self.ADAPTIVE_HISTOGRAM_EQUALIZATION = "Adaptive Histogram Equalization"
        self.histogram_profile=[None, None, None]
        self.USE_HISTOGRAM_MEMORY=False
        # self.createCalibrationSliders()
        self.histplotter = None
        self.setupLinevariabless()


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


    def calibrateColor(self, frame):
        
        # create switch for ON/OFF functionality
        original_frame=frame.copy()
        while True:
            s = cv2.getTrackbarPos(self.switch,'image')

            if s:
                frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2LAB)
            else:
                frame = original_frame
            
            # get current positions of four trackbars
            lpos = cv2.getTrackbarPos('L','image')
            apos = cv2.getTrackbarPos('A','image')
            bpos = cv2.getTrackbarPos('B','image')
            
            # get trackbar positions
            lposH = cv2.getTrackbarPos('L_High', 'image')
            aposH = cv2.getTrackbarPos('A_High', 'image')
            bposH = cv2.getTrackbarPos('B_High', 'image')

            # Reject colors outside the range of low and high sliders.
            lower_hsv = np.array([lpos, apos, bpos])
            higher_hsv = np.array([lposH, aposH, bposH])
            mask = cv2.inRange(frame, lower_hsv, higher_hsv)
            output_img = cv2.bitwise_and(frame, frame, mask=mask)

            # Try thresholding values
            L, a, b = cv2.split(frame)
            ret,thresh_l = cv2.threshold(L,lpos,255,cv2.THRESH_BINARY)
            ret,thresh_a = cv2.threshold(a,apos,255,cv2.THRESH_BINARY)
            ret,thresh_b = cv2.threshold(b,bpos,255,cv2.THRESH_BINARY)
            # ret,lab_thresh_img = cv2.threshold(lab_img[:-1],200,255,cv2.THRESH_BINARY)
            # output_img = cv2.merge([thresh_l,thresh_a,thresh_b])
            # ret,lab_thresh_img = cv2.threshold(lab_img[:-2],20,255,cv2.THRESH_BINARY)
            # lab_img[:-2] = lab_thresh_img
            # lab_img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2BGR)
            # lab_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # lab_img = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            if s:
                output_img = cv2.cvtColor(output_img, cv2.COLOR_LAB2BGR)
            cv2.imshow("image", output_img)
            
            break

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


    def setupLinevariabless(self):
        self.max_line_length = -float('inf')
        self.long_line = None

        self.min_line_length = float('inf')
        self.short_line = None
        self.short_line_bin = None
        self.long_line_bin = None

    def detectStraightLane(self, frame):
        """
        Detects straight Lanes in a given frame
        """
        global fig, ax, lineL
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
        cdstP = np.copy(cdst)
             
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
                    print(self.short_line_bin, self.long_line_bin)
                    self.histplotter.l2.set_xdata((self.short_line_bin)*width/no_bins)

                                
                if not isinstance(self.long_line, type(None)) and not isinstance(self.short_line, type(None)):
                    (leftline, rightline) =  (self.long_line, self.short_line) if self.long_line_bin < self.short_line_bin else (self.short_line, self.long_line)
                    src = np.array([[leftline[2], leftline[3]], [rightline[0], rightline[1]], [rightline[2], rightline[3]], [leftline[0], leftline[1]]  ])
                    dst = np.array([[0,0],[0,width],[height,width], [height,0]])
                    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
                    warped_img = cv2.warpPerspective(frame, H,(height,width))
                    cv2.imshow("warped_img",warped_img)


                color = RED
                # print(l,highest_bin_val,(long_line_bin+1)*width/no_bins)
                if current_line_bin==self.short_line_bin:
                    # if l[0] < highest_bin_val or l[2] < highest_bin_val or l[1] < highest_bin_val:
                        color = GREEN

                cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), color, 3, cv2.LINE_AA)
                

        cv2.imshow("Hough Lines", cdstP)

        return cdstP


    def detectCurvedLane(self,frame):
        """
        Detects Curvatures of Lanes in a given frame
        """

        pass 