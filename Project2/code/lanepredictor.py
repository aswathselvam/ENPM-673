from attr import NOTHING
import cv2
import numpy as np
# import cupy as cp
from utils import ImageGrid
from preprocess import *
class LanePredictor:

    def __init__(self,data=None,video=None):
        self.data=data
        self.video=video
        self.HISTOGRAM_EQUALIZATION = "Histogram Equalization"
        self.ADAPTIVE_HISTOGRAM_EQUALIZATION = "Adaptive Histogram Equalization"
        self.histogram_profile=[None, None, None]
        self.USE_HISTOGRAM_MEMORY=False

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

    def detectStraightLane(self,frame):
        """
        Detects straight Lanes in a given frame
        """
        cv2.namedWindow('image')
        def nothing(x):
            pass
        
        # create trackbars for color change
        cv2.createTrackbar('L','image',0,255,nothing)
        cv2.createTrackbar('A','image',0,255,nothing)
        cv2.createTrackbar('B','image',0,255,nothing)

        # create switch for ON/OFF functionality
        switch = '0 : OFF \n1 : ON'
        cv2.createTrackbar(switch, 'image',0,1,nothing)

        lab_img=frame
        while True:
            # lab_img = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            lab_img=frame
            # get current positions of four trackbars
            lpos = cv2.getTrackbarPos('L','image')
            apos = cv2.getTrackbarPos('A','image')
            bpos = cv2.getTrackbarPos('B','image')
            s = cv2.getTrackbarPos(switch,'image')



            L, a, b = cv2.split(lab_img)
            ret,thresh_l = cv2.threshold(L,lpos,255,cv2.THRESH_BINARY)
            ret,thresh_a = cv2.threshold(a,apos,255,cv2.THRESH_BINARY)
            ret,thresh_b = cv2.threshold(b,bpos,255,cv2.THRESH_BINARY)
            # ret,lab_thresh_img = cv2.threshold(lab_img[:-1],200,255,cv2.THRESH_BINARY)
            lab_img = cv2.merge([thresh_l,thresh_a,thresh_b])
            # ret,lab_thresh_img = cv2.threshold(lab_img[:-2],20,255,cv2.THRESH_BINARY)
            # lab_img[:-2] = lab_thresh_img
            # lab_img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2BGR)
            # lab_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # lab_img = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            cv2.imshow("image", lab_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        pass


    def detectCurvedLane(self,frame):
        """
        Detects Curvatures of Lanes in a given frame
        """

        pass 