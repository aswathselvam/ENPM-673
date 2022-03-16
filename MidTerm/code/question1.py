import cv2
import numpy as np
import os.path

from google_drive_downloader import GoogleDriveDownloader as gdd



image_file = './Q1image.png'

if not os.path.isfile(image_file): 
    gdd.download_file_from_google_drive(file_id='11FwkdoLVlf27uCKXC4lj5T_xh7pYVdul',
                                        dest_path='./Q1image.png',
                                        unzip=False)

image = cv2.imread(image_file)
print(image.shape)

kernel = np.ones((20, 20), 'uint8')

erode_img = cv2.erode(image, kernel, iterations=1)
cv2.imshow('Eroded Image', erode_img)


imgray = cv2.cvtColor(erode_img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(image, contours, 0, (0,255,0), 3)
cv2.imshow('Contours', image)

# There are 24 coins in the given image 

cv2.waitKey(0)