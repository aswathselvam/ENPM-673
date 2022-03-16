import cv2
import numpy as np
from utils import Plot, drive_downloader, getKeyPoints

ORB = 'orb'
SIFT = 'sift'

image_fileA = '../data/Q2imageA.png'
image_fileB = '../data/Q2imageB.png'
drive_downloader('1WsnJ5xi9yj-6cLcXABsfdWtw7k_kspz1',image_fileA)
drive_downloader('17Gc-4gA6GGiiAD0HXL24XrWNQJUR9rt1',image_fileB)

imageA = cv2.imread(image_fileA)
imageB = cv2.imread(image_fileB)
# imageA = cv2.cvtColor(imageA,cv2.COLOR_BGR2RGB)
# imageB = cv2.cvtColor(imageB,cv2.COLOR_BGR2RGB)

grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
kpsA,descA = getKeyPoints(grayA,ORB)
siftImgA = imageA.copy()
cv2.drawKeypoints(siftImgA,kpsA,siftImgA,(0,255,0),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("SIFT Keypoints on Image A ",siftImgA)

grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
kpsB,descB = getKeyPoints(grayB,ORB)
siftImgB = imageA.copy()
cv2.drawKeypoints(siftImgB,kpsB,siftImgB,(0,0,255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("SIFT Keypoints on Image B ",siftImgB)

bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
matches = bf.match(descA,descB)
matches = sorted(matches, key = lambda x:x.distance)

matches_img = np.zeros(imageA.shape)
matches_img = cv2.drawMatches(imageA, kpsA, imageB, kpsB, matches[:50], matches_img, flags=2)
cv2.imshow("Keypoint Matches", matches_img)

print(matches)

plot = Plot(3,3)
plot.set(imageA, "Image A")
plot.set(imageB, "Image B")
plot.set(siftImgA, "SIFT on Image A")
plot.set(siftImgB, "SIFT on Image B")
plot.set(matches_img, "SIFT on Image B")
plot.save('../outputs/',"output2.png")

cv2.waitKey(0)
