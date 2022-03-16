import cv2
import numpy as np
from utils import Plot, drive_downloader, getKeyPoints, trimImage

ORB = 'orb'
SIFT = 'sift'

image_fileA = '../data/Q2imageA.png'
image_fileB = '../data/Q2imageB.png'
drive_downloader('1WsnJ5xi9yj-6cLcXABsfdWtw7k_kspz1',image_fileA)
drive_downloader('17Gc-4gA6GGiiAD0HXL24XrWNQJUR9rt1',image_fileB)

imageA = cv2.imread(image_fileA)
imageB = cv2.imread(image_fileB)
imageA = cv2.cvtColor(imageA,cv2.COLOR_BGR2RGB)
imageB = cv2.cvtColor(imageB,cv2.COLOR_BGR2RGB)

grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
kpsA,descA = getKeyPoints(grayA,ORB)
siftImgA = imageA.copy()
cv2.drawKeypoints(siftImgA,kpsA,siftImgA)
cv2.imshow("SIFT Keypoints on Image A ",siftImgA)

grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
kpsB,descB = getKeyPoints(grayB,ORB)
siftImgB = imageB.copy()
cv2.drawKeypoints(siftImgB,kpsB,siftImgB)
cv2.imshow("SIFT Keypoints on Image B ",siftImgB)

bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
matches = bf.match(descA,descB)

# Sort the matches in asceding order, i.e. closest matches first. 
matches = sorted(matches, key = lambda x:x.distance)


matches_img = np.zeros(imageA.shape)
matches_img = cv2.drawMatches(imageA, kpsA, imageB, kpsB, matches[:50], matches_img, flags=2)
cv2.imshow("Keypoint Matches", matches_img)

src = np.float32([kpsA[mat.queryIdx].pt for mat in matches]).reshape(-1,1,2)
dst = np.float32([kpsB[mat.trainIdx].pt for mat in matches]).reshape(-1,1,2)

H, masked = cv2.findHomography(dst, src, cv2.RANSAC, 5.0)
stitchedImg = cv2.warpPerspective(imageB,H,(imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
cv2.imshow("Stitched Image", stitchedImg)

trimmedImg = trimImage(stitchedImg)



imageAB = np.concatenate((imageA, imageB), axis=1)

plot = Plot(3,2)
plot.set(imageAB, "Original Image A and Image B")
plot.set(siftImgA, "SIFT on Image A")
plot.set(siftImgB, "SIFT on Image B")
plot.set(matches_img, "Matches on Image A and B")
plot.set(stitchedImg, "Stitched Image")
plot.set(trimmedImg, "Trimmed Image")
plot.save('../outputs/',"output2.png")

cv2.waitKey(0)
