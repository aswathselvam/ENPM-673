from turtle import distance
import cv2
import numpy as np
from utils import Plot, drive_downloader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
class KMeans:
    def __init__(self, K, image):
        self.K = K
        self.image = image
        self.means = np.random.randint(255, size=(K, 3))
        print("Randomly initialized means are: \n", self.means)
        self.pts=np.empty((0,3), int)
        for i in range(self.image.shape[0]):
            for j in range(self.image.shape[1]):
                self.pts = np.append(self.pts, [self.image[i,j,:].copy()],axis=0)
        print("Number of Feature Points: ",self.pts.shape)
        # self.visualizeData()

    def visualizeData(self):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        for pt in self.pts:
            x = pt[0]
            y = pt[1]
            z = pt[2]
            ax.scatter(x, y, z, marker='o',c=np.array([pt])/255)
        ax.set_xlabel('Red')
        ax.set_ylabel('Green')
        ax.set_zlabel('Blue')
        plt.show()
                

    def distance(self,p1,p2):
        dist = np.linalg.norm(p1-p2)
        return dist

    def cluster(self, target_iters=10): 
        pass


image_file = '../data/Q4image.png'
drive_downloader('1Fr78nf4LNAfDWU4R4GgGEuS7Rbk5tG_H',image_file)

image = cv2.imread(image_file)
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
image = cv2.resize(image,(20,20))
cv2.imshow("Original Image",image)
print("Image shape",image.shape)

K=4
image = np.array(image)
kmeans = KMeans(K, image)
kmeans.cluster()


plot = Plot(3,1)
plot.set(image, "Original Image")
plot.set(image, "3D points")

cv2.waitKey(0)