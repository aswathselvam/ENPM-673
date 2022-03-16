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
        self.pts=self.getPointsfromImage(self.image)
        print("Number of Feature Points: ",self.pts.shape)
        # self.visualizeData(self.pts)
    
    def getPointsfromImage(self,image):
        pts = np.empty((0,3), int)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                pts = np.append(pts, [image[i,j,:].copy()],axis=0)
        return pts

    def visualizeData(self,pts):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        for pt in pts:
            x = pt[0]
            y = pt[1]
            z = pt[2]
            ax.scatter(x, y, z, marker='o', s=200,c=np.array([pt])/255)
        ax.set_xlabel('Red')
        ax.set_ylabel('Green')
        ax.set_zlabel('Blue')
        plt.show()
                

    def distance(self,p1,p2):
        dist = np.linalg.norm(p1-p2)
        return dist

    def train(self, target_iters=10): 
        mean_diff = float('inf')
        iterations = 0
        self.THRESHOLD = 10
        while(mean_diff<self.THRESHOLD or iterations<target_iters):
            D =  {}  
            for pt in self.pts:
                mindist=float('inf')
                min_mean_idx = None
                for idx, mean in enumerate(self.means):
                    dist = self.distance(mean,pt)
                    if dist<mindist:
                        mindist = dist
                        min_mean_idx = idx
                if min_mean_idx not in D:
                    D[min_mean_idx] = []
                D[min_mean_idx].append(pt)
            
            for key, val in D.items():
                # print("key: ",key," lenfth: ", len(val),val[0])
                mean = np.average(val,axis=0)
                # print("Mean ",key," difference: ",self.means[key],mean)
                self.means[key] = mean
            print("Iteration ", iterations, " complete")
            iterations+=1
        
        self.visualizeData(self.means)

    def classify(self,image): 
        classes = image.copy()
        for i in range(self.image.shape[0]):
            for j in range(self.image.shape[1]):
                mindist=float('inf')
                min_mean_idx = None
                for idx, mean in enumerate(self.means):
                    dist = self.distance(mean,self.image[i,j,:])
                    if dist<mindist:
                        mindist = dist
                        min_mean_idx = idx

                classes[i,j,:] = self.means[min_mean_idx]
        cv2.imshow("Classified image", cv2.cvtColor(classes,cv2.COLOR_BGR2RGB))
        return classes

image_file = '../data/Q4image.png'
drive_downloader('1Fr78nf4LNAfDWU4R4GgGEuS7Rbk5tG_H',image_file)

image = cv2.imread(image_file)
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
image = cv2.resize(image,(100,100))
cv2.imshow("Original Image",cv2.cvtColor(image,cv2.COLOR_RGB2BGR))
print("Image shape",image.shape)

K=4
image = np.array(image)
kmeans = KMeans(K, image)
kmeans.train(100)
classes = kmeans.classify(image)

plot = Plot(2,1)
plot.set(image, "Original Image")
plot.set(classes, "Classified Image")
plot.save("../outputs/","output4.png")
cv2.waitKey(0)