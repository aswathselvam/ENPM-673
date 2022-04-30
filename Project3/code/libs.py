import cv2
import numpy as np
from utils import ImageGrid, Plotter
import math 
import matplotlib.pyplot as plt
from collections import defaultdict
from sympy import *
from tqdm import tqdm


RED = (0,0,255)
GREEN = (0,255,0)
YELLOW = (0,255,255)
OLIVE = (0,128,128)
ORCHID = (204,50,153)
WHITE= (200,200,200)

ORB = 'orb'
SIFT = 'sift'


class DepthComputer:

    def __init__(self,left_image,right_image, d):
        self.left_image = left_image
        self.right_image = right_image
        self.K = d['k']
        self.baseline = d['baseline']
        self.name = d['name']

    def calibrate(self):
        #Select Matching Features:
        def getKeyPoints(img):
            # Initiate ORB detector
            sift = cv2.SIFT_create()
            # find the keypoints with ORB
            kp = sift.detect(img,None)
            # compute the descriptors with ORB
            kp, des = sift.compute(img, kp)
            return kp, des

        imageA = self.left_image
        imageB = self.right_image
        imageA = cv2.cvtColor(imageA,cv2.COLOR_BGR2RGB)
        imageB = cv2.cvtColor(imageB,cv2.COLOR_BGR2RGB)

        grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
        kpsA,descA = getKeyPoints(grayA)
        siftImgA = imageA.copy()
        cv2.drawKeypoints(siftImgA,kpsA,siftImgA)
        cv2.imshow("SIFT Keypoints on Image A ",siftImgA)

        grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
        kpsB,descB = getKeyPoints(grayB)
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

        # Compute F
        #1. xFx'=0
        #2. XF=0
        # X = np.array([])
        # for x,x_ in zip(src,dst):
        #     #[x1*x_1, x1*y_1, x1, y1*x_1, y1*y_1, y1, x_1, y_1, 1] 
        #     np.hstack(X,[x[0]*x_[0], x[0]*x_[1], x[0], x[1]*x_[0], x[1]*x_[1], x[1], x_[0], x_[1], 1])
        
        # u, s, v = np.linalg.svd(X, full_matrices=True)
        # v = v.T
        # F = v[:,-1]

        # F = F.reshape((3,3))
    
        # Uf, sigma_f, Vf = np.linalg.svd(F)
        # #forcing the rank 2 constraint
        # sigma_f[-1] = 0

        features_image_1 , features_image_2 , img1 , img2 = self.match_features(self.left_image, self.right_image)
        F, r_points_1, r_points_2 = self.getF(features_image_1,features_image_2)
        self.F = F
        self.r_points_1 = r_points_1
        self.r_points_2 = r_points_2

        # self.F, self.r_points_1, self.r_points_2 = self.getF(src,dst)


        # Compute Essential Matrix E
        # E  = KTFK
        E = self.essentialMatrix(self.F)
        R, T = self.pose(E)
        R, T = self.Triangulate(R, T, features_image_1, features_image_2)



        print('Rotation matrix ', '\n' )
        print(R, '\n')
        print('Translation matrix ', '\n' )
        print(T, '\n')
        print('\n')

        # P=KR[I3×3−C] 


        H, masked = cv2.findHomography(dst, src, cv2.RANSAC, 15.0)

        self.R = R
        self.T = T


        epilines_1 = cv2.computeCorrespondEpilines(r_points_2.reshape(-1,1,2), 2, self.F)
        epilines_1 = epilines_1.reshape(-1,3)

        epilines_2 = cv2.computeCorrespondEpilines(r_points_1.reshape(-1,1,2), 1, self.F)
        epilines_2 = epilines_2.reshape(-1,3)


        img_1_copy1 = img1.copy()
        img_2_copy1 = img2.copy()

        img1, img2 = self.drawLines(img1,img2,epilines_1,r_points_1[:100],r_points_2[:100])
        img1, img2 = self.drawLines(img2,img1,epilines_2,r_points_1[:100],r_points_2[:100])
        cv2.imwrite(self.name+"_epipolar_lines.jpg", np.hstack((img1,img2)))

        one_s = np.ones((r_points_1.shape[0],1))
        r_points_1 = np.concatenate((r_points_1,one_s),axis = 1)
        r_points_2 = np.concatenate((r_points_2,one_s),axis = 1)

        Hom_0 , Hom_1 = self.rectify(F,features_image_1, features_image_2)

        print('Homography Matrix for left image : ',Hom_0)
        print('Homography Matrix for right image: ',Hom_1)

        left_rectified = cv2.warpPerspective(img1, Hom_0, (640,480))
        right_rectified = cv2.warpPerspective(img2, Hom_1, (640,480))
        cv2.imwrite(self.name+"_rectified.jpg", np.hstack((left_rectified,right_rectified)))


        left_rec_nolines = cv2.warpPerspective(img_1_copy1, Hom_0, (640,480))
        right_rec_nolines = cv2.warpPerspective(img_2_copy1, Hom_1, (640,480))
        

        imageA = left_rec_nolines
        imageB = right_rec_nolines
        imageA = cv2.cvtColor(imageA,cv2.COLOR_BGR2RGB)
        imageB = cv2.cvtColor(imageB,cv2.COLOR_BGR2RGB)

        grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
        kpsA,descA = getKeyPoints(grayA)
        siftImgA = imageA.copy()

        grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
        kpsB,descB = getKeyPoints(grayB)
        siftImgB = imageB.copy()

        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
        matches = bf.match(descA,descB)

        # Sort the matches in asceding order, i.e. closest matches first. 
        matches = sorted(matches, key = lambda x:x.distance)

        src = np.float32([kpsA[mat.queryIdx].pt for mat in matches]).reshape(-1,1,2)
        dst = np.float32([kpsB[mat.trainIdx].pt for mat in matches]).reshape(-1,1,2)

        H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        left_rec_nolines = cv2.warpPerspective(left_rec_nolines,H,(640,480))

        cv2.imwrite(self.name+"_rectified_aligned.jpg", np.hstack((left_rec_nolines,right_rec_nolines)))


        cv2.imshow("Epilines Drawn on Rectified Left Image ",left_rec_nolines)
        cv2.imshow("Epilines Drawn on Rectified Right Image ",right_rec_nolines)
        print('Enter any Key')
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        To_Depth = '1'
        if To_Depth == '1':
    
            #Remember to un-hash the correct parameters at the top of the code before proceeding
            
            disp = self.getDisparity(left_rec_nolines,right_rec_nolines)
            
            print('f from K[0][0]: ', self.K[0][0])
            depth = self.baseline * self.K[0][0] / disp
            
            # min_value = np.min(depth[depth>0])
            # max_value = np.max(depth[depth>0])
            # depth = 255*depth/(max_value-min_value)

            plt.imshow(depth, cmap='gray_r', interpolation='bilinear')
            plt.title('Depth Plot Gray')
            plt.colorbar()
            plt.savefig(self.name+'_depth_gray.png')
            plt.show()

            plt.imshow(depth, cmap='inferno_r', interpolation='bilinear')
            plt.title('Depth Plot Gray')
            plt.colorbar()
            plt.savefig(self.name+'_depth_color.png')
            plt.show()
            
        return


    def match_features(self,img_1,img_2):     
        img_1_sift = img_1.copy()
        img_2_sift = img_2.copy()
        
        img_1_sift = cv2.resize(img_1_sift,(640,480),fx=0,fy=0,interpolation=cv2.INTER_AREA)
        img_2_sift = cv2.resize(img_2_sift,(640,480),fx=0,fy=0,interpolation=cv2.INTER_AREA)
        
        sift = cv2.SIFT_create()
        
        keypoints_1, descriptors_1 = sift.detectAndCompute(img_1_sift,None)
        keypoints_2, descriptors_2 = sift.detectAndCompute(img_2_sift,None)
        
        
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        
        matches = bf.match(descriptors_1,descriptors_2)
        matches = sorted(matches, key = lambda x:x.distance)
        
        features_1 = []
        features_2 = []
        for i in matches:
            features_1.append(keypoints_1[i.queryIdx].pt)
            features_2.append(keypoints_2[i.trainIdx].pt)

        return features_1,features_2,img_1_sift,img_2_sift
    
    def F_matrix(self,f1,f2):
    
        f1_x = [] ; f1_y = [] ; f2_x = [] ; f2_y = []

        f1 = np.asarray(f1)
        f2 = np.asarray(f2)
        
        f1_x_mean = np.mean(f1[:,0])    
        f1_y_mean = np.mean(f1[:,1])    
        f2_x_mean = np.mean(f2[:,0])        
        f2_y_mean = np.mean(f2[:,1])
        
        for i in range(len(f1)): f1[i][0] = f1[i][0] - f1_x_mean
        for i in range(len(f1)): f1[i][1] = f1[i][1] - f1_y_mean
        for i in range(len(f2)): f2[i][0] = f2[i][0] - f2_x_mean
        for i in range(len(f2)): f2[i][1] = f2[i][1] - f2_y_mean
        
        f1_x = np.array(f1[:,0])
        f1_y = np.array(f1[:,1])
        f2_x = np.array(f2[:,0])
        f2_y = np.array(f2[:,1])

        sum_f1 = np.sum((f1)**2, axis = 1)
        sum_f2 = np.sum((f2)**2, axis = 1)
        
        k_1 = np.sqrt(2.)/np.mean(sum_f1**(1/2))
        k_2 = np.sqrt(2.)/np.mean(sum_f2**(1/2))
                                
        s_f1_1 = np.array([[k_1,0,0],[0,k_1,0],[0,0,1]])
        s_f1_2 = np.array([[1,0,-f1_x_mean],[0,1,-f1_y_mean],[0,0,1]])
        
        s_f2_1 = np.array([[k_2,0,0],[0,k_2,0],[0,0,1]])
        s_f2_2 = np.array([[1,0,-f2_x_mean],[0,1,-f2_y_mean],[0,0,1]])
        
        t_1 = np.dot(s_f1_1,s_f1_2)
        t_2 = np.dot(s_f2_1,s_f2_2)
        
        x1 = ( (f1_x).reshape((-1,1)) ) * k_1
        y1 = ( (f1_y).reshape((-1,1)) ) * k_1
        x2 = ( (f2_x).reshape((-1,1)) ) * k_2
        y2 = ( (f2_y).reshape((-1,1)) ) * k_2
        
        # A (8X9) matrix
        Alist = []
        for i in range(x1.shape[0]):
            X1, Y1 = x1[i][0],y1[i][0]
            X2, Y2 = x2[i][0],y2[i][0]
            Alist.append([X2*X1 , X2*Y1 , X2 , Y2 * X1 , Y2 * Y1 ,  Y2 ,  X1 ,  Y1, 1])
        A = np.array(Alist)
        
        U, sigma, VT = np.linalg.svd(A)
        
        v = VT.T
        
        f_val = v[:,-1]
        f_mat = f_val.reshape((3,3))
        
        Uf, sigma_f, Vf = np.linalg.svd(f_mat)
        
        #Force the rank 2, by setting last row to zero
        sigma_f[-1] = 0
        
        sigma_final = np.zeros(shape=(3,3)) 
        sigma_final[0][0] = sigma_f[0] 
        sigma_final[1][1] = sigma_f[1] 
        sigma_final[2][2] = sigma_f[2] 
        
        #un-normalizing 
        f_main = np.dot(Uf , sigma_final)
        f_main = np.dot(f_main , Vf)
        
        f_unnorm = np.dot(t_2.T , f_main)
        f_unnorm = np.dot(f_unnorm , t_1)
        
        f_unnorm = f_unnorm/f_unnorm[-1,-1]
        
        return f_unnorm

    def getF(self, img_1_features,img_2_features):
    
        #RANSAC parameters
        N = 1000
        sample = 0
        thresh = 0.05
        inliers_atm = 0
        P = 0.99
        R_fmat = []

        while sample < N:
            
            rand_p1 = [] ; rand_p2 = []
            
            #getting a set of random 8 points
            index = np.random.randint( len(img_1_features) , size = 8)
            
            for i in index:
                
                rand_p1.append(img_1_features[i])
                rand_p2.append(img_2_features[i])
            
            Fundamental = self.F_matrix(rand_p1, rand_p2)
            
            #Hartley's 8 points algorithm
            ones = np.ones((len(img_1_features),1))
            x_1 = np.concatenate((img_1_features,ones),axis=1)
            x_2 = np.concatenate((img_2_features,ones),axis=1)
            
            line_1 = np.dot(x_1, Fundamental.T)
            
            line_2 = np.dot(x_2,Fundamental)
        
            e1 = np.sum(line_2* x_1,axis=1,keepdims=True)**2
            
            e2 = np.sum(np.hstack((line_1[:, :-1],line_2[:,:-1]))**2,axis=1,keepdims=True)
            
            error =  e1 / e2 
            
            inliers = error <= thresh
            
            inlier_count = np.sum(inliers)
            
            #estimating best Fundamental M
            if inliers_atm <  inlier_count:
                
                inliers_atm = inlier_count
                
                good_ones = np.where(inliers == True)
                
                x_1_pts = np.array(img_1_features)
                x_2_pts = np.array(img_2_features)
                
                in_points_x1 = x_1_pts[good_ones[0][:]]
                in_points_x2 = x_2_pts[good_ones[0][:]]

                R_fmat = Fundamental
                
            #iterating for N number of times
            inlier_ratio = inlier_count/len(img_1_features)
            
            denominator = np.log(1-(inlier_ratio**8))
            
            numerator = np.log(1-P)
            
            if denominator == 0: continue
            N =  numerator / denominator
            sample += 1
            
        return R_fmat, in_points_x1, in_points_x2

    #essential matrix calculation using the best fundamental matrix and camera parameters
    def essentialMatrix(self, F_matrix):
        e_mat = np.dot(self.K.T,F_matrix)
        e_mat = np.dot(e_mat,self.K)
        #solving for E using SVD
        Ue, sigma_e, Ve = np.linalg.svd(e_mat)
        sigma_final = np.zeros((3,3))
        
        for i in range(3):
            sigma_final[i,i] = 1
        sigma_final[-1,-1] = 0

        E_mat = np.dot(Ue,sigma_final)
        E_mat = np.dot(E_mat,Ve)
        
        return E_mat

    def pose(self,e_mat):

        Ue, sigma, Ve = np.linalg.svd(e_mat)
        d = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        
        Rot_1 = np.dot(Ue, d)
        Rot_1 = np.dot(Rot_1, Ve)
        c1 = Ue[:, 2]
        if (np.linalg.det(Rot_1) < 0):
            Rot_1 = -Rot_1
            c1 = -c1
            
        Rot_2 = Rot_1
        c2 = -Ue[:, 2]
        if (np.linalg.det(Rot_2) < 0):
            Rot_2 = -Rot_2
            c2 = -c2
            
        Rot_3 = np.dot(Ue, d.T)
        Rot_3 = np.dot(Rot_1, Ve)
        c3 = Ue[:, 2]
        if (np.linalg.det(Rot_3) < 0):
            Rot_3 = -Rot_3
            c3 = -c3
            
        Rot_4 = Rot_3
        c4 = -Ue[:, 2]
        if (np.linalg.det(Rot_4) < 0):
            Rot_4 = -Rot_4
            c4 = -c4
        
        c1 = c1.reshape((3,1))
        c2 = c2.reshape((3,1))
        c3 = c3.reshape((3,1))
        c4 = c4.reshape((3,1))
        
        rot_final = [Rot_1,Rot_2,Rot_3,Rot_4]
        c_final = [c1,c2,c3,c4]
        
        return rot_final,c_final

    #estimating point correspondences 
    def threeD_pts(self, r2,c2,p1,p2):
        
        c1 = np.array([[0],[0],[0]])
        
        r1 = np.identity(3)
        
        r1_c1 = -np.dot(r1,c1)
        r2_c2 = -np.dot(r2,c2)

        j1 = np.concatenate((r1, r1_c1), axis = 1)
        j2 = np.concatenate((r2, r2_c2), axis = 1)


        P1 = np.dot(self.K,j1)
        P2 = np.dot(self.K,j2)

        l_sol = []
        
        for i in range(len(p1)):
            
            x_1 = np.array(p1[i])
            x_2 = np.array(p2[i])
            
            x_1 = np.reshape(x_1,(2,1))
            q = np.array([1])
            q = np.reshape(q,(1,1))
            
            x_1 = np.concatenate((x_1,q), axis = 0)
            x_2 = np.reshape(x_2,(2,1))
            x_2 = np.concatenate((x_2,q), axis = 0)
    
            x_1_skew = np.array([[0,-x_1[2][0],x_1[1][0]],[x_1[2][0], 0, -x_1[0][0]],[-x_1[1][0], x_1[0][0], 0]])
            x_2_skew = np.array([[0,-x_2[2][0],x_2[1][0]],[x_2[2][0], 0, -x_2[0][0]],[-x_2[1][0], x_2[0][0], 0]])
            
            A1 = np.dot(x_1_skew, P1)
            A2 = np.dot(x_2_skew, P2)
            #calculating A and solving using SVD
            A = np.zeros((6,4))
            for i in range(6):
                if i<=2:
                    A[i,:] = A1[i,:]
                else:
                    A[i,:] = A2[i-3,:]
                    
            U, sigma, VT = np.linalg.svd(A)
            VT = VT[3]
            VT = VT/VT[-1]
            l_sol.append(VT)
            
        l_sol = np.array(l_sol) 
        
        return l_sol    

    #checks the cheirality condition 
    def Triangulate(self, R_lis,C_lis,p1,p2):
        
        clis = list()
        for i in range(4):
            x = self.threeD_pts(R_lis[i],C_lis[i],p1,p2)
            n = 0
            for j in range(x.shape[0]):
                cord = x[j,:].reshape(-1,1)
                if np.dot(R_lis[i][2], (cord[0:3] - C_lis[i])) > 0 and cord[2]>0:
                    n += 1
            clis.append(n)
            ind = clis.index(max(clis))
            if C_lis[ind][2]>0:
                C_lis[ind] = -C_lis[ind]
        return R_lis[ind], C_lis[ind]


    #least squares technique
    def least_squares(self, x_1_ls, x_2_ls):
        
        lis = list()

        #forming the X matrix
        X = x_1_ls
        Y = np.reshape(x_2_ls, (x_2_ls.shape[0], 1))

        #computing B matrix 
        X_total = np.dot(X.T, X)
        X_total_inv = np.linalg.inv(X_total)
        Y_total = np.dot(X.T, Y)
        B_mat = np.dot(X_total_inv, Y_total)

        #computing the y coordinates and forming a list to return
        new_y = np.dot(X, B_mat)
        for i in new_y:
            for a in i:
                lis.append(a)

        return B_mat

    #rectification function to get the homography matrices
    def rectify(self, F_mat,points1,points2):
        
        points1 = np.asarray(points1)
        points2 = np.asarray(points2)
        # epipoles of left and right images
        U, sigma, VT = np.linalg.svd(F_mat)
        V = VT.T
        s = np.where(sigma < 0.00001)
        
        e_left = V[:,s[0][0]]
        e_right = U[:,s[0][0]]
        
        e_left = np.reshape(e_left,(e_left.shape[0],1))
        e_right = np.reshape(e_right,(e_right.shape[0],1))
        
        T_1 = np.array([[1,0,-(640/2)],[0,1,-(480/2)],[0,0,1]])
        e_final = np.dot(T_1,e_right)
        e_final = e_final[:,:]/e_final[-1,:]
        
        len = ((e_final[0][0])**(2)+(e_final[1][0])**(2))**(1/2)
        
        if e_final[0][0] >= 0:
            
            alpha = 1
        else:
            
            alpha = -1
            
        T_2 = np.array([[(alpha*e_final[0][0])/len, (alpha*e_final[1][0])/len, 0],
                        [-(alpha*e_final[1][0])/len, (alpha*e_final[0][0])/len, 0],[0, 0, 1]])
        e_final = np.dot(T_2,e_final)
        
        T_3 = np.array([[1, 0, 0],[0, 1, 0],[((-1)/e_final[0][0]), 0, 1]])
        e_final = np.dot(T_3,e_final)
        
        PHI2 = np.dot(np.dot(np.linalg.inv(T_1),T_3),np.dot(T_2,T_1))

        h_ones = np.array([1,1,1])
        h_ones = np.reshape(h_ones,(1,3))
        
        z = np.array([[0,-e_left[2][0],e_left[1][0]],[e_left[2][0],0,
                                    -e_left[0][0]],[-e_left[1][0],e_left[0][0],0]])
        
        M = np.dot(z,F_mat) + np.dot(e_left,h_ones)

        Homography = np.dot(PHI2,M)
        
        ones = np.ones((points1.shape[0],1))
        points_1 = np.concatenate((points1,ones), axis = 1)
        points_2 = np.concatenate((points2,ones), axis = 1)
        
        x_1 = np.dot(Homography,points_1.T)
        x_1 = x_1[:,:]/x_1[2,:]
        x_1 = x_1.T
        
        x_2 = np.dot(PHI2,points_2.T)
        x_2 = x_2[:,:]/x_2[2,:]
        x_2 = x_2.T
        
        x_2_dash = np.reshape(x_2[:,0], (x_2.shape[0],1))
        
        L_S = self.least_squares(x_1,x_2_dash)
        
        d_1 = np.array([[L_S[0][0],L_S[1][0],L_S[2][0]],[0,1,0],[0,0,1]])
        
        PHI1 = np.dot(np.dot(d_1,PHI2),M)
        
        return PHI1,PHI2


    #sum of absolute difference 
    def abs_diff(self, pixel_vals_1, pixel_vals_2):

        if pixel_vals_1.shape != pixel_vals_2.shape:
            return -1
        return np.sum(abs(pixel_vals_1 - pixel_vals_2))


    #compare left block of pixels with multiple blocks from the right image using 
    def window_compare(self, y, x, left_local, right, window_size=5):
        local_window = 10
        #get search range for the right image
        x_min = max(0, x - local_window)
        x_max = min(right.shape[1], x + local_window)
        min_sad = None
        index_min = None
        first = True
        
        for x in range(x_min, x_max):
            right_local = right[y: y+window_size,x: x+window_size]
            sad = self.abs_diff(left_local, right_local)
            if first:
                min_sad = sad
                index_min = (y, x)
                first = False
            else:
                if sad < min_sad:
                    min_sad = sad
                    index_min = (y, x)

        return index_min

    def getDisparity(self, image_left,image_right):
        
        left = np.asarray(image_left)
        right = np.asarray(image_right)
        
        left = left.astype(int)
        right = right.astype(int)
        
        if left.shape != right.shape:
            print("Image Shapes do not match!!")
        
        h, w , g = left.shape
        window = 10

        disparity = np.zeros((h, w))

        #going over each pixel
        for y in tqdm(range(window, h-window)):
            for x in range(window, w-window):
                left_local = left[y:y + window, x:x + window]
                index_min = self.window_compare(y, x, left_local, right, window_size = window)
                disparity[y, x] = abs(index_min[1] - x)
                
         
        # min_value = np.min(disparity)
        # max_value = np.max(disparity)
        # disparity = 255*disparity/(max_value-min_value)

        disp = np.zeros_like(disparity)
        disparity = cv2.normalize(disparity, disp, 0,255, cv2.NORM_MINMAX)
        plt.imshow(disparity, cmap='hot', interpolation='bilinear')
        plt.title('Disparity Plot Heat')
        plt.colorbar()
        plt.savefig(self.name+'_disparity_image_heat.png')
        plt.show()
        
        plt.imshow(disparity, cmap='gray', interpolation='bilinear')
        plt.title('Disparity Plot Gray')
        plt.colorbar()
        plt.savefig(self.name+'_disparity_image_gray.png')
        plt.show()
        
        return disparity

    #function to draw the epipolar lines on the given images
    def drawLines(self, drawline_im1,drawline_im2,lines,pts1,pts2):
    
        sh = drawline_im1.shape
        r = sh[0]
        c = sh[1]
        
        for r,pt1,pt2 in zip(lines,pts1,pts2):
            pt1 = [int(pt1[0]),int(pt1[1])]
            pt2 = [int(pt2[0]),int(pt2[1])]
            
            color = tuple(np.random.randint(0,255,3).tolist())
            x0,y0 = map(int, [0, -r[2]/r[1] ])
            x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
            drawline_im1 = cv2.line(drawline_im1, (x0,y0), (x1,y1), color,1)
            drawline_im1 = cv2.circle(drawline_im1,tuple(pt1),2,color,-1)
            drawline_im2 = cv2.circle(drawline_im2,tuple(pt2),2,color,-1)
            
        return drawline_im1,drawline_im2