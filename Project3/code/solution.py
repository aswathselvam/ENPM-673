import cv2
import os
from cv2 import VideoWriter
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
import argparse
from utils import *
from libs import DepthComputer
import glob

import yaml
import sys


def main():
    
    
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--data', default='../data/', help='stero images path , Default:../data/')
    Parser.add_argument('--savepath', default='../outputs/', help='Template Path , Default: ../outputs/')
    Parser.add_argument('--problem', default='2', help='The problem numer ex: 1, 2 or 3, default is 1')
    
    args = Parser.parse_args()
    data = args.data
    savepath = args.savepath
    problem = int(args.problem)

    if(not (os.path.isdir(savepath))):
        os.makedirs(savepath)

    depthComputer = []

    shapesFolder=['curule/','octagon/','pendulum/']

    K = []
    cam0=np.array([ [1758.23, 0, 977.42], [0, 1758.23, 552.15], [0, 0, 1] ])
    cam1=np.array([ [1758.23, 0, 977.42], [0, 1758.23, 552.15], [0, 0, 1] ])
    K.append(cam0)

    cam0=np.array([ [1742.11, 0, 804.90], [0, 1742.11, 541.22], [0, 0, 1] ])
    cam1=np.array([ [1742.11, 0, 804.90], [0, 1742.11, 541.22], [0, 0, 1] ])
    K.append(cam0)

    cam0=np.array([ [1729.05, 0, -364.24], [0, 1729.05, 552.22], [0, 0, 1] ])
    cam1=np.array([ [1729.05, 0, -364.24], [0, 1729.05, 552.22], [0, 0, 1] ])
    K.append(cam0)

    for idx, shapeFolder in enumerate(shapesFolder):
        image_files = sorted(glob.glob(data+shapeFolder+"*.png"))
        left_image = cv2.imread(image_files[0])
        right_image = cv2.imread(image_files[1])
        
        d = {}
        with open(data+shapeFolder+"calib.txt") as f:
            for line in f:
                (key, val) = line.split("=")
                if 'cam' not in key:
                    d[key]= float(val.strip())
                d['k']=K[idx]
                d['name'] = shapeFolder.strip('/')
        depthComputer.append(DepthComputer(left_image,right_image,d))
        

    #-----------Problem 1: Calibration-------------#
    for i in range(len(depthComputer)):
        depthComputer[i].calibrate()

    #-----------Problem 2: Rectification-----------#
    for i in range(len(depthComputer)):
        pass


    #-----------Problem 3: Correspondence-----------#

    
    
    #-----------Problem 4: Compute Depth Image------#


    # epilines_1 = cv2.computeCorrespondEpilines(r_points_2.reshape(-1,1,2), 2,best_f_matrix)
    # epilines_1 = epilines_1.reshape(-1,3)

    # epilines_2 = cv2.computeCorrespondEpilines(r_points_1.reshape(-1,1,2), 1,best_f_matrix)
    # epilines_2 = epilines_2.reshape(-1,3)

    # img1, img2 = drawlines(img1,img2,epilines_1,r_points_1[:100],r_points_2[:100])
    # img1, img2 = drawlines(img2,img1,epilines_2,r_points_1[:100],r_points_2[:100])

    # one_s = np.ones((r_points_1.shape[0],1))
    # r_points_1 = np.concatenate((r_points_1,one_s),axis = 1)
    # r_points_2 = np.concatenate((r_points_2,one_s),axis = 1)

    # Hom_0 , Hom_1 = to_rectify(best_f_matrix,features_image_1, features_image_2)

    # print('Homography Mat 1 : ',Hom_0)
    # print('Homography Mat 2 : ',Hom_1)

    # left_rectified = cv2.warpPerspective(img1, Hom_0, (640,480))
    # right_rectified = cv2.warpPerspective(img2, Hom_1, (640,480))

    # left_rec_nolines = cv2.warpPerspective(img_1_copy1, Hom_0, (640,480))
    # right_rec_nolines = cv2.warpPerspective(img_2_copy1, Hom_1, (640,480))

    # cv2.imshow("Epilines Drawn on Rectified Left Image ",left_rectified)
    # cv2.imshow("Epilines Drawn on Rectified Right Image ",right_rectified)
    # print('Enter any Key')
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # To_Depth = input('Are the epipolar lines parallel? \nEnter 1 for YES and anything else for NO --> ')
    # #run the depth and disparity calculation part only if the epipolar lines seem parallel for better results
    # if To_Depth == '1':
        
    #     #Remember to un-hash the correct parameters at the top of the code before proceeding
        
    #     disp = disparity_calc(left_rec_nolines,right_rec_nolines)
        
    #     #disp[disp >= 3] = 3
    #     cond1 = np.logical_and(disp >= 0,disp < 10)
    #     cond2 = disp > 40
        
    #     disp[cond1] = 10
    #     disp[cond2] = 40
        
    #     depth = 88.3 * f / disp
        
    #     plt.imshow(depth, cmap='gray', interpolation='bilinear')
    #     plt.title('Depth Plot Gray')
    #     plt.savefig('depth_gray.png')
    #     plt.show()
        
    #     plt.imshow(depth, cmap='hot', interpolation='bilinear')
    #     plt.title('Depth Plot Heat')
    #     plt.savefig('depth_heat.png')
    #     plt.show()
        
    # else:
    #     print('Please Re-run the Code')
    #     sys.exit()


if __name__ == '__main__':
    main()
    