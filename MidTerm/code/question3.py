from tkinter import W
import numpy as np

"""
Image points |  World points
x       y    |   X   Y   Z
757    13    |   0   0   0
758    15    |   0   3   0
758    86    |   0   7   0
759    66    |   0   11  0
1190   172   |   7   1   0
329    1041  |   0   11  7
1204   850   |   7   9   0
340    59    |   0   1   7
"""

img_pts = np.array([[757, 13],[758, 15],[758,86 ],[759,66],[1190,172],[329,1041],[1204,850],[340,59]])
world_pts = np.array([[0, 0,0 ],[0, 3, 0],[0,7,0],[0,11,0],[7,1,0],[0,11,7],[7,9,0],[0,1,7]])


img_pts = np.array([[757, 13],[758, 15],[758,86 ],[759,66],[1190,172],[329,1041],[1204,850],[340,59]])
world_pts = np.array([[0, 0,0 ],[0, 3, 0],[0,7,0],[0,11,0],[7,1,0],[0,11,7],[7,9,0],[0,1,7]])


def lsqr(A,b):
	temp = np.matmul(A.T,A)
	inA = np.linalg.inv(temp)
	X = np.matmul((inA), np.matmul((A.T),b))
	return X
print()
print(lsqr(world_pts,img_pts).T)