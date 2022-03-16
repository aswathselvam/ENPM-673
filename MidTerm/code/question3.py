import numpy as np
import time 
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
world_pts = np.array([[0, 0,0,1 ],[0, 3, 0,1],[0,7,0,1],[0,11,0,1],[7,1,0,1],[0,11,7,1],[7,9,0,1],[0,1,7,1]])

#-----------------P matrix----------------------#
A = np.empty([3,12],dtype=float)
O = np.zeros(4,dtype=float)
for x,X in zip(img_pts,world_pts):
    u = x[0]
    v = x[1]
    w = 1
    # X = np.asarray(X,dtype=float)
    # print('X: ',X,' x: ', x,'\n')
    # i = np.asarray([O,-w*X, v*X])
    i = np.asarray([np.asarray([O, -w*X, v*X ]).flatten(), 
                np.asarray([w*X, O, -u*X]).flatten(),
                np.asarray([-v*X, u*X, O]).flatten()],dtype=float)
    i = np.squeeze(i)
    # print("Ashape: ",A.shape, "i : ",i.shape,"\n",i)
    A=np.vstack((A,i))
    # print("A\n",A)
    # input()

print("A: ",A.shape)
U, D, V = np.linalg.svd(A)
m, n = A.shape
A_reconstructed = U[:,:n] @ np.diag(D) @ V[:m,:]
print("U: ",U.shape)
print("D: ",D.shape, "\n",  D)
print("V: ",V.shape, "\n",  V.shape)

print("\nP is in the last column of V:\n",  V[:,n-1])
p1 = V[0:4,n-1]
p2 = V[4:8,n-1]
p3 = V[8:12,n-1]

P = np.array([p1,p2,p3])
print("\nFinal P matrix: \n",P)


