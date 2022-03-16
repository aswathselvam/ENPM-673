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
A = np.empty([0,12],dtype=float)
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

print("\nP is in the last column of V:\n",  V[:,-1])
p1 = V[0:4,-1]
p2 = V[4:8,-1]
p3 = V[8:12,-1]

P = np.array([p1,p2,p3])
# P = P * np.sign(np.linalg.det(P[:3,:3]))
print("\nFinal P matrix: \n",P)


#------------C matrix---------------#
#P*C=0
U, D, V = np.linalg.svd(P)
m, n = P.shape
P_reconstructed = U[:,:n] @ np.diag(D) @ V[:m,:]
C = V[:,n-1]/V[-1,-1]; C = C[:-1]
# C = V[:,n-1]; # C = C[:-1]
# print("\nC is in the last column of V:\n",  C)

#-----------K, R matrix----------------#

#M=KR
#K is upper Triangular matrix 
#R is othogonal matrix

def RQ(M):
    [Q,R] = np.linalg.qr(np.flipud(M.T))
    R = np.flipud(R.T)
    R = np.fliplr(R)
    Q = Q.T
    Q = np.flipud(Q)
    return R, Q

K,R = RQ(P[:3,:3])
D = np.diag(np.sign(np.diag(K)))
K = K * D
R = D * R
K = K/K[-1,-1]

print("K matrix: \n", K)

#-------------- t -----------------#
t = -R * C
print("Translation is: \n", t)




"""
For reference:

MATLAB Code:


img_pts = [757, 13; 758, 15; 758,86 ; 759,66 ; 1190,172 ; 329,1041; 1204,850; 340,59];
world_pts = [0, 0,0; 0, 3, 0; 0,7,0; 0,11,0; 7,1,0; 0,11,7; 7,9,0; 0,1,7];


[P,error]=estimateCameraMatrix(img_pts,world_pts);
P = P';
P = P * sign(det(P(1:3, 1:3)));
[K, R] = rq(P(1:3,1:3));
D = diag(sign(diag(K)));
K = K*D
K = K/K(end,end)

function [R,Q] = rq(M)
    [Q,R] = qr(rot90(M,3));
    R = rot90(R,2)';
    Q = rot90(Q);
end








"""