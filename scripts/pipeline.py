#!/usr/bin/env python
from __future__ import print_function, division
import cv2
import numpy as np 
from features import *
from helper import *
import skimage.io as io
import util
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import my_intrinsics as param


## Extract features from images and match
im1_path = 'capture/image3.jpg'
im2_path = 'capture/image4.jpg'
I1 = io.imread(im1_path)
I2 = io.imread(im2_path)
matches, locs1, locs2 = matchPics(I1, I2) # ORB

plotMatches(I1, I2, matches, np.flip(locs1, axis=1), np.flip(locs2, axis=1))
# process them
pts1 = locs1[matches[:,0]]
pts2 = locs2[matches[:,1]]

## Calculate F using 8-point alg
# 2. Run eight_point to compute F
#M_ = np.max([I1.shape[0], I1.shape[1]])
#F = util.eight_point(pts1, pts2, M_)

F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
# # We select only inlier points
pts1 = pts1[mask.ravel()==1]
pts2 = pts2[mask.ravel()==1]

print(F)
#displayEpipolarF(I1, I2, F)
#epipolarMatchGUI(I1, I2, F)

'''
# 3. Load points in image 1 from data/temple_coords.npz
data1 = np.load('data/temple_coords.npz')
y1 = data1['x1'] # 288 x 1
x1 = data1['y1']
data1.close()
points1 = np.hstack((x1, y1))'''
gray1 = cv2.cvtColor(I1,cv2.COLOR_RGB2GRAY)
corners = cv2.goodFeaturesToTrack(gray1,250,0.01,3)
corners = np.int0(corners)
#points3 = np.int0(corners).reshape((250,2))
points1 = corners.reshape((corners.shape[0], 2))
# 4. Run epipolar_correspondences to get points in image 2
points2 = util.epipolar_correspondences(I1, I2, F, points1)
#points4 = util.epipolar_correspondences(I1, I2, F, points3)
#plt.scatter(points2[:,0], points2[:,1])
for i in corners:
    x,y = i.ravel()
    cv2.circle(I1,(x,y),3,255,-1)

plt.imshow(I1),plt.show()
plt.show()
K1 = np.array(param.K).reshape((3,3))
K2 = np.array(param.K).reshape((3,3))
E = util.essential_matrix(F, K1, K2)
print(E)

# 5. Compute the camera projection matrix P1
I = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
P1 = K1.dot(I)
print(P1)


# 6. Run triangulate using the projection matrices
retval1, R1, t1, mask1 = cv2.recoverPose(E, points1, points2, K1)
P2 = np.dot(K1, np.hstack((R1, np.reshape(t1, (3,1)))))
pt = util.triangulate(P1, points1, P22, points2)


# 7 Scatter plot the correct 3D points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = pt[:,0]
y = pt[:,1]
z = pt[:,2]
ax.scatter(x, y, z, c='b', marker='o')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()


## Save the computed extrinsic parameters (R1,R2,t1,t2) to data/extrinsics.npz
# outfile = 'data/extrinsics.npz'
# R1 = I[:,0:3]
# t1 = np.array([I[:,3]]).T
# M2 = np.dot(np.linalg.inv(K2), P2)
# R2 = M2[:,0:3]
# t2 = np.array([M2[:,3]]).T
# np.savez(outfile, R1=R1, R2=R2, t1=t1, t2=t2)



## Re-projection Error