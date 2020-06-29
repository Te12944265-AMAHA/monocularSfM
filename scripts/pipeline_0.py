#!/usr/bin/env python
from __future__ import print_function, division
import cv2
import numpy as np
import helper as hlp
import skimage.io as io
import util
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Load the two temple images and the points from data/some_corresp.npz
I1 = io.imread('data/im1.png')
I2 = io.imread('data/im2.png')
data = np.load('data/some_corresp.npz')
pts1 = data["pts1"]
pts2 = data["pts2"]
data.close()

# 2. Run eight_point to compute F
M_ = np.max([I1.shape[0], I1.shape[1]])
F = util.eight_point(pts1, pts2, M_)
print(F)
#hlp.displayEpipolarF(I1, I2, F)
#hlp.epipolarMatchGUI(I1, I2, F)

# 3. Load points in image 1 from data/temple_coords.npz
data1 = np.load('data/temple_coords.npz')
y1 = data1['x1'] # 288 x 1
x1 = data1['y1']
data1.close()
points1 = np.hstack((x1, y1))
# 4. Run epipolar_correspondences to get points in image 2
points2 = util.epipolar_correspondences(I1, I2, F, points1)
plt.scatter(points2[:,0], points2[:,1])
plt.scatter(points1[:,0], points1[:,1])
plt.show()
cam = np.load('data/intrinsics.npz')
K1 = cam['K1']
K2 = cam['K2']
cam.close()
E = util.essential_matrix(F, K1, K2)
print(E)

# 5. Compute the camera projection matrix P1
I = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
P1 = K1.dot(I)
print(P1)

# 6. Use camera2 to get 4 camera projection matrices P2
M = hlp.camera2(E)
P2_1 = np.dot(K2, M[:,:,0])
P2_2 = np.dot(K2, M[:,:,1])
P2_3 = np.dot(K2, M[:,:,2])
P2_4 = np.dot(K2, M[:,:,3])
P_all = np.array([P2_1,P2_2,P2_3,P2_4])
print(P_all)

# 7. Run triangulate using the projection matrices
pt1 = util.triangulate(P1, points1, P2_1, points2)
pt2 = util.triangulate(P1, points1, P2_2, points2)
pt3 = util.triangulate(P1, points1, P2_3, points2)
pt4 = util.triangulate(P1, points1, P2_4, points2)
pt_all = np.array([pt1,pt2,pt3,pt4])

# 8. Figure out the correct P2
# Find the configuration where most points are in front of both cameras
s1 = np.sum(pt1[:,-1] >= 0)
s2 = np.sum(pt2[:,-1] >= 0)
s3 = np.sum(pt3[:,-1] >= 0)
s4 = np.sum(pt4[:,-1] >= 0)
S = np.array([s1,s2,s3,s4])
print('Number of points with positive z coordinate: ', S)
index = np.argmax(S)
print(index)
P2 = P_all[index] # Correct P2 and 3d points
pt = pt_all[index]

# 9. Scatter plot the correct 3D points
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