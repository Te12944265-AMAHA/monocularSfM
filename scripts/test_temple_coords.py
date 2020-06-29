import numpy as np
import helper as hlp
import skimage.io as io
import submission as sub
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. Load the two temple images and the points from data/some_corresp.npz
I1 = io.imread('../data/im1.png')
I2 = io.imread('../data/im2.png')
data = np.load('../data/some_corresp.npz')
pts1 = data["pts1"]
pts2 = data["pts2"]
data.close()

# 2. Run eight_point to compute F
M_ = np.max([I1.shape[0], I1.shape[1]])
F = sub.eight_point(pts1, pts2, M_)
print(F)
#hlp.displayEpipolarF(I1, I2, F)
#hlp.epipolarMatchGUI(I1, I2, F)

# 3. Load points in image 1 from data/temple_coords.npz
data1 = np.load('../data/temple_coords.npz')
y1 = data1['x1'] # 288 x 1
x1 = data1['y1']
data1.close()
points1 = np.hstack((x1, y1))
# 4. Run epipolar_correspondences to get points in image 2
points2 = sub.epipolar_correspondences(I1, I2, F, points1)
plt.scatter(points2[:,0], points2[:,1])
cam = np.load('../data/intrinsics.npz')
K1 = cam['K1']
K2 = cam['K2']
cam.close()
E = sub.essential_matrix(F, K1, K2)
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
pt1 = sub.triangulate(P1, points1, P2_1, points2)
pt2 = sub.triangulate(P1, points1, P2_2, points2)
pt3 = sub.triangulate(P1, points1, P2_3, points2)
pt4 = sub.triangulate(P1, points1, P2_4, points2)
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

# 10. Save the computed extrinsic parameters (R1,R2,t1,t2) to data/extrinsics.npz
outfile = '../data/extrinsics.npz'
R1 = I[:,0:3]
t1 = np.array([I[:,3]]).T
M2 = np.dot(np.linalg.inv(K2), P2)
R2 = M2[:,0:3]
t2 = np.array([M2[:,3]]).T
np.savez(outfile, R1=R1, R2=R2, t1=t1, t2=t2)



# Re-projection Error
def calc_reproj_err(pt, pt3d, P):
    N = np.shape(pt)[0]
    pt3d_h = np.concatenate((pt3d, np.ones((pt3d.shape[0], 1))), axis=1)
    pt_p = np.dot(P, pt3d_h.T).T
    denom = np.tile(pt_p[:,2],(2,1)).T
    pt_pp = pt_p[:,0:2]/denom
    dist = np.sum(np.sqrt(np.sum((pt_pp-pt) ** 2, axis=1)))/N
    return dist

data = [[' ', 'pts1', 'pts2'],
        ['P2_1',0,0],
        ['P2_2',0,0],
        ['P2_3',0,0],
        ['P2_4',0,0]]
for i in range(4):
    e1 = calc_reproj_err(points1, pt_all[i], P1)
    e2 = calc_reproj_err(points2, pt_all[i], P_all[i])
    data[i+1][1] = e1 
    data[i+1][2] = e2

for i in range(len(data)):
    if i == 0:
      print('{:<10s}{:<12s}{:<12s}'.format(data[i][0],data[i][1],data[i][2]))
    else:
      print('{:<10s}{:<8.4f}      {:<8.4f}'.format(data[i][0],data[i][1],data[i][2]))

