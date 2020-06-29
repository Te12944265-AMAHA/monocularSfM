import numpy as np
import helper as hlp
import skimage.io as io
import submission as sub
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d.art3d import PolyCollection

# Load image, cad, point x, 3d point X
infile = np.load('../data/pnp.npz', allow_pickle=True)
image = infile['image']
cad = infile['cad']
x = infile['x']
X = infile['X']
cad_pts = cad[0][0][0]
cad_idx = cad[0][0][1]

# Compute P, and then K, R, t
P = sub.estimate_pose(x, X)
K, R, t = sub.estimate_params(P)

# Use P to project X onto the image
X_h = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
x_p = np.dot(P, X_h.T).T
denom = np.tile(x_p[:,2],(2,1)).T
xp = x_p[:,0:2]/denom

# Plot x and the projected 3d points, overlay
implot = plt.imshow(image)
plt.scatter(x=xp[:,0], y=xp[:,1],  s=400, facecolor='none', edgecolor='black')
plt.scatter(x=x[:,0], y=x[:,1], c='yellow', s=15)
plt.show()

# Draw the CAD rotated by R
cad_r = np.dot(R, cad_pts.T).T
fig = Axes3D(plt.figure())
all = []
for i in range(cad_idx.shape[0]):
    a, b, c = cad_idx[i]
    pt1, pt2, pt3 = cad_r[a-1], cad_r[b-1], cad_r[c-1]
    verts = [pt1, pt2, pt3]
    all.append(verts)
mesh = Poly3DCollection(all, alpha=0.5, color='m', linewidths=0.3)
fig.add_collection3d(mesh)
plt.show()

# Project the CAD's all vertices on the image, overlay
cad_ = np.concatenate((cad_pts, np.ones((cad_pts.shape[0], 1))), axis=1)
cad_p = np.dot(P, cad_.T).T
denom = np.tile(cad_p[:,2],(2,1)).T
cadp = cad_p[:,0:2]/denom
fig, ax = plt.subplots()
plt.imshow(image)
all = []
for i in range(cad_idx.shape[0]):
    a, b, c = cad_idx[i]
    pt1, pt2, pt3 = cadp[a-1], cadp[b-1], cadp[c-1]
    verts = [pt1, pt2, pt3]
    all.append(verts)
mesh = PolyCollection(all, alpha=0.5, color='m', linewidths=0.3)
ax.add_collection(mesh)
plt.show()