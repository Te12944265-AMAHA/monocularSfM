#!/usr/bin/env python
from __future__ import print_function, division
import numpy as np 
import helper as hlp
import cv2
from scipy.spatial import distance
from scipy.ndimage import gaussian_filter
from scipy.linalg import rq
import matplotlib.pyplot as plt

"""
Q3.1.1 Eight Point Algorithm
       [I] pts1, points in image 1 (Nx2 matrix)
           pts2, points in image 2 (Nx2 matrix)
           M, scalar value computed as max(H1,W1)
       [O] F, the fundamental matrix (3x3 matrix)
"""
def eight_point(pts1, pts2, M):
    #pts1 = np.flip(pts1, axis=1)
    #pts2 = np.flip(pts2, axis=1)
    # Turn into homogeneous coordinate
    N = np.shape(pts1)[0]
    ones = np.ones((N,1))
    pts1_h = np.concatenate((pts1, ones), axis=1)
    pts2_h = np.concatenate((pts2, ones), axis=1)
    # Normalize coordinate
    T = np.array([[1/M, 0, 0], [0, 1/M, 0], [0, 0, 1]])
    pts1_n = np.dot(T, pts1_h.T).T
    pts2_n = np.dot(T, pts2_h.T).T 
    # Construct A
    A = np.zeros((N, 9))
    for i in range(N):
        x1 = pts1_n[i,:]
        x2 = pts2_n[i,:]
        first = x1[0] * x2 
        second = x1[1] * x2 
        Ai = np.concatenate((first, second), axis=0)
        Ai = np.concatenate((Ai, x2), axis=0)
        A[i,:] = Ai
    # SVD of A
    u, s, vt = np.linalg.svd(A)
    # F ~ col of V w/ least eigval (last row of vt)
    F = vt[-1, :]
    F = np.reshape(F, (3, 3))
    # Enforce rank(F) = 2
    uf, sf, vtf = np.linalg.svd(F)
    sf[-1] = 0
    sf_new = np.diag(sf)
    F = np.dot(uf, sf_new)
    F = np.dot(F, vtf)
    # Refine F
    F = hlp.refineF(F, pts1/M, pts2/M)
    # Unormalize F by T'FT
    F = np.dot(T.T, F)
    F = np.dot(F, T)
    return F


"""
Helper function for Epipolar Correspondences
Find the best match of a pixel along epipolar line
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           p1, [x1, y1], the pixel in im1 we are trying to match
           pts2, candidate points in image 2
           wsize, window size (odd number)
           pdist, max distance between p1 and p2
       [O] p2, [x2, y2], best match of p1 in image 2
"""
def find_closest(im1, im2, p1, pts2, wsize, pdist):
    delta = wsize//2
    h2 = im2.shape[0]
    w2 = im2.shape[1]
    x1, y1 = p1 
    window1 = im1[y1-delta:y1+delta+1, x1-delta:x1+delta+1]
    #window1_ = window1.flatten()
    dist = None
    bestMatch = None
    # Scan through all candidates in im2 and find the best match
    for j in range(pts2.shape[0]):
        p2 = pts2[j,:]
        x2, y2 = p2 # (x2, y2)
        if x2-delta<0 or x2+delta>=w2 or y2-delta<0 or y2+delta>=h2:
            pass
        elif np.linalg.norm(p1-p2) > pdist:
            pass
        else:
            window2 = im2[y2-delta:y2+delta+1, x2-delta:x2+delta+1]
            tmpDist = np.sum(np.square(gaussian_filter(window1 - window2, sigma=1.0)))
            #window2_ = window2.flatten()
            if dist == None or tmpDist < dist:
                dist = tmpDist
                bestMatch = pts2[j,:]
    return bestMatch


"""
Q3.1.2 Epipolar Correspondences
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           F, fundamental matrix from image 1 to image 2 (3x3 matrix)
           pts1, points in image 1 (Nx2 matrix)
       [O] pts2, points in image 2 (Nx2 matrix)
"""
def epipolar_correspondences(im1, im2, F, pts1):
    N = pts1.shape[0]
    ones = np.ones((N,1))
    pts1_h = np.concatenate((pts1, ones), axis=1)
    l2 = np.dot(F, pts1_h.T) # 3 x N
    width = im2.shape[1] 
    x = np.array([np.arange(0, width)]).T # width * 1
    wsize = 19
    pdist = 40
    pts2 = np.zeros_like(pts1)
    for i in range(N): 
        # Scan the epipolar line to find the best match
        p1 = pts1[i,:] # (x1, y1)
        a, b, c = l2[:,i]
        y = np.round((- c - a * x) / b).astype(np.int) # width * 1
        candidates = np.concatenate((x, y), axis=1)
        bestMatch = find_closest(im1, im2, p1, candidates, wsize, pdist)
        pts2[i,:] = bestMatch
    return pts2


"""
Q3.1.3 Essential Matrix
       [I] F, the fundamental matrix (3x3 matrix)
           K1, camera matrix 1 (3x3 matrix)
           K2, camera matrix 2 (3x3 matrix)
       [O] E, the essential matrix (3x3 matrix)
"""
def essential_matrix(F, K1, K2):
    E = np.dot(K2.T, F) 
    E = np.dot(E, K1)
    return E


"""
Q3.1.4 Triangulation
       [I] P1, camera projection matrix 1 (3x4 matrix)
           pts1, points in image 1 (Nx2 matrix)
           P2, camera projection matrix 2 (3x4 matrix)
           pts2, points in image 2 (Nx2 matrix)
       [O] pts3d, 3D points in space (Nx3 matrix)
"""
def triangulate(P1, pts1, P2, pts2):
    N = pts1.shape[0]
    P1_1 = P1[0,:] # 4, 
    P1_2 = P1[1,:]
    P1_3 = P1[2,:]
    P2_1 = P2[0,:]
    P2_2 = P2[1,:]
    P2_3 = P2[2,:]
    pts3d = np.zeros((N, 3))
    # For each point in the plane, find a match in 3d
    for i in range(N):
        x1, y1 = pts1[i,:]
        x2, y2 = pts2[i,:]
        first_1 = y1 * P1_3 - P1_2  # Pn_m is the m-th row of Pn
        second_1 = x1 * P1_3 - P1_1
        first_2 = y2 * P2_3 - P2_2 
        second_2 = x2 * P2_3 - P2_1
        A = np.vstack((first_1, second_1, first_2, second_2))
        # print(A)
        # SVD of A
        u, s, vt = np.linalg.svd(A)
        # X ~ col of V w/ least eigval (last row of vt)
        X = vt[-1, 0:3]/vt[3,3]
        pts3d[i,:] = X
    return pts3d


"""
Q3.2.1 Image Rectification
       [I] K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] M1 M2, rectification matrices (3x3 matrix)
           K1p K2p, rectified camera matrices (3x3 matrix)
           R1p R2p, rectified rotation matrices (3x3 matrix)
           t1p t2p, rectified translation vectors (3x1 matrix)
"""
def rectify_pair(K1, K2, R1, R2, t1, t2):
    # Compute the optical center 
    c1 = -np.dot(np.linalg.inv(np.dot(K1, R1)), np.dot(K1, t1)) # 3 x 1
    c2 = -np.dot(np.linalg.inv(np.dot(K2, R2)), np.dot(K2, t2))
    # Compute the new rotation matrix R~
    r1 = (c1 - c2) / np.linalg.norm(c1 - c2)
    r1.transpose()[0]
    r2 = np.cross(r1.T[0], R1[2,:].T)
    r3 = np.cross(r2, r1.T[0])
    r1.transpose()[0]
    R_n = np.array([r1.transpose()[0], r2, r3])
    R1p = R_n 
    R2p = R_n
    # Compute the new intrinsic params as K'_1 = K'_2 = K_2
    K_n = K2 
    K1p = K_n
    K2p = K_n
    # Compute the new translation vectors as t1 = -R~c1 and t2..
    t1p = -np.dot(R_n, c1)
    t2p = -np.dot(R_n, c2)
    # Compute the rectification matrices
    M1 = np.dot(np.dot(K1p, R1p), np.linalg.inv(np.dot(K1, R1)))
    M2 = np.dot(np.dot(K2p, R2p), np.linalg.inv(np.dot(K2, R2)))
    return M1, M2, K1p, K2p, R1p, R2p, t1p, t2p


"""
Q3.2.2 Disparity Map
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           max_disp, scalar maximum disparity value
           win_size, scalar window size value
       [O] dispM, disparity map (H1xW1 matrix)
"""
def get_disparity(im1, im2, max_disp, win_size):
    r, c = im1.shape
    dispM = np.zeros_like(im1)
    w = (win_size-1)//2
    # For every pixel, find minimum distance (disparity)
    for i in range(max_disp+w+2, r-w-max_disp-2):
        for j in range(w+max_disp+2, c-w-2):
            distList = np.zeros(max_disp)
            for d in range(max_disp):
                win1 = im1[i-w:i+w+1, j-w:j+w+1]
                win2 = im2[i-w:i+w+1, j-w-d:j+w+1-d]
                tmpDist = np.linalg.norm(win1 + win2)
                distList[d] = tmpDist
            minIndex = np.argmin(distList)
            dispM[i,j] = minIndex
        print(i)
    print('done!!!!!!!!!!!!!')
    return dispM


"""
Q3.2.3 Depth Map
       [I] dispM, disparity map (H1xW1 matrix)
           K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] depthM, depth map (H1xW1 matrix)
"""
def get_depth(dispM, K1, K2, R1, R2, t1, t2):
    c1 = -np.dot(np.linalg.inv(np.dot(K1, R1)), np.dot(K1, t1)) # 3 x 1
    c2 = -np.dot(np.linalg.inv(np.dot(K2, R2)), np.dot(K2, t2))
    b = np.linalg.norm(c1 - c2)
    f = K1[0,0]
    with np.errstate(divide='ignore'):
        depthM = np.divide(b*f, dispM)
    depthM[depthM == np.inf] = 0
    depthM[depthM == -np.inf] = 0
    return depthM


"""
Q3.3.1 Camera Matrix Estimation
       [I] x, 2D points (Nx2 matrix)
           X, 3D points (Nx3 matrix)
       [O] P, camera matrix (3x4 matrix)
"""
def estimate_pose(x, X):
    N = np.shape(x)[0]
    ones = np.ones((N,1))
    X_h = np.concatenate((X, ones), axis=1)
    # Construct A
    A = np.zeros((N * 2, 12))
    zero = np.zeros(4)
    for i in range(N):
        xi_x, xi_y = x[i,:]
        Xi = X_h[i,:]
        third1 = -xi_x * Xi
        third2 = -xi_y * Xi
        Ai_1 = np.hstack((Xi, zero, third1))
        Ai_2 = np.hstack((zero, Xi, third2))
        Ai = np.vstack((Ai_1, Ai_2))
        A[i*2:(i+1)*2,:] = Ai
    # SVD of A
    u, s, vt = np.linalg.svd(A)
    # P ~ col of V w/ least eigval (last row of vt)
    P = vt[-1, :]
    P = np.reshape(P, (3, 4))
    return P


"""
Q3.3.2 Camera Parameter Estimation
       [I] P, camera matrix (3x4 matrix)
       [O] K, camera intrinsics (3x3 matrix)
           R, camera extrinsics rotation (3x3 matrix)
           t, camera extrinsics translation (3x1 matrix)
"""
def estimate_params(P):
    # Compute camera center c by svd
    u, s, vt = np.linalg.svd(P)
    # c ~ col of V w/ least eigval (last row of vt)
    c = np.array([vt[-1, 0:3]/vt[-1,-1]]).T  # (3,)
    # Compute K and R by QR decomposition
    M = P[:,0:3]
    K, R = rq(M)
    # Compute t = -Rc
    t = -np.dot(R, c)
    return K, R, t


def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2


def foo(pts1, pts2, img1, img2):
    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
    lines1 = lines1.reshape(-1,3)
    img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)
    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
    lines2 = lines2.reshape(-1,3)
    img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)
    plt.subplot(121),plt.imshow(img5)
    plt.subplot(122),plt.imshow(img3)
    plt.show()


def calc_reproj_err(pt, pt3d, P):
    N = np.shape(pt)[0]
    pt3d_h = np.concatenate((pt3d, np.ones((pt3d.shape[0], 1))), axis=1)
    pt_p = np.dot(P, pt3d_h.T).T
    denom = np.tile(pt_p[:,2],(2,1)).T
    pt_pp = pt_p[:,0:2]/denom
    dist = np.sum(np.sqrt(np.sum((pt_pp-pt) ** 2, axis=1)))/N
    return dist


def summarize_reproj_err(points1, points2, pt_all, P1, P_all):
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
            print('{:<10s}{:<8.4f}{:<8.4f}'.format(data[i][0],data[i][1],data[i][2]))


def skew(x):
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])