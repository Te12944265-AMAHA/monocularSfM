#!/usr/bin/env python
from __future__ import print_function, division
import cv2
import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanade(It, It1, rect):
    # Input: 
    #   It: template image
    #   It1: Current image
    #   rect: Current position of the object
    #   (top left, bot right coordinates: x1, y1, x2, y2)
    # Output:
    #   p: movement vector dx, dy
    
    # set up the threshold
    threshold = 0.00001
    err = None
    maxIters = 100
    p = np.zeros(2)         
    x1,y1,x2,y2 = rect
    eps = 0.15
    # put your implementation here
    rt, ct = It.shape
    rt1, ct1 = It1.shape
    linetr, linetc = np.arange(rt), np.arange(ct)
    linet1r, linet1c = np.arange(rt1), np.arange(ct1)
    linet = RectBivariateSpline(linetr, linetc, It)
    linet1 = RectBivariateSpline(linet1r, linet1c, It1)

    for iter in range(maxIters):
        # Template
        xt, yt = np.arange(x1, x2+eps), np.arange(y1, y2+eps)
        mxt, myt = np.meshgrid(xt, yt)
        interpt = linet.ev(myt, mxt)
        # Current
        xt1, yt1 = np.arange(x1+p[0], x2+eps+p[0]), np.arange(y1+p[1], y2+eps+p[1])
        mxt1, myt1 = np.meshgrid(xt1, yt1)
        interpt1 = linet1.ev(myt1, mxt1)
        # Calculate A and b
        gradx = linet1.ev(myt1, mxt1, dx=0, dy=1).flatten()
        grady = linet1.ev(myt1, mxt1, dx=1, dy=0).flatten()
        A = np.array([gradx, grady]).T 
        b = (interpt - interpt1).flatten()
        # Calculate delta p
        dp, _, __, ___ = np.linalg.lstsq(A, b, rcond=None)
        # Update W, err
        err = np.linalg.norm(dp) ** 2
        if err < threshold: 
            break
        p += dp.flatten()
    return p

