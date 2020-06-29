#!/usr/bin/env python
from __future__ import print_function
import cv2
import numpy as np
import scipy.optimize
import numpy.linalg as la
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.ndimage import gaussian_filter
from scipy.linalg import rq

import skimage.color
from helper import briefMatch
from helper import computeBrief
from helper import corner_detection


# Use ORB to detect and match features
def matchPics(I1, I2):
    # I1, I2 : Images to match
    # Convert Images to GrayScale
    I1 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
    I2 = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)
    # Detect Features in Both Images
    locs1 = corner_detection(I1)
    locs2 = corner_detection(I2)
    # Obtain descriptors for the computed feature locations
    desc1, locs1 = computeBrief(I1, locs1)
    desc2, locs2 = computeBrief(I2, locs2)
    # Match features using the descriptors

    locs1 = np.flip(locs1, axis=1)
    locs2 = np.flip(locs2, axis=1)
    matches = briefMatch(desc1, desc2, 0.8)
    print(matches.shape)
    return matches, locs1, locs2