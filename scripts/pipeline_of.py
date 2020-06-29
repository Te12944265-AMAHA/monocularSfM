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
import copy

def get_random_color(a=0, b=256, c=0, d=256, e=0, f=256):
    B = np.random.randint(a, b)
    G = np.random.randint(c, d)
    R = np.random.randint(e, f)
    return (B, G, R)

class Feature:
    def __init__(self, pos, color):
        self.pos = pos 
        self.color = color

    def draw(self, img, size=6):
        cv2.circle(img,pos,size,self.color,-1)
        
    def __eq__(self, other):
        return isinstance(other, Feature) and other.pos == self.pos

    def __ne__(self, other):
        return (not isinstance(other, Feature)) or other.pos != self.pos

    def __hash__(self):
        return hash(self.pos)


class Tracker:
    def __init__(self):
        self.curr = None 
        self.prevFeatures = None 
        self.currFeatures = []
        self.colors = []
        

    def load_curr(self, curr, features):
        self.curr = curr
        self.prevFeatures = copy.deepcopy(self.currFeatures)
        for f in features:
            color = get_random_color()
            fe = Feature(f, color)
            self.currFeatures.append(fe)

    def draw(self):
        for f in self.currFeatures:
            f.draw(self.curr, 6)
        cv2.imshow('frame',self.curr)


cap = cv2.VideoCapture('capture/output_cut.mp4')
buffer = None
tracker = Tracker()

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
                  
while(cap.isOpened()):
    ret, frame = cap.read()

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    p = cv2.goodFeaturesToTrack(gray1,250,0.01,5)
    points1 = corners.reshape((corners.shape[0], 2))

    cv2.calcOpticalFlowPyrLK(imgBuffer, img, cornerBuffer)
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    cv2.imshow('frame',img)




cap.release()
cv2.destroyAllWindows()
