#!/usr/bin/env python
from __future__ import print_function
import cv2
import numpy as np
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError



class image_converter(object):
    def __init__(self):
        self.i = 0
        self.bridge = CvBridge()
        # Subscribes ROS images
        self.image_sub = rospy.Subscriber("/image",Image,self.callback)

    def callback(self,data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        print(cv_image.shape)
        cv_image_flipped = cv2.flip(cv_image, 0)
        out.write(cv_image_flipped)

        cv2.imshow('frame',cv_image_flipped)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.imwrite('capture/image%d.jpg'%(self.i), cv_image_flipped)
            self.i += 1


def main():
    ic = image_converter()
    rospy.init_node('image_converter', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

# if __name__ == '__main__':
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     out = cv2.VideoWriter('capture/output.avi',fourcc, 20.0, (1024,512))
#     main()
#     print('here!!!')
#     out.release()
#     cv2.destroyAllWindows()

