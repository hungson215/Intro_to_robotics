from myro import *
import numpy as np
import cv2



def detectBlobs(im):
    #mask = cv2.inRange(im, np.array([0, 100, 100],np.uint8), np.array([10, 255, 255],np.uint8))
    #thresh = cv2.bitwise_and(mask, mask, mask =mask)
    #ret,thresh = cv2.threshold(thresh,1,255,cv2.THRESH_BINARY_INV)
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold =0
    params.maxThreshold = 255
    params.filterByColor = True
    params.blobColor =0
    params.filterByArea = True
    params.minArea = 4000
    params.maxArea = 999999
    params.filterByCircularity = True
    params.minCircularity = 0
    params.maxCircularity = 1
    params.filterByInertia= True
    params.minInertiaRatio = 0
    params.maxInertiaRatio = 1
    params.filterByConvexity = True
    params.minConvexity = 0
    params.maxConvexity = 1
    detector = cv2.SimpleBlobDetector(params)
    keypoints = detector.detect(im)
    return keypoints

def intruderDetect():
    detect_flag = False
    degree = 0
    while not detect_flag:
        if degree > 360:
            print "Nothing after turning 360"
            break
        picture = takePicture("color")
        savePicture(picture,"test.jpg")
        im = cv2.imread('test.jpg')
        hsv = cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
        keypoints = detectBlobs(hsv)
        if len(keypoints) != 0:
            print "Detected!"
            detect_flag = True
            color=(0,255,0)
            im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), color, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.imwrite('testoutput.jpg',im_with_keypoints)
            return keypoints
        else:
            print "Nothing detected"
        turnBy(60, "deg")
        degree += 60

def intruderDetectBlob():
    detect_flag = False
    degree = 0
    configureBlob(y_low=100, y_high=200,u_low=80,u_high=120,v_low=180,v_high=255)
    while not detect_flag:
        if degree > 360:
            print "Nothing after turning 360"
            break
        picture = takePicture("blob")
        savePicture(picture,"test.jpg")
        im = cv2.imread('test.jpg')
        #hsv = cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
        keypoints = detectBlobs(im)
        if len(keypoints) != 0:
            print "Detected!"
            detect_flag = True
            color=(0,255,0)
            im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), color, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.imwrite('testoutput.jpg',im_with_keypoints)
            return keypoints
        else:
            print "Nothing detected"
        turnBy(60, "deg")
        degree += 60

def testBlob():
    configureBlob(y_low=100, y_high=200,u_low=80,u_high=120,v_low=180,v_high=255)
    picture = takePicture("blob")
    savePicture(picture,"test.jpg")
    im = cv2.imread('test.jpg')
    #hsv = cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
    keypoints = detectBlobs(im)
    color=(0,255,0)
    im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), color, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite('testoutput.jpg',im_with_keypoints)
    show(im)
def main():
    keypoints = intruderDetect()
