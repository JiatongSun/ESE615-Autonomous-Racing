#!/usr/bin/env python3

import cv2
import numpy as np
import os
import glob
import time

from infer import trt_infer

CAM_DIM = [540, 960]
TRT_DIM = [180, 320]

def connect_cam():
    cam = cv2.VideoCapture("v4l2src device=/dev/video2 extra-controls=\"c,exprosure_auto=3\" ! video/x-raw, width=960, height=540 ! videoconvert ! video/x-raw, format=BGR ! appsink")
    time_old = time.time()
    if cam.isOpened():
        cv2.namedWindow("Realsense", cv2.WINDOW_AUTOSIZE)
        print("Press Q to quit...")

        while True:
            time_now = time.time()
            ret, image = cam.read()
            
            print(1/(time_now - time_old), 'Hz')
            time_old = time_now
            
            cv2.imshow('Realsense', image)
            if cv2.waitKey(1) == ord('q'):
                cv2.destroyAllWindows()
                break

    else:
        print('Camera open failed')


def calibration(images, dims):

    CHECKERBOARD = dims

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
    threedpoints = []
    twodpoints = []
 
    objectp3d = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objectp3d[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

    prev_img_shape = None

    for filename in images:
        image = cv2.imread(filename)
        grayColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(grayColor, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

        if ret == True:
            threedpoints.append(objectp3d)
 
            corners2 = cv2.cornerSubPix(grayColor, corners, (11, 11), (-1, -1), criteria)
            twodpoints.append(corners2)
 
            image = cv2.drawChessboardCorners(image, CHECKERBOARD, corners2, ret)
    
        cv2.imshow('img', image)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

    h, w = image.shape[:2]

    ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(threedpoints, twodpoints, grayColor.shape[::-1], None, None)

    print(" Camera matrix:")
    print(matrix)
 
    print("\n Distortion coefficient:")
    print(distortion)
     
    print("\n Rotation Vectors:")
    print(r_vecs)
     
    print("\n Translation Vectors:")
    print(t_vecs)

    return matrix, distortion, r_vecs, t_vecs


def get_mounting_height(K, y, X_car):

    f_y = K[1, 1]
    y_0 = K[1, 2]
    # X_car =  0.4 #in meters
    # y = 500 # in image plane

    mounting_height = (y - y_0) * X_car/f_y

    return mounting_height


def get_distance(K, x, y, H_mount):

    f_x = K[0, 0]
    f_y = K[1, 1]
    x_0 = K[0, 2]
    y_0 = K[1, 2]
    # X_car =  0.4 #in meters
    # y = 500 # in image plane

    X_car = H_mount/(y - y_0) * f_y
    Y_car = (x - x_0) * X_car /f_x

    return X_car, Y_car

def open_image():
    image = cv2.imread('distance/cone_unknown.png')
    cv2.imshow('img', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_lane(images):

    for filename in images:
        
        image = cv2.imread(filename)

        hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_thresh = np.array([20, 70, 75], dtype = 'uint8')
        upper_thresh = np.array([30, 255, 255], dtype = 'uint8')

        mask = cv2.inRange(hsvImage, lower_thresh, upper_thresh)

        kernel = np.ones((5,5), np.uint8)
        # mask = cv2.dilate(mask, kernel)
        # mask = cv2.erode(mask, kernel)
        # mask = cv2.dilate(mask, kernel)

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(image, contours, -1, (0,255,0), 3)
        cv2.imshow('img', image)
        cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == '__main__':

    distance_images = glob.glob('../dist/*.png')
    lane_images = glob.glob('../lane/*.png')

    calibration_images = glob.glob('../calibration/*.png')
    checkerboard_dims = (6, 8)


    # 1. Connect Camera
    print('\nCapturing Images...')
    connect_cam()


    # 2. Get Intrinic Parameters
    print('\nStart Intrinsic Parameters Calculation!')
    K, _, R, t = calibration(calibration_images, checkerboard_dims)


    # 3. Get Mounting Height
    print('\nStart Mounting Height Calculation!')
    x, y = 665, 500 # from image plane {known image}
    X_car = 0.4 # Given, in meters

    H_mount = get_mounting_height(K, y, X_car)
    print('Mounting Height: {} m'.format(round(H_mount, 3)))


    # 4. Get Distances from Car
    print('\nStart Cone Distance Estimation!')
    x, y = 600, 415 # from image plane {unknown image}
    X_car, Y_car = get_distance(K, x, y, H_mount)
    print('X Distance from Car: {} m'.format(round(X_car, 3)))
    print('Y Distance from Car: {} m'.format(round(Y_car, 3)))


    # 5. Get Lanes
    print('\nStart Lane Detection!')
    get_lane(lane_images)

    # 6. Get Distance from TensorRT Model
    print('\nStart Object Detection!')
    bbox = trt_infer()
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    x, y = (x1 + x2) / 2, y2  # use center of bottom line of bounding box
    x = x * CAM_DIM[1] / TRT_DIM[1]
    y = y * CAM_DIM[0] / TRT_DIM[0]
    X_car, Y_car = get_distance(K, x, y, H_mount)
    print('X Distance from Car: {} m'.format(round(X_car, 3)))
    print('Y Distance from Car: {} m'.format(round(Y_car, 3)))


