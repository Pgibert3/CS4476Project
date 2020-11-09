import numpy as np
import cv2
from cv2 import aruco
import glob
import os


def calibrate_charuco():
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    aruco_params = aruco.DetectorParameters_create()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)
    board = cv2.aruco.CharucoBoard_create(5, 7, .04, .03, aruco_dict)
    images = glob.glob('../test_footage/calibration_images/*.jpeg')

    all_corners = []
    all_ids = []
    imsize = None

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
        if ids is not None:
            # sub pixel detection
            for i in range(len(corners)):
                corners[i] = cv2.cornerSubPix(gray, corners[i],
                        winSize=(3,3),
                        zeroZone=(-1,-1),
                        criteria=criteria)
            retval, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(
                        corners,
                        ids,
                        gray,
                        board)

            if retval is not None and ch_corners.shape[0] >= 4:
                all_corners.append(ch_corners)
                all_ids.append(ch_ids)
        imsize = gray.shape

    camera_matrix_init = np.array([[ 1000.,    0., imsize[0]/2.],
                              [    0., 1000., imsize[1]/2.],
                              [    0.,    0.,           1.]])
    
    dist_coeffs_init = np.zeros((5,1))
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_ASPECT_RATIO)
    retval, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
            charucoCorners=all_corners,
            charucoIds=all_ids,
            board=board,
            imageSize=imsize,
            flags=flags,
            criteria=criteria,
            cameraMatrix=camera_matrix_init,
            distCoeffs=dist_coeffs_init)
    
    return camera_matrix, dist_coeffs






    


    

