import numpy as np
import cv2
from cv2 import aruco
import os
import glob
import pdb
from tracker import Tracker


def main():
    # get poses of object at several different angles
    camera_params = np.load('camera_params.npy', allow_pickle=True)
    marker_len = 0.0365125
    camera_matrix = camera_params[0]
    dist_coeffs = camera_params[1]
    axis_len = marker_len / 2
    tracker = Tracker(marker_len=marker_len, camera_params=camera_params)

    fnames = glob.glob("../test_footage/comb_calibrate/*.jpeg")

    rmats = {}
    tvecs = {}

    for f in fnames:
        img = cv2.imread(f, -1)
        rvec, tvec, obj_points, corners, ids = tracker.track_frame(img, standalone=True)
        # draw
        if ids is not None:
            for i in range(0, tvec.shape[0]):
                m_id = ids[i,0]
                if m_id not in rmats:
                    rmats[m_id] = []
                    tvecs[m_id] = []
                rmats[m_id].append(cv2.Rodrigues(rvec[i])[0])
                tvecs[m_id].append(tvec[i])

            for i in range(tvec.shape[0]):
                img = cv2.aruco.drawAxis(img, camera_matrix, dist_coeffs, rvec[i], tvec[i], axis_len)
        # cv2.imshow("Tracker", cv2.resize(img, (img.shape[1]//4, img.shape[0]//4)))
        # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.waitKey(1)

    # build and solve matrix for c
    c = {}
    for m_id in rmats.keys():
        c[m_id] = None
        for i in range(len(rmats)-1):
            A = rmats[m_id][i] - rmats[m_id][i+1]
            b = (tvecs[m_id][i] - tvecs[m_id][i+1]).T
            x = np.linalg.lstsq(A, b)[0]
            if c[m_id] is None:
                c[m_id] = x
            else:
                c = np.average(np.concatenate((c[m_id], x),axis=1), axis=1)[:, np.newaxis]

    # call tracker with origin
    for f in fnames:
        img = cv2.imread(fnames[0], -1)
        rvec, tvec, obj_points, corners, ids = tracker.track_frame(img, standalone=True)
        if ids is not None:
            for i in range(0, tvec.shape[0]):
                m_id = ids[i,0]
                imagePoints, _ = cv2.projectPoints(c[m_id], rvec[i], tvec[i], camera_matrix, dist_coeffs)
                x = imagePoints[0,0,0]
                y = imagePoints[0,0,1]
                img = cv2.circle(img, (int(x), int(y)), 5, (255,0,0), -1)
            cv2.imshow("Tracker", cv2.resize(img, (img.shape[1]//4, img.shape[0]//4)))
            cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)


if __name__ == '__main__':
    main()