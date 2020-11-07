import numpy as np
import cv2
from cv2 import aruco
import matplotlib.pyplot as plt
import os
import pdb

footage_dir = 'test_footage/'
MIN_MARKERS = 8

aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
marker_detect_params = aruco.DetectorParameters_create()
flow_params = dict( winSize  = (15,15),
                    maxLevel = 2,
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
                    ) # TODO: tune flow parameters according to the paper

def main():
    fname = 'desk.MOV'
    prev_gray = None
    prev_corners = None
    prev_ids = None
    #cap = cv2.VideoCapture(os.path.join(footage_dir, fname))
    cap = cv2.VideoCapture(0) # live feed

    while (cap.isOpened()):
        # APE
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=marker_detect_params)
        
        out = frame.copy()
        # Optic Flow
        has_init_frame = prev_gray is not None
        # ape_failed = ids.shape[0] < MIN_MARKERS # TODO better define when to use optical flow
        if has_init_frame and prev_ids is not None:
            if ids is None:
                corners = []
                ids = np.empty((0,1))
            for prev_id in prev_ids:
                if prev_id not in ids:
                    p0 = np.array(prev_corners)[prev_ids == prev_id].reshape(-1,1,2)
                    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **flow_params)
                    valid_p0 = p0[st==1]
                    valid_p1 = p1[st==1]
                    if valid_p1.shape[0] == 4: # ensure optic flow found all four corners
                        # TODO: you should be able to derive the 4th corner given 3
                        for i in range(valid_p0.shape[0]):
                            start = tuple(valid_p0[i,:])
                            end = tuple(valid_p1[i,:])
                            color = (255, 255, 0)
                            thickness = 3
                            out = cv2.arrowedLine(out, start, end, color, thickness)
                        corners.append(valid_p1[np.newaxis, :, :])
                        ids = np.concatenate((ids, [prev_id]))
                    
        prev_gray = gray
        prev_corners = corners
        prev_ids = ids
        
        # TODO: implement eq 3 from paper to confine ARUCO's search region

        if ids is not None:
            frame_markers = aruco.drawDetectedMarkers(out, corners, ids)
            cv2.imshow('frame', frame_markers)
        else:
            cv2.imshow('frame', out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()