import numpy as np
import cv2
from cv2 import aruco
import os
import pdb
from calibrate import calibrate_charuco


class Tracker:
    def __init__(self, aruco_dict=None, aruco_params=None, flow_params=None, marker_len=.02, camera_params=None):
        '''
        A tracking model based on the 3-stage pipeline proposed in oculus's dodecapen research
        paper.

        Params:
        aruco_dict - the dictionary of aruco markers in use
        aruco_params - the detection parameters passed to aruco for marker tracking
        flow_params - the paramaters passed to the optical flow algorithm
        '''
        if aruco_dict is None:
            aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        self.aruco_dict = aruco_dict

        if aruco_params is None:
            self.aruco_params = aruco.DetectorParameters_create() # TODO: tune
        self.aruco_params = self.aruco_params

        if flow_params == None:
            flow_params = dict(
                    winSize = (15,15),
                    maxLevel = 2,
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
                    )
        self.flow_params = flow_params

        self.marker_len = marker_len
        self.camera_params = camera_params

        # These are buffers to permit better visulizations in future versions
        self.gray_buf = []
        self.corners_buf = []
        self.ids_buf = []
        

    def track_source(self, src, output=None, useflow=True, wait=1):
        '''
        Tracks aruco markers for a video source

        Params:
        src - the value passed to cv2.VideoCapture(). 0 for webcam or str for a video file
        '''
        det_markers = 0
        det_frames = 0
        tot_frames = 0

        cap = cv2.VideoCapture(src)
        vout = None
        if output is not None:
            width = int(cap.get(3)) 
            height = int(cap.get(4))
            size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            fps = 15
            vout = cv2.VideoWriter()
            ret = vout.open(output, cv2.VideoWriter_fourcc(*'MJPG'), fps, size, True)
            if not ret:
                raise Exception("VideoWriter failed to open")

        while (cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                break
            out = frame.copy()
            rvec, tvec, obj_points, corners, ids, out = self.track_frame(frame, useflow=useflow, out=out)
            
            # draw
            if ids is not None:
                out = self._draw_marker_poses(out, rvec, tvec)
                out = self._draw_marker_frames(out, corners, ids)
                det_markers += ids.shape[0]
                det_frames += 1
    
            if vout is not None:
                vout.write(out)
            else:
                cv2.imshow('Tracker', cv2.resize(out, (out.shape[1]//3, out.shape[0]//3)))  

            tot_frames += 1
            
            if cv2.waitKey(wait) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.waitKey(1)
        if vout:
            vout.release()
            cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)

        return det_markers, det_frames, tot_frames


    def _draw_marker_poses(self, img, rvec, tvec):
        camera_matrix = self.camera_params[0]
        dist_coeffs = self.camera_params[1]
        axis_len = self.marker_len / 2
        for i in range(tvec.shape[0]):
            img = cv2.aruco.drawAxis(img, camera_matrix, dist_coeffs, rvec[i], tvec[i], axis_len)
        return img


    def _draw_marker_frames(self, img, corners, ids):
        return aruco.drawDetectedMarkers(img, corners, ids)


    def _update_buffers(self, gray, corners, ids, buf_size=5):
        '''
        Adds gray, corners, and ids to their respective buffers. Will remove old
        items in the buffers based on buf_size
        '''
        self.gray_buf.append(gray)
        self.corners_buf.append(corners)
        self.ids_buf.append(ids)
        if len(self.ids_buf) > buf_size:
            self.gray_buf.pop(0)
            self.corners_buf.pop(0)
            self.ids_buf.pop(0)


    def track_frame(self, frame, useflow=False, out=None):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids = self._detect_markers(gray)
        if useflow and self._did_ape_fail(ids):
            corners, ids, out = self._find_more_corners(gray, corners, ids, out=out)
        if useflow:
            self._update_buffers(gray, corners, ids)
        
        rvec, tvec, obj_points = None, None, None
        if ids is not None:
            rvec, tvec, obj_points = self._get_marker_poses(corners)
        return rvec, tvec, obj_points, corners, ids, out


    def _detect_markers(self, gray):
        '''
        Uses aruco to detect markers in the grayscale image gray.

        Returns:
        corners - A list of 4x1x2 np.ndarray objects representing to four corners of each marker
        ids - A Nx1 list of ids of each marker. The n-th id corresponds to the corners[n] corner positions 
        '''
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
        return corners, ids
    

    def _did_ape_fail(self, ids):
        '''
        Returns True if aproximate pose estimation fails.

        TODO: Implement
        '''
        return True
    

    def _find_more_corners(self, gray, corners, ids, out=None):
        '''
        Given a grayscale image and the corners and ids found via APE, use optical flow
        to find additional corners.

        Looks back at the buffers to see which markers were successfully detected last frame,
        but were not detected this frame. Then uses optical flow to find these markers in the
        next frame, where APE failed. If optical flow is successfull, the new corners and id are
        added to the corners and ids arrays respectively.
        
        Returns the original corners and ids arrays with the new values founded (via optical flow) added.

        Returns:
        corners - A list of 4x1x2 np.ndarray objects representing to four corners of each marker
        ids - A Nx1 list of ids of each marker. The n-th id corresponds to the corners[n] corner positions
        '''
        if ids is None:
            return None, None, out

        if len(self.gray_buf) == 0:
            # cannot compute optical flow without history in the buffers
            return corners, ids, out

        if self.ids_buf[-1] is None:
            # cannot compute optical flow if the last frame contains zero found markers
            return corners, ids, out
        
        prev_ids = self.ids_buf[-1]
        if ids is None:
            corners = []
            ids = np.empty((0,1))
        for pid in prev_ids: # loop for all of the markers found last frame
            if pid not in ids: # if it was not found this frame, compute flow
                p0 = self._get_p0(pid, -1)
                p1, st, err = cv2.calcOpticalFlowPyrLK(self.gray_buf[-1], gray, p0, None, **self.flow_params)
                valid_p1 = p1[st==1]
                if valid_p1.shape[0] == 4: # if flow was found all 4 of the marker corners. TODO: possible to do it with only 3
                    if out is not None:
                        out = self._draw_flow(out, p0, p1, st)
                    # add the new find to the corners and ids arrays
                    corners.append(valid_p1[np.newaxis, :, :])
                    ids = np.concatenate((ids, [pid]))

        return corners, ids, out
    

    def _get_p0(self, pid, buf_index):
        '''
        Indexes the buffers to retrieve the corner points associated with a
        given marker id.

        Params:
        pid - int, marker id
        buf_index - int, the index of the buffer to use as the last frame
        '''
        prev_ids = self.ids_buf[buf_index]
        prev_corners = self.corners_buf[buf_index]
        p0 = np.array(prev_corners)[prev_ids == pid].reshape(-1,1,2)
        return p0

    def _draw_flow(self, img, p0, p1, st):
        '''
        A helper method for visualizing optical flow.
        TODO: Improve to take advantage of the entire buffer for cooler visualizations

        Params:
        img - the image to draw on (can be color)
        p0 - the set of marker corners from the previous frame
        p1 - the set of marker corners from this frame
        st - the status return value from cv2.calcOpticalFlowPyrLK()
        '''

        valid_p1 = p1[st==1]
        valid_p0 = p0[st==1]
        for i in range(valid_p0.shape[0]):
            start = tuple(valid_p0[i,:].astype(np.int))
            end = tuple(valid_p1[i,:].astype(np.int))
            color = (255, 0, 255) # pink
            thickness = 10
            img = cv2.arrowedLine(img, start, end, color, thickness)
        return img

    def _get_marker_poses(self, corners):
        camera_matrix = self.camera_params[0]
        dist_coeffs = self.camera_params[1]
        rvecs, tvecs, obj_points = cv2.aruco.estimatePoseSingleMarkers(
                corners,
                self.marker_len,
                camera_matrix,
                dist_coeffs)
        
        return rvecs, tvecs, obj_points
    
    def _to_pen_tip(tvec, rvec, pen_len):
        '''
        converts the world coordinates of the center of a marker to the world coordinates of the pen
        tip. Assumes basic pen/marker arangement.

        Params:
        tvec - ???
        rvec - ???
        pen_len - length of the pen in meters

        Returns:
        X - Nx3 matrix of pen tip coordinate. Each row corresponds to the x, y, z world location of the pen tip
        '''
            

def demo():
    # camera_params = calibrate_charuco()
    # np.save('camera_params.npy', camera_params)
    camera_params = np.load('camera_params.npy', allow_pickle=True)
    marker_len = 0.0365125
    tracker = Tracker(marker_len=marker_len, camera_params=camera_params)
    # tracker.track_source(0)
    tracker.track_source('../test_footage/paul.MOV', wait=0)


if __name__ == '__main__':
    demo()