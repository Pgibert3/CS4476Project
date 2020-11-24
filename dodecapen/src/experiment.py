import numpy as np
import cv2
import os
import glob
import matplotlib.pyplot as plt
from tracker import Tracker
import pdb


def main():
    camera_params = np.load('camera_params.npy', allow_pickle=True)
    marker_len = 0.0365125

    experiments_dir = '../test_footage/experiments'
    results_dir = '../results'

    experiments = glob.glob(os.path.join(experiments_dir, '*/'))
    for exp in experiments:
        exp_name = exp.split('/')[-2]
        output = os.path.join(results_dir, exp_name + '_results')
        os.mkdir(output)
        run_experiment(exp, output=output, camera_params=camera_params, marker_len=marker_len)
    
    results = glob.glob(os.path.join(results_dir, '*/'))
    for r in results:
        rname =  r.split('/')[-2]
        fig = plot_results(r)
        fname = rname + '.jpeg'
        fig.savefig(os.path.join(r, fname))


def run_experiment(footage_dir, output=None, camera_params=None, marker_len=None):
    if camera_params is None or marker_len is None:
        raise Exception('missing params')
    
    fnames = glob.glob(os.path.join(footage_dir, '*.MOV'))
    tracker = Tracker(marker_len=marker_len, camera_params=camera_params)

    for f in fnames:
        test_name = os.path.splitext(os.path.basename(f))[0]
        run_trial(f, test_name, output, useflow=False, tracker=tracker)
        run_trial(f, test_name, output, useflow=True, tracker=tracker)


def run_trial(fname, test_name, output, useflow=True, tracker=None):
    if tracker is None:
        raise Exception("no tracker")
    tag = 'flow' if useflow else 'noflow'
    vout_name = os.path.join(output , test_name + '_' + tag + '_.avi')
    det_markers, det_frames, tot_frames = tracker.track_source(fname, output=vout_name, useflow=useflow)
    save_results(output, test_name + '_' + tag, useflow, det_markers, det_frames, tot_frames)


def save_results(output_dir, test_name, useflow, det_markers, det_frames, tot_frames):
    data = {
            'test_name' : test_name,
            'useflow' : useflow,
            'det_markers' : det_markers,
            'det_frames' : det_frames,
            'tot_frames' : tot_frames
        }
    fname = os.path.join(output_dir, test_name + '_results')
    np.save(fname, data, allow_pickle=True)
        


def plot_results(results_dir):
    fnames = glob.glob(os.path.join(results_dir, '*.npy'))
    det_markers_flow = []
    det_markers_noflow = []
    det_frames_flow = []
    det_frames_noflow = []
    tot_frames_flow = []
    tot_frames_noflow = []

    for f in fnames:
        data = np.load(f, allow_pickle=True).item()
        if data['useflow']:
            det_markers_flow.append(data['det_markers'])
            det_frames_flow.append(data['det_frames'])
            tot_frames_flow.append(data['tot_frames'])
        else:
            det_markers_noflow.append(data['det_markers'])
            det_frames_noflow.append(data['det_frames'])
            tot_frames_noflow.append(data['tot_frames'])
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
    fig.tight_layout(pad=3)
    ax[0].set_title('Success At Tracking At Least One Marker Per Frame')
    ax[0].set_xlabel('Frames Per Trial')
    ax[0].set_ylabel('Number Of Frames Detected')
    ax[0].bar(tot_frames_flow, det_frames_flow, color='green', label='With Optic Flow')
    ax[0].bar(tot_frames_noflow, det_frames_noflow, color='red', label='Without Optic Flow')
    handles, labels = ax[0].get_legend_handles_labels()
    ax[0].legend(handles, labels)

    ax[1].set_title('Success At Tracking Individual Markers')
    ax[1].set_xlabel('Frames Per Trial')
    ax[1].set_ylabel('Number Of Markers Detected')
    ax[1].bar(tot_frames_flow, det_markers_flow, color='green', label='With Optic Flow')
    ax[1].bar(tot_frames_noflow, det_markers_noflow, color='red', label='Without Optic Flow')
    handles, labels = ax[1].get_legend_handles_labels()
    ax[1].legend(handles, labels)

    return fig


if __name__ == '__main__':
    main()