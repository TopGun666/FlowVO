#!/usr/bin/python2.7
import cv2
import numpy as np
import os
import argparse

TAG_FLOAT = 202021.25


def read(file):

    assert type(file) is str, "file is not str %r" % str(file)
    assert os.path.isfile(file) is True, "file does not exist %r" % str(file)
    assert file[-4:] == '.flo', "file ending is not .flo %r" % file[-4:]
    f = open(file,'rb')
    flo_number = np.fromfile(f, np.float32, count=1)[0]
    assert flo_number == TAG_FLOAT, 'Flow number %r incorrect. Invalid .flo file' % flo_number
    w = np.fromfile(f, np.int32,count=1)
    h = np.fromfile(f, np.int32,count=1)
    #if error try: data = np.fromfile(f, np.float32, count=2*w[0]*h[0])
    data = np.fromfile(f, np.float32, count=2*w*h)
    # Reshape data into 3D array (columns, rows, bands)
    flow = np.resize(data, (int(h), int(w), 2))
    f.close()

    return flow

# Load optical flow
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--flowfile', type=str, default='colorTest.flo', help='Flow file')
    file = parser.parse_args().flowfile
    flow = read(file)

    # load image
    prev = cv2.imread('./rgb-e55d63ab-440b-4f7d-a42b-91238424586c.png')

# calculate Mat
    w = int(prev.shape[1])
    h = int(prev.shape[0])
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    coords = np.float32(np.dstack([x_coords, y_coords]))
    pixel_map = coords + flow
    new_frame = cv2.remap(prev, pixel_map, None, cv2.INTER_LINEAR)
 
    cv2.imwrite('new_frame.png', new_frame)
