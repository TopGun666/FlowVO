import os
import sys
import numpy as np
from scipy.misc import imread, imsave
import tensorflow as tf
import uuid
import time
import cv2

from FlowNet2_src import FlowNet2, LONG_SCHEDULE

from visual_odometry import PinholeCamera, VisualOdometry

flags = tf.app.flags
flags.DEFINE_string("ckpt_file", "FlowNet2_src/checkpoints/FlowNet2/flownet-2.ckpt-0", "save the checkpoints file")
FLAGS = flags.FLAGS


cam = PinholeCamera(1241.0, 376.0, 718.8560, 718.8560, 607.1928, 185.2157)
vo = VisualOdometry(cam, '/home/ubuntu/users/tongpinmo/dataset/KITTI_odometry_dataset/dataset/poses/01.txt')
traj = np.zeros((1200, 1200, 3), dtype=np.uint8)
if __name__ == '__main__':

    # # Load model
    # ckpt_file = 'FlowNet2_src/checkpoints/FlowNet2/flownet-2.ckpt-0'
    # saver = tf.train.Saver()
    # sess = tf.Session()
    # saver.restore(sess, ckpt_file)

    image_ref_tensor = tf.placeholder(tf.float32, [1, 376, 1241, 3])
    image_cur_tensor = tf.placeholder(tf.float32, [1, 376, 1241, 3])
    flownet2 = FlowNet2()
    inputs = {'input_a': image_ref_tensor, 'input_b': image_cur_tensor}

    flow_dict = flownet2.model(inputs, LONG_SCHEDULE, trainable=False)
    pred_flow = flow_dict['flow']
    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, FLAGS.ckpt_file)

    # Read
    img_seq = []
    with open('/home/ubuntu/users/tongpinmo/dataset/KITTI_odometry_dataset/dataset/sequences/01/image_2/img_seq.txt', 'r') as f:
        for line in f:
            img_seq.append(list(line.strip('\n').split(',')))

    PATH = "/home/ubuntu/users/tongpinmo/dataset/KITTI_odometry_dataset/dataset/sequences/01/image_2/"
    for img_id in range(len(img_seq) - 2):
        # start = time.clock()
        print'NO.', img_id + 1, 'frame.'
        img_seq_str = ''.join(img_seq[img_id])
        img_dir = os.path.join(PATH + img_seq_str)
        img_id = img_id + 1

        img = imread(img_dir, 0)/1.
        cv2.imshow('Road facing camera', img)
        img = np.array([img]).astype(np.float32)
        vo.update(img, img_id, sess, pred_flow, image_ref_tensor, image_cur_tensor)

        # end = time.clock()
        # print("time:", str(end - start))
        cur_t = vo.cur_t
        if(img_id > 2):
            x, y, z = cur_t[0], cur_t[1], cur_t[2]
        else:
            x, y, z = 0., 0., 0.

        draw_x, draw_y = int(x)+290, int(z)+90
        true_x, true_y = int(vo.trueX)+290, int(vo.trueZ)+90

        cv2.circle(traj, (draw_x, draw_y), 1, (img_id*255/len(img_seq), 255-img_id*255/len(img_seq), 0), 1)
        cv2.circle(traj, (true_x, true_y), 1, (0,0,255), 2)
        cv2.rectangle(traj, (10, 20), (600, 60), (0, 0, 0), -1)
        text = "Coordinates: x=%2fm y=%2fm z=%2fm"%(x, y, z)
        cv2.putText(traj, text, (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)

        # cv2.imshow('Road facing camera', img)
        cv2.imshow('Trajectory', traj)
        cv2.waitKey(1)

    cv2.imwrite('map_2.png', traj)




