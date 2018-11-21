import os
import sys
import numpy as np
from scipy.misc import imread, imsave
import tensorflow as tf
import uuid
import time

from FlowNet2_src import FlowNet2, LONG_SCHEDULE
from FlowNet2_src import flow_to_image
from FlowNet2_src import write_flow

flags = tf.app.flags
flags.DEFINE_integer("checkpoints_file",'FlowNet2_src/checkpoints/FlowNet2/flownet-2.ckpt-0',"save the checkpoints file")
FLAGS = flags.FLAGS

if __name__ == '__main__':

    # # Load model
    # ckpt_file = 'FlowNet2_src/checkpoints/FlowNet2/flownet-2.ckpt-0'
    # saver = tf.train.Saver()
    # sess = tf.Session()
    # saver.restore(sess, ckpt_file)

    # Read
    img_seq = []
    with open('./fandikai/img_seq.txt', 'r') as f:
        for line in f:
            img_seq.append(list(line.strip('\n').split(',')))

    PATH = "./fandikai/"
    for i in range(len(img_seq) - 2):
        start = time.clock()
        print 'NO.',i+1,'frame.'
        img_seq_str1 = ''.join(img_seq[i])
        img_dir1 = os.path.join(PATH + img_seq_str1)
        img_seq_str2 = ''.join(img_seq[i+1])
        img_dir2 = os.path.join(PATH + img_seq_str2)
        i = i + 1

        im1 = imread(img_dir1)/255.
        im2 = imread(img_dir2)/255.

        im1 = np.array([im1]).astype(np.float32)
        im2 = np.array([im2]).astype(np.float32)


        end = time.clock()
        print "time:", str(end - start)





