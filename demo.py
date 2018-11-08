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

if __name__ == '__main__':
    # Graph construction
    # im1_pl = tf.placeholder(tf.float32, [1, 384, 512, 3])
    # im2_pl = tf.placeholder(tf.float32, [1, 384, 512, 3])

    im1_pl = tf.placeholder(tf.float32, [1, 720, 1280, 3])
    im2_pl = tf.placeholder(tf.float32, [1, 720, 1280, 3])

    flownet2 = FlowNet2()
    inputs = {'input_a': im1_pl, 'input_b': im2_pl}
    flow_dict = flownet2.model(inputs, LONG_SCHEDULE, trainable=False)
    pred_flow = flow_dict['flow']

    # Feed forward
    # im1 = imread('FlowNet2_src/example/0000000000.png')/255.
    # im2 = imread('FlowNet2_src/example/0000000001.png')/255.

    ckpt_file = 'FlowNet2_src/checkpoints/FlowNet2/flownet-2.ckpt-0'
    saver = tf.train.Saver()

    sess = tf.Session()
    saver.restore(sess, ckpt_file)

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

        # ckpt_file = 'FlowNet2_src/checkpoints/FlowNet2/flownet-2.ckpt-0'
        # saver = tf.train.Saver()

        #with tf.Session() as sess:
        #    saver.restore(sess, ckpt_file)
            # Double check loading is correct
            #for var in tf.all_variables():
            #  print(var.name, var.eval(session=sess).mean())
        feed_dict = {im1_pl: im1, im2_pl: im2}
        pred_flow_val = sess.run(pred_flow, feed_dict=feed_dict)
        end = time.clock()
        print str(end - start)
        # Save .flo
        out_path = './result/fandikai/'
        unique_name = 'flow-' + str(uuid.uuid4())
        full_out_path = os.path.join(out_path, unique_name + '.flo')
        write_flow(pred_flow_val, full_out_path)

        # Save flow_image
        flow_img = flow_to_image(pred_flow_val[0])
        full_out_path = os.path.join(out_path, unique_name + '.png')
        imsave(full_out_path, flow_img)

    # Visualization
    # import matplotlib.pyplot as plt
    # flow_im = flow_to_image(pred_flow_val[0])
    # plt.imshow(flow_img)
    # plt.savefig("./result/flow_im.png")
    # plt.imsave("flow_ing", flow_im, cmap='hot')
    # plt.show()



