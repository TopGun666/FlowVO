import numpy as np 
import cv2
import tensorflow as tf

from FlowNet2_src import FlowNet2, LONG_SCHEDULE

flags = tf.app.flags
flags.DEFINE_string("ckpt_file", "FlowNet2_src/checkpoints/FlowNet2/flownet-2.ckpt-0", "save the checkpoints file")
FLAGS = flags.FLAGS

STAGE_FIRST_FRAME = 0
STAGE_SECOND_FRAME = 1
STAGE_DEFAULT_FRAME = 2
kMinNumFeature = 1500
np.set_printoptions(threshold='nan')
lk_params = dict(winSize  = (21, 21), 
				#maxLevel = 3,
             	criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


def featureTracking(image_ref, image_cur):
	# from demo_KITTI import cam
	# image_ref_tensor = tf.placeholder(tf.float32, [1, int(cam.height), int(cam.width), 3])
	# image_cur_tensor = tf.placeholder(tf.float32, [1, int(cam.height), int(cam.width), 3])
	w = int(image_ref.shape[2])
	h = int(image_ref.shape[1])
	print "h:", h, "w:", w
	image_ref_tensor = tf.placeholder(tf.float32, [1, h, w, 3])
	image_cur_tensor = tf.placeholder(tf.float32, [1, h, w, 3])
	print "image_ref", image_ref.shape, "image_cur", image_cur.shape
	flownet2 = FlowNet2()
	inputs = {'input_a': image_ref_tensor, 'input_b': image_cur_tensor}
	flow_dict = flownet2.model(inputs, LONG_SCHEDULE, trainable=False)
	pred_flow = flow_dict['flow']
	feed_dict = {image_ref_tensor: image_ref, image_cur_tensor: image_cur}
	saver = tf.train.Saver()
	sess = tf.Session()
	saver.restore(sess, FLAGS.ckpt_file)
	print "pred_flow:", pred_flow.shape
	pred_flow_val = sess.run(pred_flow, feed_dict=feed_dict)
	y_coords, x_coords = np.mgrid[0:h, 0:w]
	kp1 = np.float32(np.dstack([x_coords, y_coords]))
	kp2 = kp1 + pred_flow_val
	kp1 = np.reshape(kp1, (466616, 2))
	kp2 = np.reshape(kp2, (466616, 2))
	print "pass#################################################################"
	# print "kp1:", kp1, "kp2:", kp2
	return kp1, kp2

class PinholeCamera:
	def __init__(self, width, height, fx, fy, cx, cy,
				k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0):
		self.width = width
		self.height = height
		self.fx = fx
		self.fy = fy
		self.cx = cx
		self.cy = cy
		self.distortion = (abs(k1) > 0.0000001)
		self.d = [k1, k2, p1, p2, k3]



class VisualOdometry:
	def __init__(self, cam, annotations):
		self.frame_stage = 0
		self.cam = cam
		self.new_frame = None
		self.last_frame = None
		self.cur_R = None
		self.cur_t = None
		self.px_ref = None
		self.px_cur = None
		self.focal = cam.fx
		self.pp = (cam.cx, cam.cy)
		self.trueX, self.trueY, self.trueZ = 0, 0, 0
		# self.detector = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
		with open(annotations) as f:
			self.annotations = f.readlines()


	
	def getAbsoluteScale(self, frame_id):  # specialized for KITTI odometry dataset
		ss = self.annotations[frame_id-1].strip().split() # ss is pose groundtruth.
		x_prev = float(ss[3])
		y_prev = float(ss[7])
		z_prev = float(ss[11])
		ss = self.annotations[frame_id].strip().split()
		x = float(ss[3])
		y = float(ss[7])
		z = float(ss[11])
		self.trueX, self.trueY, self.trueZ = x, y, z
		return np.sqrt((x - x_prev)*(x - x_prev) + (y - y_prev)*(y - y_prev) + (z - z_prev)*(z - z_prev))

	# def processFirstFrame(self):
	# 	self.px_ref = self.detector.detect(self.new_frame)
	# 	self.px_ref = np.array([x.pt for x in self.px_ref], dtype=np.float32)
	# 	self.frame_stage = STAGE_SECOND_FRAME
	# 	# print "x.pt:", x.pt, "px_ref:", self.px_ref.shape
	# def processSecondFrame(self):
	# 	self.px_ref, self.px_cur = featureTracking(self.last_frame, self.new_frame, self.px_ref)
	# 	E, mask = cv2.findEssentialMat(self.px_cur, self.px_ref, focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=1)
	# 	_, self.cur_R, self.cur_t, mask = cv2.recoverPose(E, self.px_cur, self.px_ref, focal=self.focal, pp = self.pp)
	# 	self.frame_stage = STAGE_DEFAULT_FRAME
	# 	self.px_ref = self.px_cur
	# 	# print "px_ref2:", self.px_ref
	def processFrame(self, frame_id):
		# self.px_ref, self.px_cur = featureTracking(self.last_frame, self.new_frame, self.px_ref)
		self.px_ref, self.px_cur = featureTracking(self.last_frame, self.new_frame)
		E, mask = cv2.findEssentialMat(self.px_cur, self.px_ref, focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=1)
		_, R, t, mask = cv2.recoverPose(E, self.px_cur, self.px_ref, focal=self.focal, pp = self.pp)
		# print ("R:", R)
		absolute_scale = self.getAbsoluteScale(frame_id)
		if(absolute_scale > 0.1):
			self.cur_t = self.cur_t + absolute_scale * self.cur_R.dot(t)
			print "pass*****************************************************************"
			self.cur_R = R.dot(self.cur_R)
		# if(self.px_ref.shape[0] < kMinNumFeature):
		# 	self.px_cur = self.detector.detect(self.new_frame)
		# 	self.px_cur = np.array([x.pt for x in self.px_cur], dtype=np.float32)
		self.px_ref = self.px_cur
		# print "px_ref3:", self.px_ref
	def update(self, img, frame_id):
		assert (img.shape[1] == self.cam.height and img.shape[2] == self.cam.width), "Frame: provided image has not the same size as the camera model or image is not grayscale"
		self.new_frame = img
		if(frame_id < 3):
			self.last_frame = self.new_frame
			print "frame_id666:", frame_id
		if(frame_id >= 3):
			self.processFrame(frame_id)
			self.last_frame = self.new_frame