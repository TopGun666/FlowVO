import cv2
import numpy as np
import argparse
import os

from FlowNet2_src import read_flow
from numpy import *

# camera matrix from calibration
K = np.array([[517.67386649, 0.0, 268.65952163], [0.0, 519.75461699, 215.58959128], [0.0, 0.0, 1.0]])

def find_point_matches ( point_1, point_2 ):
	"""
	:param point_1: Pixel coordinates of the previous frame
	:param point_2: Pixel coordinates of the next frame
	:return:
	"""
	parser = argparse.ArgumentParser()
	parser.add_argument('--flowfile', type=str, default='colorTest.flo', help='Flow file')
	file = parser.parse_args().flowfile
	flow = read_flow(file)
	point_1 = np.float32(point_1)
	point_2 = np.float32(point_2)
	point_2 = point_1 + flow
	return point_1, point_2

def pose_estimate_2d2d( point_1, point_2 ):
	"""
	:param point_1: Pixel coordinates of the previous frame
	:param point_2: Pixel coordinates of the next frame
	:param matches: Matching relationship between pixels before and after two frames
	:param R:
	:param t:
	:return: return camera pose and ...
	"""

	# calculate fundamental_matrix
	Fundamental_matrix, mask_F = cv2.findFundamentalMat(point_1, point_2, cv2.FM_RANSAC, 3, 0.99)
	print ("Fundamental_matrix:", Fundamental_matrix)

	# calculate Homography_matrix
	Homography_matrix, mask_H = cv2.findHomography(point_1, point_2, cv2.RANSAC, 5.0)
	print ("Homography_matrix:", Homography_matrix)

	# calculate essential_matrix
	Essential_matrix = np.transpose(K) * Fundamental_matrix * K
	print ("Essential_matrix:", Essential_matrix)

	return Fundamental_matrix, Homography_matrix, Essential_matrix

def decomposeEssentialMat(InputArray_E):
	"""
	:param InputArray_E: Essential_matrix
	:return:
	"""
	U, Sigma, Vt = linalg.svd(K)
	OutputArray_R1 = U * Sigma * Vt
	OutputArray_R2 = U * np.transpose(Sigma) * Vt
	t = U[:, 2] * 1.0
	return OutputArray_R1, OutputArray_R2, t
	
def recoverPose(InputArray_E, point_1, point_2, focal, dd):
	