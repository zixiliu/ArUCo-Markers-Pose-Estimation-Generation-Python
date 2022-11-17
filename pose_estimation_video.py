'''
Sample Command:-
python detect_aruco_video.py --type DICT_5X5_100 --camera True
python detect_aruco_video.py --type DICT_5X5_100 --camera False --video test_video.mp4
'''

import numpy as np
from utils import ARUCO_DICT, aruco_display
import argparse
import time
import cv2
import sys
from pose_estimation import pose_esitmation
import pdb

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--camera", default="False", help="Set to True if using webcam")
ap.add_argument("-v", "--video", default = '/mnt/h/My Drive/BioRobotics/Hand/Intrinsic Sensing/Data/2022-11-15/cam_in_hand_4.mkv', help="Path to the video file")
ap.add_argument("-k", "--K_Matrix", default="calibration_matrix.npy", required=False, help="Path to calibration matrix (numpy file)")
ap.add_argument("-d", "--D_Coeff", default="distortion_coefficients.npy", required=False, help="Path to distortion coefficients (numpy file)")
ap.add_argument("-t", "--type", type=str, default="DICT_5X5_100", help="Type of ArUCo tag to detect")
args = vars(ap.parse_args())



f = open('/mnt/h/My Drive/BioRobotics/Hand/Intrinsic Sensing/Data/2022-11-15/aruco_pose_in_hand_4.csv','w')
f.write('time[s], rvec0, rvec1, rvec2, tvec0, tvec1, tvec2\n')

if args["camera"].lower() == "true":
	video = cv2.VideoCapture(0)
	time.sleep(2.0)
	
else:
	if args["video"] is None:
		print("[Error] Video file location is not provided")
		sys.exit(1)

	video = cv2.VideoCapture(args["video"])
	fps = video.get(cv2.CAP_PROP_FPS)
	print("fps = ", fps)

if ARUCO_DICT.get(args["type"], None) is None:
	print(f"ArUCo tag type '{args['type']}' is not supported")
	sys.exit(0)

arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[args["type"]])
arucoParams = cv2.aruco.DetectorParameters_create()

aruco_dict_type = ARUCO_DICT[args["type"]]
calibration_matrix_path = args["K_Matrix"]
distortion_coefficients_path = args["D_Coeff"]

k = np.load(calibration_matrix_path)
d = np.load(distortion_coefficients_path)

while True:
	ret, frame = video.read()
	
	if ret is False:
		break

	time_stamp = video.get(cv2.CAP_PROP_POS_MSEC)
	output, rvec, tvec = pose_esitmation(frame, aruco_dict_type, k, d)
	cv2.imshow("Image", output)

		
	revc_string = ','.join(['%.8f' % num for num in rvec[0,0,:]])
	tevc_string = ','.join(['%.8f' % num for num in tvec[0,0,:]])

	line = str(time_stamp/1000)+','+revc_string+','+tevc_string+'\n'
	f.write(line)

	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
	    break

cv2.destroyAllWindows()
video.release()

f.close()

