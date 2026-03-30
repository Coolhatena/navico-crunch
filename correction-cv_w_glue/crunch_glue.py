""" A template script for computer vision projects, ready to be used with a config.json file """
import cv2
import json
import os
import sys
import numpy as np
from time import sleep

# --- config ---
def get_base_path():
	if getattr(sys, "frozen", False):  # ejecutándose como .exe
		return os.path.dirname(sys.executable)
	return os.path.dirname(os.path.abspath(__file__))

BASE_DIR = get_base_path()
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")

config = {}
if os.path.exists(CONFIG_PATH):
	with open(CONFIG_PATH, "r", encoding="utf-8") as f:
		config = json.load(f)

camera_index = int(config.get("camera_index", 0))
camera_api = cv2.CAP_DSHOW
# --- fin config ---
cam = cv2.VideoCapture(camera_index)

# Double check for slow cameras
while not cam.isOpened():
	cam = cv2.VideoCapture(camera_index)
	print("Waiting for camera...")
	sleep(0.05) # Micro-tic between tries


# Filter
LOW, UPP = config["filter_pink"]
LOW = np.array(LOW)
UPP = np.array(UPP)

# ROI
ROI = config["roi"]
startX, startY = ROI[0]
endX, endY = ROI[1]
print(startX, endX)
print(startY, endY)

q_unicode = ord('q')

while True:
	_, frame = cam.read()

	frame_cropped = frame.copy()
	frame_cropped = frame_cropped[startY:endY, startX:endX]

	hsv = cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2HSV)
	msk = cv2.inRange(hsv, LOW, UPP)
	filtered = cv2.bitwise_and(frame_cropped, frame_cropped, mask= msk)

	gray_cropped = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
	_, binary_cropped = cv2.threshold(gray_cropped, 1, 255, cv2.THRESH_BINARY)
	contours, _ = cv2.findContours(binary_cropped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	largest_contour_corners = None
	if contours:
		largest_contour = max(contours, key=cv2.contourArea)
		x, y, w, h = cv2.boundingRect(largest_contour)
		top_left = (x, y)
		bottom_right = (x + w, y + h)
		largest_contour_corners = (top_left, bottom_right)

		mask = np.zeros(filtered.shape[:2], dtype=np.uint8)
		cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
		cropped_obj = cv2.bitwise_and(filtered, filtered, mask=mask)
		# cropped_obj = frame_cropped[y:y+h, x:x+w]

		gray_obj = cv2.cvtColor(cropped_obj, cv2.COLOR_BGR2GRAY)
		blurred_obj = cv2.GaussianBlur(gray_obj, (5, 5), 0)
		edges = cv2.Canny(blurred_obj, 10, 20)

		cv2.imshow('edges', edges)

		edge_contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

		for contour in edge_contours:
			contour_x, contour_y, contour_w, contour_h = cv2.boundingRect(contour)
			if 0 < contour_w < 30:
				contour_in_frame = contour + np.array([[[startX, startY]]])
				cv2.drawContours(frame, [contour_in_frame], -1, (0, 255, 0), 2)

		cv2.imshow('Filtered', cropped_obj)

	cv2.imshow('Frame', frame)

	key = cv2.waitKey(1)
	if key == q_unicode: # If 'q' is pressed, close program (Its case sensitive)
		break

cam.release()
cv2.destroyAllWindows()
