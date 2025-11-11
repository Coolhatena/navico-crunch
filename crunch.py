""" A template script for computer vision projects """
import cv2
from time import sleep
import numpy as np
import helpers

camera_index = 2
cam = cv2.VideoCapture(camera_index)

is_frame_ok = False
while not cam.isOpened() and not is_frame_ok:
	cam = cv2.VideoCapture(camera_index)
	is_frame_ok, _ = cam.read()
	print("Waiting for camera...")
	sleep(0.05)

filter_pink = (np.array([0, 0, 0]), np.array([25, 255, 255]))
filter_gold = (np.array([18, 0, 173]), np.array([49, 168, 248]))
filter_white = (np.array([0, 0, 162]), np.array([180, 28, 255]))

filter_selected = filter_pink # Default

area = ((250,190), (390,290))
(x1, y1), (x2, y2) = area
center_x = x1 + ((x2 - x1) // 2)
center_line_pts = ((center_x, y1), (center_x, y2))

q_unicode = ord('q')
b_unicode = ord('b')

one_unicode = ord('1')
two_unicode = ord('2')
three_unicode = ord('3')

print(f"one: {one_unicode}")
print(f"two: {two_unicode}")
print(f"three: {three_unicode}")

saved_reference_center = None
while True:
	is_frame_ok, frame = cam.read()
	frame = cv2.rotate(frame, cv2.ROTATE_180)
	
	if (not is_frame_ok):
		continue

	subframe = frame[y1:y2, x1:x2]
	LOW, UPP = filter_selected
	hsv = cv2.cvtColor(subframe, cv2.COLOR_BGR2HSV)
	msk = cv2.inRange(hsv, LOW, UPP)

	contours, _ = cv2.findContours(msk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	y_top = y_bottom = None
	if contours:
		c = max(contours, key=cv2.contourArea)

		# Máscara rellena solo del contorno grande
		filled = np.zeros(msk.shape, np.uint8)
		cv2.drawContours(filled, [c], -1, 255, thickness=cv2.FILLED)

		# X del punto más a la izquierda del contorno
		x_left = int(c[:, 0, 0].min()) + 5
		# x_left, _, _, _ = cv2.boundingRect(c)  # equivalente

		target_x = np.clip(x_left, 0, filled.shape[1] - 1)

		# Y's blancos en esa X
		ys = np.where(filled[:, target_x] > 0)[0]
		if ys.size > 0:
			y_top    = int(ys.min())
			y_bottom = int(ys.max())

	
	cv2.rectangle(frame, area[0], area[1], (0, 255, 0), 2)
	# cv2.line(frame, center_line_pts[0], center_line_pts[1], (0, 0, 255), 2)

	if y_top and y_bottom:
		# print("top")
		# print((x1 + target_x, y1 + y_top))
		# print("Bottom")
		# print((x1 + target_x, y1 + y_bottom))
		# cv2.circle(frame, (x1 + target_x, y1 + y_top), 2, (255, 0, 0), -1)
		# cv2.circle(frame, (x1 + target_x, y1 + y_bottom), 2, (255, 0, 0), -1)
		pt_top = (x1 + target_x, y1 + y_top)
		pt_bottom = (x1 + target_x, y1 + y_bottom)
		pt_center = helpers.calculate_middle_point(pt_top, pt_bottom)
		cv2.circle(frame, pt_center, 2, (0, 255, 255), -1)

		if saved_reference_center:
			cv2.line(frame, pt_center, saved_reference_center, (255, 255, 255), 2)
			cv2.circle(frame, saved_reference_center, 5, (255, 0, 0), -1)

			x_diff, y_diff = (abs(saved_reference_center[0] - pt_center[0]), abs(saved_reference_center[1] - pt_center[1]))

			text_coords = (x1 - 60, y2 - 120) 
			cv2.putText(frame, f"Correccion: {x_diff}, {y_diff}", text_coords, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)


	cv2.imshow('Frame', frame)
	cv2.imshow('Filtered', msk)
	h, w = frame.shape[:2]
	# 240, 320
	# print(h, w)


	key = cv2.waitKey(1)
	if key == q_unicode: # If 'q' is pressed, close program (Its case sensitive)
		break

	if key == b_unicode: # If 'b' is pressed, save center 
		saved_reference_center = pt_center

	# Change filters
	if key == one_unicode:
		filter_selected = filter_pink
		saved_reference_center = None


	if key == two_unicode:
		filter_selected = filter_gold
		saved_reference_center = None

	if key == three_unicode:
		filter_selected = filter_white
		saved_reference_center = None

	print(filter_selected)

cam.release()
cv2.destroyAllWindows()
