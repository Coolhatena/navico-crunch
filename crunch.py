""" A template script for computer vision projects """
import cv2
from time import sleep
import numpy as np
import helpers
import socket
import threading
import json
import os

def load_filters_config():
    """Load HSV filters from config.json -> {filters:{name:{low:[H,S,V], high:[H,S,V]}}}.
    Returns numpy array pairs for pink, gold, white. Falls back to defaults.
    """
    defaults = {
        "pink":  {"low": [0, 0, 0],   "high": [25, 255, 255]},
        "gold":  {"low": [18, 0, 173], "high": [49, 168, 248]},
        "white": {"low": [0, 0, 162], "high": [180, 28, 255]},
    }
    cfg_path = os.path.join(os.path.dirname(__file__), "config.json")

    def to_pair(cfg, name):
        base = defaults[name]
        low = (cfg.get(name, {}) or {}).get("low", base["low"]) if isinstance(cfg, dict) else base["low"]
        high = (cfg.get(name, {}) or {}).get("high", base["high"]) if isinstance(cfg, dict) else base["high"]
        return (np.array(low), np.array(high))

    filters_cfg = None
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "filters" in data:
            filters_cfg = data["filters"]
    except Exception as e:
        print(f"[CONFIG:FILTERS] Using defaults. Reason: {e}")

    filter_pink = to_pair(filters_cfg, "pink")
    filter_gold = to_pair(filters_cfg, "gold")
    filter_white = to_pair(filters_cfg, "white")
    return filter_pink, filter_gold, filter_white

# Load filters from config.json (with defaults)
filter_pink, filter_gold, filter_white = load_filters_config()

filter_selected = filter_pink  # Default

q_unicode = ord('q')
b_unicode = ord('b')

one_unicode = ord('1')
two_unicode = ord('2')
three_unicode = ord('3')

print(f"one: {one_unicode}")
print(f"two: {two_unicode}")
print(f"three: {three_unicode}")

saved_reference_center = None

def load_config():
    """Load IP/PORT from config.json placed next to this script.
    Falls back to defaults if file is missing or invalid.
    """
    defaults = {"ip": "192.168.10.25", "port": 2001}
    cfg_path = os.path.join(os.path.dirname(__file__), "config.json")
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Support either top-level or nested under "server"
        if isinstance(data, dict) and "server" in data and isinstance(data["server"], dict):
            data = data["server"]
        ip = str(data.get("ip", defaults["ip"]))
        port = int(data.get("port", defaults["port"]))
        return ip, port
    except Exception as e:
        print(f"[CONFIG] Using defaults. Reason: {e}")
        return defaults["ip"], defaults["port"]

# Socket server config
# Same IP, different port. Loaded from config.json
IP, PORT = load_config()

# Shared state for detection results
values_lock = threading.Lock()
latest_values = {"x_diff": 0, "y_diff": 0}


def diffs_server_thread():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server.bind((IP, PORT))
    except OSError as e:
        print(f"[SERVER BIND ERROR] {e}")
        return
    server.listen()
    print(f"[LISTENING] Diff server listening on {IP}:{PORT}")

    while True:
        try:
            conn, addr = server.accept()
        except OSError as e:
            print(f"[ACCEPT ERROR] {e}")
            sleep(0.1)
            continue

        with values_lock:
            x = latest_values["x_diff"]
            y = latest_values["y_diff"]

        payload = f"{x},{y}\n".encode("utf-8")
        try:
            conn.sendall(payload)
            print(f"[SENT] to {addr}: {x},{y}")
        except Exception as e:
            print(f"[SEND ERROR] {e}")
        finally:
            try:
                conn.close()
            except Exception:
                pass


# Start server in background
threading.Thread(target=diffs_server_thread, daemon=True).start()

def load_camera_area_config():
    """Load camera index and detection area from config.json.
    Supports top-level keys: camera_index, area (as [[x1,y1],[x2,y2]] or {x1,y1,x2,y2})
    Also supports nested: {"camera": {"index": int}, "detection": {"area": ...}}
    """
    defaults_index = 0
    defaults_area = ((250, 190), (390, 290))
    cfg_path = os.path.join(os.path.dirname(__file__), "config.json")
    cam_idx = defaults_index
    area = defaults_area
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # camera index
        if isinstance(data, dict):
            if "camera_index" in data:
                cam_idx = int(data.get("camera_index", defaults_index))
            elif "camera" in data and isinstance(data["camera"], dict):
                cam_idx = int(data["camera"].get("index", defaults_index))

            # area
            raw_area = None
            if "area" in data:
                raw_area = data["area"]
            elif "detection" in data and isinstance(data["detection"], dict):
                raw_area = data["detection"].get("area")

            if isinstance(raw_area, list) and len(raw_area) == 2:
                (ax1, ay1), (ax2, ay2) = raw_area
                area = ((int(ax1), int(ay1)), (int(ax2), int(ay2)))
            elif isinstance(raw_area, dict):
                ax1 = int(raw_area.get("x1", defaults_area[0][0]))
                ay1 = int(raw_area.get("y1", defaults_area[0][1]))
                ax2 = int(raw_area.get("x2", defaults_area[1][0]))
                ay2 = int(raw_area.get("y2", defaults_area[1][1]))
                area = ((ax1, ay1), (ax2, ay2))
    except Exception as e:
        print(f"[CONFIG:CAM/AREA] Using defaults. Reason: {e}")

    # normalize ordering
    (ax1, ay1), (ax2, ay2) = area
    x1n, x2n = (ax1, ax2) if ax1 <= ax2 else (ax2, ax1)
    yn1, yn2 = (ay1, ay2) if ay1 <= ay2 else (ay2, ay1)
    return cam_idx, ((x1n, yn1), (x2n, yn2))

# Load camera index and area
camera_index, area = load_camera_area_config()
(x1, y1), (x2, y2) = area
center_x = x1 + ((x2 - x1) // 2)
center_line_pts = ((center_x, y1), (center_x, y2))

# Initialize camera using configured index
cam = cv2.VideoCapture(camera_index)
is_frame_ok = False
while not cam.isOpened() and not is_frame_ok:
    cam = cv2.VideoCapture(camera_index)
    is_frame_ok, _ = cam.read()
    print("Waiting for camera...")
    sleep(0.05)
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
			# Update shared latest diffs for the socket server
			with values_lock:
				latest_values["x_diff"] = int(x_diff)
				latest_values["y_diff"] = int(y_diff)

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
