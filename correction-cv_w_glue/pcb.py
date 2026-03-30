""" A template script for computer vision projects """
import cv2
from time import sleep
import numpy as np
import helpers
import sys
import socket
import threading
import json
import os
import re
import tkinter as tk
from datetime import datetime

TCP_DEBUG_RECV = True
TCP_RECV_POLL_TIMEOUT_S = 0.5
PIXEL_TO_DELTA_SCALE = 0.1
TCP_X_FACTOR = -1
TCP_Y_FACTOR = -1
VALID_TCP_CMDS = ("pink", "gold", "white", "p", "g", "w", "c", "d", "r", "z", "x", "o", "i")

def get_base_path():
	if getattr(sys, "frozen", False):  # ejecutándose como .exe
		return os.path.dirname(sys.executable)
	return os.path.dirname(os.path.abspath(__file__))

def debug_recv_data(data):
	"""Print useful diagnostics to infer payload encoding."""
	if not TCP_DEBUG_RECV:
		return

	print(f"[TCP RX] len={len(data)}")
	print(f"[TCP RX] bytes={data!r}")

	for enc in ("ascii", "utf-8"):
		try:
			decoded = data.decode(enc)
			print(f"[TCP RX] {enc}: OK -> {decoded!r}")
		except UnicodeDecodeError as e:
			print(f"[TCP RX] {enc}: FAIL ({e})")


def extract_cmd(data):
	"""Return first valid command token found in payload."""
	text = data.decode("utf-8", errors="ignore").strip().lower()
	if not text:
		return ""

	aliases = {
		"pink": "pink",
		"gold": "gold",
		"white": "white",
		"p": "pink",
		"g": "gold",
		"w": "white",
		"c": "center",
		"d": "dispenser_center",
		"r": "relative_reference",
		"z": "delta",
		"x": "relative_delta",
		"o": "reset_operator",
		"i": "reset_item",
	}

	tokens = re.split(r"[^a-z0-9_]+", text)
	for token in tokens:
		if token in aliases and token in VALID_TCP_CMDS:
			return aliases[token]

	if "pink" in text:
		return "pink"
	if "gold" in text:
		return "gold"
	if "white" in text:
		return "white"
	if "center" in text:
		return "center"
	if "dispenser" in text:
		return "dispenser_center"
	if "relative" in text:
		return "relative_reference"
	return ""


def load_filters_config():
	"""Load HSV filters from config.json -> {filters:{name:{low:[H,S,V], high:[H,S,V]}}}.
	Returns numpy array pairs for pink, gold, white, dispenser. Falls back to defaults.
	"""
	defaults = {
		"pink": {"low": [0, 0, 0], "high": [25, 255, 255]},
		"gold": {"low": [18, 0, 173], "high": [49, 168, 248]},
		"white": {"low": [0, 0, 162], "high": [180, 28, 255]},
		"dispenser": {"low": [0, 0, 97], "high": [180, 22, 142]},
	}
	base_path = get_base_path()
	cfg_path = os.path.join(base_path, "config.json")

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

	return (
		to_pair(filters_cfg, "pink"),
		to_pair(filters_cfg, "gold"),
		to_pair(filters_cfg, "white"),
		to_pair(filters_cfg, "dispenser"),
	)


filter_pink, filter_gold, filter_white, filter_dispenser = load_filters_config()
filter_selected = filter_pink  # Default

q_unicode = ord("q")
b_unicode = ord("b")
c_unicode = ord("c")
one_unicode = ord("1")
two_unicode = ord("2")
three_unicode = ord("3")
o_unicode = ord("o")
p_unicode = ord("p")
i_unicode = ord("i")
n_unicode = ord("n")
m_unicode = ord("m")
x_unicode = ord("x")

saved_reference_center = None
saved_dispenser_reference_center = None
saved_relative_diff = None


def load_config():
	"""Load IP/PORTs from config.json placed next to this script.
	Falls back to defaults if file is missing or invalid.
	"""
	defaults = {
		"ip": "192.168.10.25",
		"port_recv": 2000,
		"port_send": 2001,
		"pixel_mm_scale": 0.1,
		"tcp_x_factor": -1,
		"tcp_y_factor": -1,
	}
	base_path = get_base_path()
	cfg_path = os.path.join(base_path, "config.json")
	try:
		with open(cfg_path, "r", encoding="utf-8") as f:
			data = json.load(f)
		if isinstance(data, dict) and "server" in data and isinstance(data["server"], dict):
			data = data["server"]
		ip = str(data.get("ip", defaults["ip"]))
		port_recv = int(data.get("port_recv", data.get("port", defaults["port_recv"])))
		port_send = int(data.get("port_send", data.get("port", defaults["port_send"])))
		pixel_mm_scale = float(data.get("pixel_mm_scale", defaults["pixel_mm_scale"]))
		tcp_x_factor = float(data.get("tcp_x_factor", defaults["tcp_x_factor"]))
		tcp_y_factor = float(data.get("tcp_y_factor", defaults["tcp_y_factor"]))
		return ip, port_recv, port_send, pixel_mm_scale, tcp_x_factor, tcp_y_factor
	except Exception as e:
		print(f"[CONFIG] Using defaults. Reason: {e}")
		return (
			defaults["ip"],
			defaults["port_recv"],
			defaults["port_send"],
			defaults["pixel_mm_scale"],
			defaults["tcp_x_factor"],
			defaults["tcp_y_factor"],
		)


IP, PORT_RECV, PORT_SEND, PIXEL_TO_DELTA_SCALE, TCP_X_FACTOR, TCP_Y_FACTOR = load_config()

# Shared state for detection results + remote control
state_lock = threading.Lock()
latest_state = {
	"x_diff": 0,
	"y_diff": 0,
	"relative_x_diff": 0,
	"relative_y_diff": 0,
	"coords": None,            # Principal contour center (x, y)
	"dispenser_coords": None,  # Dispenser contour center (x, y)
	"pending_filter": None,   # "pink" | "gold" | "white"
	"pending_center": False,  # True -> save dispenser center + relative diff reference
	"pending_dispenser_center": False,
	"pending_relative_reference": False,
	"last_response": b"0,0\n", # Latest response generated by recv server
	"operator_id": None,
	"item_id": None,
	"pending_reset_operator": False,
	"pending_reset_item": False,
}


def _sanitize_filename_token(value):
	token = str(value or "").strip()
	if not token:
		return "na"
	token = re.sub(r"[^A-Za-z0-9]+", "_", token)
	token = token.strip("_")
	return token or "na"


def save_delta_payload(x, y, operator_id, item_id, now):
	date_str = now.strftime("%Y-%m-%d")
	time_str = now.strftime("%H-%M-%S")
	payload = f"{x},{y},{operator_id},{item_id},{date_str},{time_str}\n".encode("utf-8")

	timestamp = now.strftime("%Y%m%d_%H%M%S_%f")
	operator_token = _sanitize_filename_token(operator_id)
	item_token = _sanitize_filename_token(item_id)
	filename = f"{timestamp}_{operator_token}_{item_token}.txt"

	base_path = get_base_path()
	data_dir = os.path.join(base_path, "data")
	os.makedirs(data_dir, exist_ok=True)
	file_path = os.path.join(data_dir, filename)

	with open(file_path, "wb") as f:
		f.write(payload)

	return payload, file_path


def compute_scaled_delta(reference_point, current_point):
	x_diff = (reference_point[0] - current_point[0]) * PIXEL_TO_DELTA_SCALE
	y_diff = (reference_point[1] - current_point[1]) * PIXEL_TO_DELTA_SCALE
	return round(x_diff, 2), round(y_diff, 2)


def format_tcp_response(x_value, y_value):
	response_x = round(x_value * TCP_X_FACTOR, 2)
	response_y = round(y_value * TCP_Y_FACTOR, 2)
	response_text = f"{response_x},{response_y}"
	return response_text.encode("utf-8") + b"\n", response_text


def detect_contour_center(frame, area_coords, filter_pair, edge_mode):
	x1_area, y1_area = area_coords[0]
	x2_area, y2_area = area_coords[1]
	subframe = frame[y1_area:y2_area, x1_area:x2_area]

	result = {
		"mask": None,
		"center": None,
		"target_x": None,
		"y_top": None,
		"y_bottom": None,
		"has_detection": False,
	}

	if subframe.size == 0:
		return result

	low, upp = filter_pair
	hsv = cv2.cvtColor(subframe, cv2.COLOR_BGR2HSV)
	mask = cv2.inRange(hsv, low, upp)
	contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	result["mask"] = mask
	if not contours:
		return result

	contour = max(contours, key=cv2.contourArea)
	filled = np.zeros(mask.shape, np.uint8)
	cv2.drawContours(filled, [contour], -1, 255, thickness=cv2.FILLED)

	if edge_mode == "right":
		x_edge = int(contour[:, 0, 0].max()) - 5
	else:
		x_edge = int(contour[:, 0, 0].min()) + 5

	target_x = int(np.clip(x_edge, 0, filled.shape[1] - 1))
	ys = np.where(filled[:, target_x] > 0)[0]
	if ys.size == 0:
		return result

	y_top = int(ys.min())
	y_bottom = int(ys.max())
	pt_top = (x1_area + target_x, y1_area + y_top)
	pt_bottom = (x1_area + target_x, y1_area + y_bottom)
	center = helpers.calculate_middle_point(pt_top, pt_bottom)

	result["center"] = center
	result["target_x"] = target_x
	result["y_top"] = y_top
	result["y_bottom"] = y_bottom
	result["has_detection"] = True
	return result


def normalize_area(area_coords, default_area):
	area = default_area
	if isinstance(area_coords, list) and len(area_coords) == 2:
		(ax1, ay1), (ax2, ay2) = area_coords
		area = ((int(ax1), int(ay1)), (int(ax2), int(ay2)))
	elif isinstance(area_coords, dict):
		ax1 = int(area_coords.get("x1", default_area[0][0]))
		ay1 = int(area_coords.get("y1", default_area[0][1]))
		ax2 = int(area_coords.get("x2", default_area[1][0]))
		ay2 = int(area_coords.get("y2", default_area[1][1]))
		area = ((ax1, ay1), (ax2, ay2))

	(ax1, ay1), (ax2, ay2) = area
	x1n, x2n = (ax1, ax2) if ax1 <= ax2 else (ax2, ax1)
	y1n, y2n = (ay1, ay2) if ay1 <= ay2 else (ay2, ay1)
	return ((x1n, y1n), (x2n, y2n))


def run_glue_pipeline(frame, roi_coords, filter_pair):
	glue_frame = frame.copy()
	cv2.rectangle(glue_frame, roi_coords[0], roi_coords[1], (255, 0, 255), 2)

	x1_roi, y1_roi = roi_coords[0]
	x2_roi, y2_roi = roi_coords[1]
	frame_cropped = glue_frame[y1_roi:y2_roi, x1_roi:x2_roi]
	if frame_cropped.size == 0:
		return glue_frame

	low, upp = filter_pair
	hsv = cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2HSV)
	mask = cv2.inRange(hsv, low, upp)
	filtered = cv2.bitwise_and(frame_cropped, frame_cropped, mask=mask)

	gray_cropped = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
	_, binary_cropped = cv2.threshold(gray_cropped, 1, 255, cv2.THRESH_BINARY)
	contours, _ = cv2.findContours(binary_cropped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	if not contours:
		return glue_frame

	largest_contour = max(contours, key=cv2.contourArea)
	contour_mask = np.zeros(filtered.shape[:2], dtype=np.uint8)
	cv2.drawContours(contour_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
	cropped_obj = cv2.bitwise_and(filtered, filtered, mask=contour_mask)

	gray_obj = cv2.cvtColor(cropped_obj, cv2.COLOR_BGR2GRAY)
	blurred_obj = cv2.GaussianBlur(gray_obj, (5, 5), 0)
	edges = cv2.Canny(blurred_obj, 10, 20)

	edge_contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	for contour in edge_contours:
		contour_x, contour_y, contour_w, contour_h = cv2.boundingRect(contour)
		if 0 < contour_w < 30:
			contour_in_frame = contour + np.array([[[x1_roi, y1_roi]]])
			cv2.drawContours(glue_frame, [contour_in_frame], -1, (0, 255, 0), 2)

	return glue_frame


def _prompt_text_with_tk(title, prompt):
	"""Ask for a non-empty value in a Tkinter window and submit on Enter."""
	while True:
		try:
			root = tk.Tk()
			root.title(title)
			root.resizable(False, False)
			root.attributes("-topmost", True)
		except tk.TclError:
			value = input(f"{prompt} ").strip()
			if value:
				return value
			continue

		value_ref = {"value": None}

		frame = tk.Frame(root, padx=16, pady=12)
		frame.pack()

		label = tk.Label(frame, text=prompt)
		label.pack(anchor="w")

		entry = tk.Entry(frame, width=32)
		entry.pack(fill="x", pady=(6, 4))
		entry.focus_set()

		error_label = tk.Label(frame, text="", fg="red")
		error_label.pack(anchor="w")

		def submit(event=None):
			value = entry.get().strip()
			if not value:
				error_label.config(text="El valor no puede estar vacio.")
				return
			value_ref["value"] = value
			root.destroy()

		submit_btn = tk.Button(frame, text="Submit", command=submit)
		submit_btn.pack(pady=(8, 0))

		root.bind("<Return>", submit)
		root.protocol("WM_DELETE_WINDOW", lambda: None)
		root.mainloop()

		if value_ref["value"]:
			return value_ref["value"]


def request_operator_and_item():
	operator_id = _prompt_text_with_tk("ID Operador", "Ingresa ID de operador:")
	item_id = _prompt_text_with_tk("ID Item", "Ingresa ID de item:")
	with state_lock:
		latest_state["operator_id"] = operator_id
		latest_state["item_id"] = item_id
	print(f"[IDS] operador={operator_id} item={item_id}")


def request_item_only():
	item_id = _prompt_text_with_tk("ID Item", "Ingresa ID de item:")
	with state_lock:
		latest_state["item_id"] = item_id
	print(f"[IDS] item={item_id}")


def process_pending_id_requests():
	do_reset_operator = False
	do_reset_item = False

	with state_lock:
		do_reset_operator = latest_state["pending_reset_operator"]
		do_reset_item = latest_state["pending_reset_item"]

		if do_reset_operator:
			latest_state["pending_reset_operator"] = False
			latest_state["pending_reset_item"] = False
		elif do_reset_item:
			latest_state["pending_reset_item"] = False

	if do_reset_operator:
		request_operator_and_item()
	elif do_reset_item:
		request_item_only()


request_operator_and_item()

def mouse_cb(event, x, y, flags, param):
	if event == cv2.EVENT_LBUTTONDOWN:
		print(f"({x}, {y})")

cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", mouse_cb)

def recv_server_thread():
	server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
	try:
		server.bind((IP, PORT_RECV))
	except OSError as e:
		print(f"[RECV SERVER BIND ERROR] {e}")
		return

	server.listen()
	print(f"[LISTENING] Recv server listening on {IP}:{PORT_RECV}")

	while True:
		try:
			conn, addr = server.accept()
		except OSError as e:
			print(f"[ACCEPT ERROR] {e}")
			sleep(0.1)
			continue

		print(f"[CONNECTED] {addr}")
		with conn:
			conn.settimeout(TCP_RECV_POLL_TIMEOUT_S)
			while True:
				delta_saved_path = None
				try:
					chunk = conn.recv(1024)
				except socket.timeout:
					continue
				except Exception as e:
					print(f"[TCP RX ERROR] {e}")
					break

				if not chunk:
					print("[DISCONNECTED] peer closed connection")
					break

				data = chunk
				debug_recv_data(data)
				cmd = extract_cmd(data)

				print("comando:")
				print(cmd)

				with state_lock:
					if cmd in ("pink", "gold", "white"):
						latest_state["pending_filter"] = cmd
						response = b"OK\n"

					elif cmd == "center":
						latest_state["pending_center"] = True
						response = b"OK\n"

					elif cmd == "dispenser_center":
						latest_state["pending_dispenser_center"] = True
						response = b"OK\n"

					elif cmd == "relative_reference":
						latest_state["pending_relative_reference"] = True
						response = b"OK\n"

					elif cmd == "reset_operator":
						latest_state["operator_id"] = None
						latest_state["item_id"] = None
						latest_state["pending_reset_operator"] = True
						latest_state["pending_reset_item"] = False
						response = b"OK\n"

					elif cmd == "reset_item":
						latest_state["item_id"] = None
						latest_state["pending_reset_item"] = True
						response = b"OK\n"

					elif cmd == "delta":
						x = latest_state["relative_x_diff"]
						y = latest_state["relative_y_diff"]
						operator_id = latest_state["operator_id"] or ""
						item_id = latest_state["item_id"] or ""
						now = datetime.now()
						payload, delta_saved_path = save_delta_payload(x, y, operator_id, item_id, now)
						response, response_text = format_tcp_response(x, y)
						print(f"[TCP Z RESPONSE] {response_text}")
						# response = payload

					elif cmd == "relative_delta":
						x = latest_state["relative_x_diff"]
						y = latest_state["relative_y_diff"]
						operator_id = latest_state["operator_id"] or ""
						item_id = latest_state["item_id"] or ""
						now = datetime.now()
						payload, delta_saved_path = save_delta_payload(x, y, operator_id, item_id, now)
						response, _ = format_tcp_response(x, y)

					else:
						x = latest_state["relative_x_diff"]
						y = latest_state["relative_y_diff"]
						response, _ = format_tcp_response(x, y)

					latest_state["last_response"] = response

				try:
					conn.sendall(response)
					print(f"[SENT] to {addr}: {response.decode('utf-8', errors='ignore').strip()}")
					if delta_saved_path:
						print(f"[DATA] saved {delta_saved_path}")
				except Exception as e:
					print(f"[SEND ERROR] {e}")
					break


def send_server_thread():
	server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
	try:
		server.bind((IP, PORT_SEND))
	except OSError as e:
		print(f"[SEND SERVER BIND ERROR] {e}")
		return

	server.listen()
	print(f"[LISTENING] Send server listening on {IP}:{PORT_SEND}")

	while True:
		try:
			conn, addr = server.accept()
		except OSError as e:
			print(f"[SEND ACCEPT ERROR] {e}")
			sleep(0.1)
			continue

		with conn:
			# Keep connection behavior similar to crunch.py: one accept -> one response.
			try:
				conn.settimeout(TCP_RECV_POLL_TIMEOUT_S)
				conn.recv(1024)
			except socket.timeout:
				pass
			except Exception:
				pass

			with state_lock:
				response = latest_state["last_response"]
				operator_id = latest_state["operator_id"]
				item_id = latest_state["item_id"]

			if not isinstance(response, (bytes, bytearray)):
				response = f"{response}\n".encode("utf-8")

			try:
				conn.sendall(response)
				print(
					f"[SEND SERVER] to {addr}: {response.decode('utf-8', errors='ignore').strip()} "
					f"(operador={operator_id}, item={item_id})"
				)
			except Exception as e:
				print(f"[SEND SERVER ERROR] {e}")


threading.Thread(target=recv_server_thread, daemon=True).start()
threading.Thread(target=send_server_thread, daemon=True).start()

def load_camera_area_config():
	"""Load camera index and detection area from config.json + extra options."""
	defaults_index = 0
	defaults_area = ((250, 190), (390, 290))
	defaults_dispenser_area = ((190, 290), (290, 490))
	defaults_glue_roi = ((250, 190), (390, 290))
	defaults_edge = "left"

	base_path = get_base_path()
	cfg_path = os.path.join(base_path, "config.json")
	cam_idx = defaults_index
	area = defaults_area
	area_dispenser = defaults_dispenser_area
	glue_roi = defaults_glue_roi
	is_rotate = False
	is_rotate90 = False
	edge = defaults_edge

	try:
		with open(cfg_path, "r", encoding="utf-8") as f:
			data = json.load(f)

		if isinstance(data, dict):
			is_rotate = bool(data.get("rotate", False))
			is_rotate90 = bool(data.get("rotate90", False))

			edge = str(data.get("edge", defaults_edge)).strip().lower()
			if edge not in ("left", "right"):
				edge = defaults_edge

			if "camera_index" in data:
				cam_idx = int(data.get("camera_index", defaults_index))
			elif "camera" in data and isinstance(data["camera"], dict):
				cam_idx = int(data["camera"].get("index", defaults_index))

			raw_area = data.get("area")
			if raw_area is None and isinstance(data.get("detection"), dict):
				raw_area = data["detection"].get("area")
			raw_area_dispenser = data.get("area_dispenser")
			raw_glue_roi = data.get("roi")

			area = normalize_area(raw_area, defaults_area)
			area_dispenser = normalize_area(raw_area_dispenser, defaults_dispenser_area)
			glue_roi = normalize_area(raw_glue_roi, defaults_glue_roi)

	except Exception as e:
		print(f"[CONFIG:CAM/AREA] Using defaults. Reason: {e}")

	return cam_idx, area, area_dispenser, glue_roi, is_rotate, is_rotate90, edge


camera_index, area, area_dispenser, glue_roi, is_rotate, is_rotate90, edge = load_camera_area_config()
(x1, y1), (x2, y2) = area
(dx1, dy1), (dx2, dy2) = area_dispenser

print(f"[CONFIG] area={area}")
print(f"[CONFIG] area_dispenser={area_dispenser}")
print(f"[CONFIG] roi={glue_roi}")
print(f"[CONFIG] rotate={is_rotate} rotate90={is_rotate90}")

cam = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
is_frame_ok = False

while not cam.isOpened() and not is_frame_ok:
	cam = cv2.VideoCapture(camera_index)
	is_frame_ok, _ = cam.read()
	print("Waiting for camera...")
	sleep(0.05)

logged_frame_shape = False

while True:
	process_pending_id_requests()

	is_frame_ok, frame = cam.read()
	if not is_frame_ok or frame is None:
		continue

	if is_rotate:
		frame = cv2.rotate(frame, cv2.ROTATE_180)

	if is_rotate90:
		frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

	if not logged_frame_shape:
		print(f"[FRAME] shape_after_rotation={frame.shape}")
		logged_frame_shape = True

	glue_frame = run_glue_pipeline(frame, glue_roi, filter_pink)
	main_detection = detect_contour_center(frame, area, filter_selected, edge)
	dispenser_detection = detect_contour_center(frame, area_dispenser, filter_dispenser, edge)
	msk = main_detection["mask"] if main_detection["mask"] is not None else np.zeros((1, 1), dtype=np.uint8)
	dispenser_msk = (
		dispenser_detection["mask"] if dispenser_detection["mask"] is not None else np.zeros((1, 1), dtype=np.uint8)
	)
	pt_center = main_detection["center"]
	dispenser_center = dispenser_detection["center"]

	cv2.rectangle(frame, area[0], area[1], (0, 255, 0), 2)
	cv2.rectangle(frame, area_dispenser[0], area_dispenser[1], (255, 128, 0), 2)

	if pt_center is not None:
		cv2.circle(frame, pt_center, 2, (0, 255, 255), -1)

		with state_lock:
			latest_state["coords"] = (int(pt_center[0]), int(pt_center[1]))

		if saved_reference_center is not None:
			cv2.line(frame, pt_center, saved_reference_center, (255, 255, 255), 2)
			cv2.circle(frame, saved_reference_center, 5, (255, 0, 0), -1)

			x_diff, y_diff = compute_scaled_delta(saved_reference_center, pt_center)

			with state_lock:
				latest_state["x_diff"] = x_diff
				latest_state["y_diff"] = y_diff

			text_coords = (x1 - 60, y2 - 120)
			cv2.putText(
				frame,
				f"Correccion: {x_diff}, {y_diff}",
				text_coords,
				cv2.FONT_HERSHEY_SIMPLEX,
				1,
				(0, 255, 0),
				1,
			)
	else:
		with state_lock:
			latest_state["coords"] = None

	if dispenser_center is not None:
		cv2.circle(frame, dispenser_center, 2, (0, 165, 255), -1)
		with state_lock:
			latest_state["dispenser_coords"] = (int(dispenser_center[0]), int(dispenser_center[1]))

		if saved_dispenser_reference_center is not None:
			cv2.line(frame, dispenser_center, saved_dispenser_reference_center, (0, 165, 255), 2)
			cv2.circle(frame, saved_dispenser_reference_center, 5, (0, 0, 255), -1)
	else:
		with state_lock:
			latest_state["dispenser_coords"] = None

	current_relative_diff = None
	if pt_center is not None and dispenser_center is not None:
		current_relative_diff = (
			dispenser_center[0] - pt_center[0],
			dispenser_center[1] - pt_center[1],
		)

	if current_relative_diff is not None and saved_relative_diff is not None:
		relative_x_diff, relative_y_diff = compute_scaled_delta(saved_relative_diff, current_relative_diff)
		with state_lock:
			latest_state["relative_x_diff"] = relative_x_diff
			latest_state["relative_y_diff"] = relative_y_diff

		relative_text_coords = (dx1 - 60, dy2 - 20)
		cv2.putText(
			frame,
			f"Disp rel: {relative_x_diff}, {relative_y_diff}",
			relative_text_coords,
			cv2.FONT_HERSHEY_SIMPLEX,
			0.8,
			(0, 165, 255),
			1,
		)
	else:
		with state_lock:
			latest_state["relative_x_diff"] = 0
			latest_state["relative_y_diff"] = 0

	# Apply pending requests from TCP (filter / center) AFTER pt_center is computed
	with state_lock:
		pending = latest_state["pending_filter"]
		do_center = latest_state["pending_center"]
		do_dispenser_center = latest_state["pending_dispenser_center"]
		do_relative_reference = latest_state["pending_relative_reference"]
		latest_state["pending_filter"] = None
		latest_state["pending_center"] = False
		latest_state["pending_dispenser_center"] = False
		latest_state["pending_relative_reference"] = False

	if pending:
		if pending == "pink":
			filter_selected = filter_pink
		elif pending == "gold":
			filter_selected = filter_gold
		elif pending == "white":
			filter_selected = filter_white
		saved_reference_center = None
		saved_relative_diff = None

	if do_center and pt_center is not None and dispenser_center is not None:
		saved_dispenser_reference_center = dispenser_center
		if current_relative_diff is not None:
			saved_relative_diff = current_relative_diff

	if do_dispenser_center and dispenser_center is not None:
		saved_dispenser_reference_center = dispenser_center

	if do_relative_reference and current_relative_diff is not None:
		saved_relative_diff = current_relative_diff

	cv2.imshow("Frame", frame)
	cv2.imshow("Glue Frame", glue_frame)
	cv2.imshow("Filtered", msk)
	cv2.imshow("Filtered Dispenser", dispenser_msk)

	key = cv2.waitKey(1)

	if key == q_unicode:
		break

	if key == b_unicode and pt_center is not None:
		saved_reference_center = pt_center

	if key == c_unicode and pt_center is not None and dispenser_center is not None:
		saved_dispenser_reference_center = dispenser_center
		if current_relative_diff is not None:
			saved_relative_diff = current_relative_diff

	if key == n_unicode and dispenser_center is not None:
		saved_dispenser_reference_center = dispenser_center

	if key == m_unicode and current_relative_diff is not None:
		saved_relative_diff = current_relative_diff

	if key == x_unicode:
		with state_lock:
			relative_x = latest_state["relative_x_diff"]
			relative_y = latest_state["relative_y_diff"]
		_, response_text = format_tcp_response(relative_x, relative_y)
		print(f"[RELATIVE DELTA] {response_text}")

	if key == one_unicode:
		filter_selected = filter_pink
		saved_reference_center = None
		saved_relative_diff = None

	if key == two_unicode:
		filter_selected = filter_gold
		saved_reference_center = None
		saved_relative_diff = None

	if key == three_unicode:
		filter_selected = filter_white
		saved_reference_center = None
		saved_relative_diff = None

	if key == p_unicode:
		filter_selected = filter_pink
		saved_reference_center = None
		saved_relative_diff = None

	if key == o_unicode:
		with state_lock:
			latest_state["operator_id"] = None
			latest_state["item_id"] = None
			latest_state["pending_reset_operator"] = True
			latest_state["pending_reset_item"] = False

	if key == i_unicode:
		with state_lock:
			latest_state["item_id"] = None
			latest_state["pending_reset_item"] = True

cam.release()
cv2.destroyAllWindows()
