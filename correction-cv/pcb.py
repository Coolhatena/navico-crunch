""" A template script for computer vision projects """
import cv2
from time import sleep
import numpy as np
import helpers
import socket
import threading
import json
import os
import sys
import re
import struct
import tkinter as tk
from datetime import datetime

TCP_DEBUG_RECV = True
TCP_RECV_POLL_TIMEOUT_S = 0.5
PIXEL_TO_DELTA_SCALE = 0.1
TCP_X_FACTOR = -1
TCP_Y_FACTOR = -1
VALID_TCP_CMDS = ("pink", "gold", "white", "p", "g", "w", "c", "d", "r", "z", "x", "o", "i", "s")
ITEM_RESET_CMD_PREFIX = "reset_items_by_command:"


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
	"""Return first valid command token and canonical response char found in payload."""
	text = data.decode("utf-8", errors="ignore").strip().lower()
	if not text:
		return "", None

	aliases = {
		"pink": ("pink", "p"),
		"gold": ("gold", "g"),
		"white": ("white", "w"),
		"p": ("pink", "p"),
		"g": ("gold", "g"),
		"w": ("white", "w"),
		"c": ("center", "c"),
		"d": ("dispenser_center", "d"),
		"r": ("relative_reference", "r"),
		"z": ("delta", "z"),
		"x": ("relative_delta", "x"),
		"o": ("reset_operator", "o"),
		"i": ("request_engineer_auth", "i"),
		"s": ("reset_item", "s"),
	}

	tokens = re.split(r"[^a-z0-9_]+", text)
	for token in tokens:
		if token in aliases and token in VALID_TCP_CMDS:
			return aliases[token]

	if "pink" in text:
		return "pink", "p"
	if "gold" in text:
		return "gold", "g"
	if "white" in text:
		return "white", "w"
	if "dispenser" in text:
		return "dispenser_center", "d"
	if "center" in text:
		return "center", "c"
	if "relative" in text:
		return "relative_reference", "r"

	for token in tokens:
		if token and get_item_configs_for_command(token):
			return f"{ITEM_RESET_CMD_PREFIX}{token}", token

	return "", None

def get_base_path():
	if getattr(sys, "frozen", False):  # ejecutándose como .exe
		return os.path.dirname(sys.executable)
	return os.path.dirname(os.path.abspath(__file__))

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
z_unicode = ord("z")
g_unicode = ord("g")
w_unicode = ord("w")
d_unicode = ord("d")
r_unicode = ord("r")
s_unicode = ord("s")

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


def load_auth_request_config():
	"""Load authorization request storage config from config.json next to script/.exe."""
	defaults = {
		"output_dir": "data/auth",
	}
	base_path = get_base_path()
	cfg_path = os.path.join(base_path, "config.json")
	try:
		with open(cfg_path, "r", encoding="utf-8") as f:
			data = json.load(f)
		output_dir = str(data.get("auth_request_output_dir", defaults["output_dir"])) if isinstance(data, dict) else defaults["output_dir"]
		return output_dir
	except Exception as e:
		print(f"[CONFIG:AUTH] Using defaults. Reason: {e}")
		return defaults["output_dir"]


AUTH_REQUEST_OUTPUT_DIR = load_auth_request_config()


def load_crunch_coms_config():
	"""Load integrated crunch listener config from config.json next to script/.exe."""
	defaults = {
		"enabled": True,
		"ip": "192.168.10.34",
		"port": 2000,
		"size": 2048,
		"real_count": 360,
		"real_stride": 4,
		"real_start": 0,
		"output_dir": "data/crunch",
	}
	base_path = get_base_path()
	cfg_path = os.path.join(base_path, "config.json")
	try:
		with open(cfg_path, "r", encoding="utf-8") as f:
			data = json.load(f)
		if isinstance(data, dict) and isinstance(data.get("crunch_coms"), dict):
			data = data["crunch_coms"]
		else:
			data = {}

		return (
			bool(data.get("enabled", defaults["enabled"])),
			str(data.get("ip", defaults["ip"])),
			int(data.get("port", defaults["port"])),
			int(data.get("size", defaults["size"])),
			int(data.get("real_count", defaults["real_count"])),
			int(data.get("real_stride", defaults["real_stride"])),
			int(data.get("real_start", defaults["real_start"])),
			str(data.get("output_dir", defaults["output_dir"])),
		)
	except Exception as e:
		print(f"[CONFIG:CRUNCH] Using defaults. Reason: {e}")
		return (
			defaults["enabled"],
			defaults["ip"],
			defaults["port"],
			defaults["size"],
			defaults["real_count"],
			defaults["real_stride"],
			defaults["real_start"],
			defaults["output_dir"],
		)


def load_item_ids_config():
	"""Load dynamic item id prompts from config.json next to script/.exe."""
	base_path = get_base_path()
	cfg_path = os.path.join(base_path, "config.json")
	try:
		with open(cfg_path, "r", encoding="utf-8") as f:
			data = json.load(f)
	except Exception as e:
		print(f"[CONFIG:ITEM IDS] Using defaults. Reason: {e}")
		return []

	if not isinstance(data, dict):
		return []

	items = data.get("item_ids")
	if not isinstance(items, list):
		return []

	valid_items = []
	for raw_item in items:
		if not isinstance(raw_item, dict):
			continue
		label = str(raw_item.get("label", "")).strip()
		key = str(raw_item.get("key", "")).strip()
		command_char = str(raw_item.get("command_char", "")).strip().lower()
		if not label or not key:
			continue
		valid_items.append({"label": label, "key": key, "command_char": command_char})

	return valid_items


def get_item_configs_for_command(command_char):
	command_char = str(command_char or "").strip().lower()
	if not command_char:
		return []
	return [item_cfg for item_cfg in ITEM_ID_CONFIGS if item_cfg.get("command_char") == command_char]


(
	CRUNCH_COMS_ENABLED,
	CRUNCH_COMS_IP,
	CRUNCH_COMS_PORT,
	CRUNCH_COMS_SIZE,
	CRUNCH_REAL_COUNT,
	CRUNCH_REAL_STRIDE,
	CRUNCH_REAL_START,
	CRUNCH_OUTPUT_DIR,
) = load_crunch_coms_config()
ITEM_ID_CONFIGS = load_item_ids_config()

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
	"item_ids": {},
	"pending_reset_operator": False,
	"pending_reset_item": False,
	"pending_reset_item_command_char": None,
	"reset_operator_in_progress": False,
	"reset_operator_result": None,
}
reset_operator_done_event = threading.Event()


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

	data_dir = os.path.join(os.path.dirname(__file__), "data")
	os.makedirs(data_dir, exist_ok=True)
	file_path = os.path.join(data_dir, filename)

	with open(file_path, "wb") as f:
		f.write(payload)

	return payload, file_path


def read_real_at(data: bytes, offset: int) -> float | None:
	end = offset + 4
	if end > len(data):
		return None
	return struct.unpack(">f", data[offset:end])[0]


def read_real_array(data: bytes, start_offset: int, count: int, stride: int = 4):
	values = []
	for index in range(count):
		offset = start_offset + index * stride
		values.append(read_real_at(data, offset))
	return values


def _format_item_ids_for_single_field(item_ids):
	if not item_ids:
		return ""
	parts = []
	for item_cfg in ITEM_ID_CONFIGS:
		key = item_cfg["key"]
		value = str(item_ids.get(key, "")).strip()
		if value:
			parts.append(f"{key}={value}")
	return "|".join(parts)


def format_crunch_payload(operator_id, item_ids, now, values):
	timestamp = now.strftime("%Y%m%d_%H%M%S_%f")
	parts = [f"id_operador: {operator_id or ''}"]
	for item_cfg in ITEM_ID_CONFIGS:
		key = item_cfg["key"]
		value = "" if not isinstance(item_ids, dict) else str(item_ids.get(key, "") or "")
		parts.append(f"{key}: {value}")
	data_lines = "\n".join("" if value is None else str(value) for value in values)
	parts.append(f"timestamp: {timestamp}")
	header_text = ", ".join(parts)
	return f"{header_text}\ndatos:\n{data_lines}\n", timestamp


def save_crunch_payload(operator_id, item_ids, now, values):
	payload_text, timestamp = format_crunch_payload(operator_id, item_ids, now, values)
	output_dir = os.path.join(get_base_path(), CRUNCH_OUTPUT_DIR)
	os.makedirs(output_dir, exist_ok=True)

	operator_token = _sanitize_filename_token(operator_id)
	item_token = _sanitize_filename_token(_format_item_ids_for_single_field(item_ids))
	file_path = os.path.join(output_dir, f"{timestamp}_{operator_token}_{item_token}.txt")
	with open(file_path, "w", encoding="utf-8") as f:
		f.write(payload_text)

	return payload_text.encode("utf-8"), file_path


def save_auth_request_result(result_text, now):
	timestamp = now.strftime("%Y%m%d_%H%M%S_%f")
	content_timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
	output_dir = os.path.join(get_base_path(), AUTH_REQUEST_OUTPUT_DIR)
	os.makedirs(output_dir, exist_ok=True)
	file_path = os.path.join(output_dir, f"{timestamp}.txt")
	with open(file_path, "w", encoding="utf-8") as f:
		f.write(f"{result_text} {content_timestamp}")
	return file_path


def compute_scaled_delta(reference_point, current_point):
	x_diff = (reference_point[0] - current_point[0]) * PIXEL_TO_DELTA_SCALE
	y_diff = (reference_point[1] - current_point[1]) * PIXEL_TO_DELTA_SCALE
	return round(x_diff, 2), round(y_diff, 2)


def format_tcp_response(x_value, y_value):
	response_x = round(x_value * TCP_X_FACTOR, 2)
	response_y = round(y_value * TCP_Y_FACTOR, 2)
	# response_text = f"{response_x},{response_y}" // Por el momento, solo se enviara Y
	response_text = f"0.0,{response_y}"
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


def _prompt_text_with_tk(title, prompt, require_non_empty=True):
	"""Prompt for text using Tkinter, optionally allowing empty/cancelled results."""
	while True:
		try:
			root = tk.Tk()
			root.title(title)
			root.resizable(False, False)
			root.attributes("-topmost", True)
			root.geometry("960x540")
			root.minsize(960, 540)
		except tk.TclError:
			value = input(f"{prompt} ").strip()
			if require_non_empty and not value:
				continue
			return value

		value_ref = {"value": None}

		frame = tk.Frame(root, padx=48, pady=36)
		frame.pack(fill="both", expand=True)

		label = tk.Label(frame, text=prompt, font=("TkDefaultFont", 30))
		label.pack(anchor="w", pady=(0, 18))

		entry = tk.Entry(frame, width=60, font=("TkDefaultFont", 30))
		entry.pack(fill="x", ipady=16, pady=(0, 12))
		entry.focus_set()

		error_label = tk.Label(frame, text="", fg="red", font=("TkDefaultFont", 22))
		error_label.pack(anchor="w")

		def submit(event=None):
			value = entry.get().strip()
			if require_non_empty and not value:
				error_label.config(text="El valor no puede estar vacio.")
				return
			value_ref["value"] = value
			root.destroy()

		def on_close():
			value_ref["value"] = ""
			root.destroy()

		submit_btn = tk.Button(
			frame,
			text="Submit",
			command=submit,
			font=("TkDefaultFont", 26),
			padx=36,
			pady=18,
		)
		submit_btn.pack(pady=(24, 0), anchor="center")

		root.bind("<Return>", submit)
		root.protocol("WM_DELETE_WINDOW", lambda: None if require_non_empty else on_close())
		root.mainloop()

		if value_ref["value"] is not None:
			return value_ref["value"]


def request_engineer_authorization():
	return _prompt_text_with_tk("Autorizacion", "Escanee el id de ingeniero", require_non_empty=False)


def get_key_command(key):
	key_map = {
		p_unicode: "pink",
		g_unicode: "gold",
		w_unicode: "white",
		c_unicode: "center",
		d_unicode: "dispenser_center",
		r_unicode: "relative_reference",
		z_unicode: "delta",
		x_unicode: "relative_delta",
		o_unicode: "reset_operator",
		i_unicode: "request_engineer_auth",
		s_unicode: "reset_item",
	}
	return key_map.get(key)


def get_command_response_char(cmd):
	response_chars = {
		"pink": "p",
		"gold": "g",
		"white": "w",
		"center": "c",
		"dispenser_center": "d",
		"relative_reference": "r",
		"reset_operator": "o",
		"request_engineer_auth": "i",
		"reset_item": "s",
	}
	if cmd.startswith(ITEM_RESET_CMD_PREFIX):
		return cmd[len(ITEM_RESET_CMD_PREFIX):]
	return response_chars.get(cmd)


def format_command_response(response_char, success=True):
	if not response_char:
		response_char = ""
	response_text = str(response_char).strip().lower()
	if not success:
		response_text = response_text.upper()
	return response_text.encode("utf-8"), response_text


def request_required_text(title, prompt):
	while True:
		value = _prompt_text_with_tk(title, prompt, require_non_empty=False)
		if value:
			return value


def request_item_ids(item_configs):
	item_values = {}
	for item_cfg in item_configs:
		label = item_cfg["label"]
		key = item_cfg["key"]
		item_values[key] = request_required_text("ID Item", f"Escanee el codigo: {label}")
	return item_values


def request_all_item_ids():
	return request_item_ids(ITEM_ID_CONFIGS)


def request_operator_and_item():
	operator_id = request_required_text("ID Operador", "Escanee el id de operador")
	item_ids = request_all_item_ids()
	with state_lock:
		latest_state["operator_id"] = operator_id
		latest_state["item_ids"] = item_ids
	print(f"[IDS] operador={operator_id} items={item_ids}")


def request_item_only(item_configs=None):
	item_ids = request_item_ids(item_configs or ITEM_ID_CONFIGS)
	with state_lock:
		latest_state["item_ids"].update(item_ids)
	print(f"[IDS] items={item_ids}")


def process_pending_id_requests():
	do_reset_operator = False
	do_reset_item = False

	with state_lock:
		do_reset_operator = latest_state["pending_reset_operator"]
		do_reset_item = latest_state["pending_reset_item"]
		reset_item_command_char = latest_state["pending_reset_item_command_char"]

		if do_reset_operator:
			latest_state["pending_reset_operator"] = False
			latest_state["pending_reset_item"] = False
			latest_state["pending_reset_item_command_char"] = None
		elif do_reset_item:
			latest_state["pending_reset_item"] = False
			latest_state["pending_reset_item_command_char"] = None

	if do_reset_operator:
		reset_success = False
		try:
			request_operator_and_item()
			reset_success = True
		except Exception as e:
			print(f"[RESET OPERATOR ERROR] {e}")
		finally:
			with state_lock:
				latest_state["reset_operator_in_progress"] = False
				latest_state["reset_operator_result"] = reset_success
			reset_operator_done_event.set()
	elif do_reset_item:
		item_configs = None
		if reset_item_command_char:
			item_configs = get_item_configs_for_command(reset_item_command_char)
			if not item_configs:
				print(f"[RESET ITEM WARN] No item_ids configured for command_char={reset_item_command_char!r}")
				return
		request_item_only(item_configs)


def handle_command(cmd, source="tcp", response_char=None):
	delta_saved_path = None
	auth_saved_path = None
	log_text = None
	if response_char is None:
		response_char = get_command_response_char(cmd)

	if cmd == "request_engineer_auth":
		auth_result = request_engineer_authorization()
		auth_saved_path = save_auth_request_result(auth_result, datetime.now())
		response, log_text = format_command_response(response_char, success=bool(auth_result))
		with state_lock:
			latest_state["last_response"] = response
		return response, log_text, delta_saved_path, auth_saved_path

	with state_lock:
		if cmd in ("pink", "gold", "white"):
			latest_state["pending_filter"] = cmd
			response, log_text = format_command_response(response_char)

		elif cmd == "center":
			latest_state["pending_center"] = True
			response, log_text = format_command_response(response_char)

		elif cmd == "dispenser_center":
			latest_state["pending_dispenser_center"] = True
			response, log_text = format_command_response(response_char)

		elif cmd == "relative_reference":
			latest_state["pending_relative_reference"] = True
			response, log_text = format_command_response(response_char)

		elif cmd == "reset_operator":
			if latest_state["pending_reset_operator"] or latest_state["reset_operator_in_progress"]:
				should_wait_reset = source == "tcp"
				response, log_text = format_command_response(response_char)
				log_text = f"{log_text} (reset_operator already pending)"
			else:
				latest_state["operator_id"] = None
				latest_state["item_ids"] = {}
				latest_state["pending_reset_operator"] = True
				latest_state["pending_reset_item"] = False
				latest_state["pending_reset_item_command_char"] = None
				latest_state["reset_operator_in_progress"] = True
				latest_state["reset_operator_result"] = None
				reset_operator_done_event.clear()
				should_wait_reset = source == "tcp"
				response, log_text = format_command_response(response_char)

		elif cmd == "reset_item":
			latest_state["item_ids"] = {}
			latest_state["pending_reset_item"] = True
			latest_state["pending_reset_item_command_char"] = None
			response, log_text = format_command_response(response_char)

		elif cmd.startswith(ITEM_RESET_CMD_PREFIX):
			command_char = cmd[len(ITEM_RESET_CMD_PREFIX):]
			item_configs = get_item_configs_for_command(command_char)
			if item_configs:
				for item_cfg in item_configs:
					latest_state["item_ids"].pop(item_cfg["key"], None)
				latest_state["pending_reset_item"] = True
				latest_state["pending_reset_item_command_char"] = command_char
				response, log_text = format_command_response(response_char)
			else:
				response, log_text = format_command_response(response_char, success=False)

		elif cmd == "delta":
			x = latest_state["relative_x_diff"]
			y = latest_state["relative_y_diff"]
			operator_id = latest_state["operator_id"] or ""
			item_id = _format_item_ids_for_single_field(latest_state["item_ids"])
			now = datetime.now()
			_, delta_saved_path = save_delta_payload(x, y, operator_id, item_id, now)
			response, log_text = format_tcp_response(x, y)

		elif cmd == "relative_delta":
			x = latest_state["relative_x_diff"]
			y = latest_state["relative_y_diff"]
			operator_id = latest_state["operator_id"] or ""
			item_id = _format_item_ids_for_single_field(latest_state["item_ids"])
			now = datetime.now()
			_, delta_saved_path = save_delta_payload(x, y, operator_id, item_id, now)
			response, log_text = format_tcp_response(x, y)

		else:
			x = latest_state["relative_x_diff"]
			y = latest_state["relative_y_diff"]
			response, log_text = format_tcp_response(x, y)

		latest_state["last_response"] = response
	if cmd == "reset_operator" and source == "tcp":
		reset_operator_done_event.wait()
		with state_lock:
			reset_success = bool(latest_state["reset_operator_result"])
		response, log_text = format_command_response(response_char, success=reset_success)
		with state_lock:
			latest_state["last_response"] = response

	return response, log_text, delta_saved_path, auth_saved_path


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
				cmd, response_char = extract_cmd(data)

				print("comando:")
				print(cmd)
				response, log_text, delta_saved_path, auth_saved_path = handle_command(
					cmd,
					source="tcp",
					response_char=response_char,
				)

				try:
					conn.sendall(response)
					print(f"[SENT] to {addr}: {response.decode('utf-8', errors='ignore').strip()}")
					if delta_saved_path:
						print(f"[DATA] saved {delta_saved_path}")
					if auth_saved_path:
						print(f"[AUTH DATA] saved {auth_saved_path}")
					if cmd == "delta" and log_text:
						print(f"[TCP Z RESPONSE] {log_text}")
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
				item_ids = dict(latest_state["item_ids"])

			if not isinstance(response, (bytes, bytearray)):
				response = f"{response}\n".encode("utf-8")

			try:
				conn.sendall(response)
				print(
					f"[SEND SERVER] to {addr}: {response.decode('utf-8', errors='ignore').strip()} "
					f"(operador={operator_id}, items={item_ids})"
				)
			except Exception as e:
				print(f"[SEND SERVER ERROR] {e}")


def crunch_recv_server_thread():
	if not CRUNCH_COMS_ENABLED:
		print("[CRUNCH COMMS] Disabled in config.")
		return

	server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
	try:
		server.bind((CRUNCH_COMS_IP, CRUNCH_COMS_PORT))
	except OSError as e:
		print(f"[CRUNCH BIND ERROR] {e}")
		return

	server.listen()
	print(f"[LISTENING] Crunch recv server listening on {CRUNCH_COMS_IP}:{CRUNCH_COMS_PORT}")

	while True:
		try:
			conn, addr = server.accept()
		except OSError as e:
			print(f"[CRUNCH ACCEPT ERROR] {e}")
			sleep(0.1)
			continue

		print(f"[CRUNCH CONNECTED] {addr}")
		with conn:
			try:
				data = conn.recv(CRUNCH_COMS_SIZE)
			except Exception as e:
				print(f"[CRUNCH RX ERROR] {e}")
				continue

			if not data:
				print("[CRUNCH DISCONNECTED] peer closed connection")
				continue

			print(f"[CRUNCH RECEIVED BYTES] {len(data)}")
			expected_size = CRUNCH_REAL_START + (CRUNCH_REAL_COUNT * CRUNCH_REAL_STRIDE)
			if len(data) < expected_size:
				print(f"[CRUNCH WARN] Expected {expected_size} bytes, got {len(data)}")

			values = read_real_array(
				data,
				CRUNCH_REAL_START,
				CRUNCH_REAL_COUNT,
				CRUNCH_REAL_STRIDE,
			)

			with state_lock:
				operator_id = latest_state["operator_id"] or ""
				item_ids = dict(latest_state["item_ids"])

			now = datetime.now()
			_, saved_path = save_crunch_payload(operator_id, item_ids, now, values)
			print(f"[CRUNCH DATA] saved {saved_path}")


threading.Thread(target=recv_server_thread, daemon=True).start()
threading.Thread(target=send_server_thread, daemon=True).start()
threading.Thread(target=crunch_recv_server_thread, daemon=True).start()

def load_camera_area_config():
	"""Load camera index and detection area from config.json + extra options."""
	defaults_index = 0
	defaults_area = ((250, 190), (390, 290))
	defaults_dispenser_area = ((190, 290), (290, 490))
	defaults_edge = "left"

	base_path = get_base_path()
	cfg_path = os.path.join(base_path, "config.json")
	cam_idx = defaults_index
	area = defaults_area
	area_dispenser = defaults_dispenser_area
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

			if isinstance(raw_area, list) and len(raw_area) == 2:
				(ax1, ay1), (ax2, ay2) = raw_area
				area = ((int(ax1), int(ay1)), (int(ax2), int(ay2)))
			elif isinstance(raw_area, dict):
				ax1 = int(raw_area.get("x1", defaults_area[0][0]))
				ay1 = int(raw_area.get("y1", defaults_area[0][1]))
				ax2 = int(raw_area.get("x2", defaults_area[1][0]))
				ay2 = int(raw_area.get("y2", defaults_area[1][1]))
				area = ((ax1, ay1), (ax2, ay2))

			raw_area_dispenser = data.get("area_dispenser")
			if isinstance(raw_area_dispenser, list) and len(raw_area_dispenser) == 2:
				(dx1, dy1), (dx2, dy2) = raw_area_dispenser
				area_dispenser = ((int(dx1), int(dy1)), (int(dx2), int(dy2)))
			elif isinstance(raw_area_dispenser, dict):
				dx1 = int(raw_area_dispenser.get("x1", defaults_dispenser_area[0][0]))
				dy1 = int(raw_area_dispenser.get("y1", defaults_dispenser_area[0][1]))
				dx2 = int(raw_area_dispenser.get("x2", defaults_dispenser_area[1][0]))
				dy2 = int(raw_area_dispenser.get("y2", defaults_dispenser_area[1][1]))
				area_dispenser = ((dx1, dy1), (dx2, dy2))

	except Exception as e:
		print(f"[CONFIG:CAM/AREA] Using defaults. Reason: {e}")

	(ax1, ay1), (ax2, ay2) = area
	x1n, x2n = (ax1, ax2) if ax1 <= ax2 else (ax2, ax1)
	y1n, y2n = (ay1, ay2) if ay1 <= ay2 else (ay2, ay1)
	(dx1, dy1), (dx2, dy2) = area_dispenser
	dx1n, dx2n = (dx1, dx2) if dx1 <= dx2 else (dx2, dx1)
	dy1n, dy2n = (dy1, dy2) if dy1 <= dy2 else (dy2, dy1)

	return cam_idx, ((x1n, y1n), (x2n, y2n)), ((dx1n, dy1n), (dx2n, dy2n)), is_rotate, is_rotate90, edge


camera_index, area, area_dispenser, is_rotate, is_rotate90, edge = load_camera_area_config()
(x1, y1), (x2, y2) = area
(dx1, dy1), (dx2, dy2) = area_dispenser

cam = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
is_frame_ok = False
last_handled_key = None

while not cam.isOpened() and not is_frame_ok:
	cam = cv2.VideoCapture(camera_index)
	is_frame_ok, _ = cam.read()
	print("Waiting for camera...")
	sleep(0.05)

while True:
	process_pending_id_requests()

	is_frame_ok, frame = cam.read()
	if not is_frame_ok or frame is None:
		continue

	if is_rotate:
		frame = cv2.rotate(frame, cv2.ROTATE_180)

	if is_rotate90:
		frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

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
	cv2.imshow("Filtered", msk)
	cv2.imshow("Filtered Dispenser", dispenser_msk)

	key = cv2.waitKey(1)
	if key == -1 or key == 255:
		last_handled_key = None
		continue

	if key == last_handled_key:
		continue

	last_handled_key = key

	if key == q_unicode:
		break

	if key == b_unicode and pt_center is not None:
		saved_reference_center = pt_center

	if key == n_unicode and dispenser_center is not None:
		saved_dispenser_reference_center = dispenser_center

	if key == m_unicode and current_relative_diff is not None:
		saved_relative_diff = current_relative_diff

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

	key_cmd = get_key_command(key)
	if key_cmd:
		response, log_text, delta_saved_path, auth_saved_path = handle_command(key_cmd, source="keyboard")
		if log_text:
			print(f"[KEY CMD] {key_cmd}: {log_text}")
		if delta_saved_path:
			print(f"[KEY DATA] saved {delta_saved_path}")
		if auth_saved_path:
			print(f"[KEY AUTH DATA] saved {auth_saved_path}")

cam.release()
cv2.destroyAllWindows()
