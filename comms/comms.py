import json
import os
import socket
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
DATA_DIR = os.path.join(BASE_DIR, "data")


def load_config():
    defaults = {
        "ip": "192.168.10.5",
        "port": 2000,
        "chunk": 2048,
        "timeout": 3.0,
        "start_offset": 0,
        "array_count": 10,
        "declared_n": None,
    }

    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            data = {}
    except Exception as e:
        print(f"[CONFIG] Using defaults. Reason: {e}")
        data = {}

    ip = str(data.get("ip", defaults["ip"]))
    port = int(data.get("port", defaults["port"]))
    chunk = int(data.get("chunk", defaults["chunk"]))
    timeout = float(data.get("timeout", defaults["timeout"]))
    start_offset = int(data.get("start_offset", defaults["start_offset"]))
    array_count = int(data.get("array_count", defaults["array_count"]))
    declared_n = data.get("declared_n", defaults["declared_n"])
    if declared_n is not None:
        declared_n = int(declared_n)

    return ip, port, chunk, timeout, start_offset, array_count, declared_n


IP, PORT, CHUNK, TIMEOUT, START_OFFSET, ARRAY_COUNT, DECLARED_N = load_config()
ADDR = (IP, PORT)


def save_received_payload(data: bytes, now: datetime):
    os.makedirs(DATA_DIR, exist_ok=True)
    filename = now.strftime("%Y%m%d_%H%M%S_%f.txt")
    file_path = os.path.join(DATA_DIR, filename)
    with open(file_path, "wb") as f:
        f.write(data)
    return file_path

def decode_s7_string_fixed(data: bytes, offset: int, declared_max_len: int):
    """Decode one Siemens STRING[n] at offset with fixed declared length n.
    Returns (text, cur_len, stride_bytes) where stride_bytes == declared_max_len + 2.
    """
    stride = declared_max_len + 2
    end = offset + stride
    if end > len(data):
        return None, 0, stride

    max_len = data[offset]
    cur_len = data[offset + 1]

    # Validación básica
    if max_len != declared_max_len or cur_len > declared_max_len:
        return None, 0, stride

    raw = data[offset + 2 : offset + 2 + cur_len]
    try:
        text = raw.decode("ascii", errors="strict")
    except UnicodeDecodeError:
        return None, 0, stride

    return text, cur_len, stride

def decode_s7_string_array_exact(data: bytes, start_offset: int, count: int, declared_max_len: int):
    """Decode exactly `count` elements of STRING[n] array starting at start_offset."""
    results = []
    stride = declared_max_len + 2
    for idx in range(count):
        off = start_offset + idx * stride
        text, cur_len, _ = decode_s7_string_fixed(data, off, declared_max_len)
        if text is None:
            results.append((idx, off, None, 0))  # inválido o incompleto
        else:
            results.append((idx, off, text, cur_len))
    return results

def main():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(ADDR)
    server.listen()
    print(f"[LISTENING] Server is listening on {IP}:{PORT}")

    while True:
        conn, addr = server.accept()
        conn.settimeout(TIMEOUT)
        print(f"[NEW CONNECTION] {addr} connected.")

        buf = bytearray()
        try:
            while True:
                chunk = conn.recv(CHUNK)
                if not chunk:
                    break
                buf.extend(chunk)

                # Si ya sabemos el tamaño total esperado, podemos cortar
                if DECLARED_N is not None:
                    needed = START_OFFSET + (DECLARED_N + 2) * ARRAY_COUNT
                    if len(buf) >= needed:
                        break
        except socket.timeout:
            # sin datos nuevos por TIMEOUT, seguimos con lo que tengamos
            pass

        data = bytes(buf)
        print(f"[RECEIVED BYTES] {len(data)} bytes")
        if data:
            now = datetime.now()
            saved_path = save_received_payload(data, now)
            print(f"[DATA] saved {saved_path}")
            print(f"[HEX PREVIEW] {data[:64].hex()}...")

            # Inferir n si no se fijó
            declared_n = DECLARED_N
            if declared_n is None:
                if START_OFFSET + 1 < len(data):
                    declared_n = data[START_OFFSET]
                else:
                    declared_n = 0

            if declared_n <= 0:
                print("[ERROR] No se pudo inferir n (longitud máxima).")
            else:
                stride = declared_n + 2
                total_needed = START_OFFSET + stride * ARRAY_COUNT
                print(f"[INFO] n={declared_n}, stride={stride} bytes, array={ARRAY_COUNT}, total_needed={total_needed}")

                if len(data) < total_needed:
                    print(f"[WARN] Buffer incompleto: tengo {len(data)} bytes, se esperan {total_needed}.")

                arr = decode_s7_string_array_exact(
                    data, START_OFFSET, ARRAY_COUNT, declared_n
                )

                print("[S7 STRING ARRAY]")
                for idx, off, text, cur_len in arr:
                    if text is None:
                        # Puede ser elemento vacío válido (cur_len=0) o inválido/incompleto.
                        # Intento distinguirlo leyendo cur_len si hay bytes suficientes:
                        if off + 2 <= len(data):
                            cl = data[off + 1]
                            mx = data[off]
                            if off + stride <= len(data) and mx == declared_n and cl == 0:
                                print(f"  [{idx}] @offset {off}: '' (vacío)")
                            else:
                                print(f"  [{idx}] @offset {off}: <incompleto o inválido>")
                        else:
                            print(f"  [{idx}] @offset {off}: <sin datos suficientes>")
                    else:
                        print(f"  [{idx}] @offset {off}: '{text}' (len={cur_len})")

        conn.close()
        print(f"[DISCONNECTED] {addr} closed the connection.")

if __name__ == "__main__":
    main()
