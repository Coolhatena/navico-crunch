import socket
import struct
import datetime
import os
import json
import sys

def get_base_path():
    if getattr(sys, "frozen", False):  # ejecutándose como .exe
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def load_config():
    """Load socket and REAL-array settings from config.json next to script/.exe."""
    defaults = {
        "ip": "192.168.10.34",
        "port": 2000,
        "size": 2048,
        "real_count": 360,
        "real_stride": 4,
        "real_start": 0,
        "output_dir": "datos",
    }

    base_path = get_base_path()
    cfg_path = os.path.join(base_path, "config.json")
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict) and "crunch_coms" in data and isinstance(data["crunch_coms"], dict):
            data = data["crunch_coms"]

        ip = str(data.get("ip", defaults["ip"]))
        port = int(data.get("port", defaults["port"]))
        size = int(data.get("size", defaults["size"]))
        real_count = int(data.get("real_count", defaults["real_count"]))
        real_stride = int(data.get("real_stride", defaults["real_stride"]))
        real_start = int(data.get("real_start", defaults["real_start"]))
        output_dir = str(data.get("output_dir", defaults["output_dir"]))
        return ip, port, size, real_count, real_stride, real_start, output_dir
    except Exception as e:
        print(f"[CONFIG] Using defaults. Reason: {e}")
        return (
            defaults["ip"],
            defaults["port"],
            defaults["size"],
            defaults["real_count"],
            defaults["real_stride"],
            defaults["real_start"],
            defaults["output_dir"],
        )


IP, PORT, SIZE, REAL_COUNT, REAL_STRIDE, REAL_START, OUTPUT_DIR = load_config()
ADDR = (IP, PORT)

def read_real_at(data: bytes, offset: int) -> float | None:
    end = offset + 4
    if end > len(data):
        return None
    return struct.unpack('>f', data[offset:end])[0]

def read_real_array(data: bytes, start_offset: int, count: int, stride: int = 4):
    vals = []
    for i in range(count):
        off = start_offset + i * stride
        vals.append(read_real_at(data, off))
    return vals

def build_base_filename(ts: datetime.datetime) -> str:
    # crunch_F_YMDYYYYMMDDH_HMSHHMMSS.txt
    return f"crunch_F_YMD{ts.strftime('%Y%m%d')}H_HMS{ts.strftime('%H%M%S')}.txt"

def next_available_filename(base_name: str) -> str:
    """Si el archivo existe, agrega _2, _3, ... usando solo '_'."""
    if not os.path.exists(base_name):
        return base_name
    root, ext = os.path.splitext(base_name)
    n = 2
    while True:
        candidate = f"{root}_{n}{ext}"
        if not os.path.exists(candidate):
            return candidate
        n += 1

def save_block(values):
    ts = datetime.datetime.now()
    base = build_base_filename(ts)
    output_path = os.path.join(get_base_path(), OUTPUT_DIR)
    os.makedirs(output_path, exist_ok=True)
    filename = next_available_filename(os.path.join(output_path, base))
    with open(filename, "w", encoding="utf-8") as f:
        for v in values:
            f.write(f"{v},\n")
    print(f"[SAVED] {len(values)} valores en {filename}")

def main():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(ADDR)
    server.listen()
    print(f"[LISTENING] Server is listening on {IP}:{PORT}")

    while True:
        conn, addr = server.accept()
        print(f"[NEW CONNECTION] {addr} connected.")

        data = conn.recv(SIZE)
        if data:
            print(f"[RECEIVED BYTES] {len(data)}")
            if len(data) < REAL_COUNT * REAL_STRIDE:
                print(f"[WARN] Expected {REAL_COUNT*REAL_STRIDE} bytes, got {len(data)}")

            # Decodificar la cantidad configurada de REALs desde el offset configurado
            arr = read_real_array(data[:REAL_COUNT*REAL_STRIDE], REAL_START, REAL_COUNT, REAL_STRIDE)

            # Guardar cada bloque en archivo con nombre único por recepción
            save_block(arr)

        conn.close()
        print(f"[DISCONNECTED] {addr} closed the connection.")

if __name__ == "__main__":
    main()
