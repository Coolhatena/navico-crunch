import argparse
import json
import socket
import struct
import sys
from pathlib import Path


def load_crunch_config(config_path: Path) -> dict:
    try:
        with config_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise ValueError(f"Config file not found: {config_path}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in config file: {config_path} ({exc})")

    if not isinstance(data, dict):
        raise ValueError("Config root must be a JSON object")

    crunch = data.get("crunch_coms")
    if not isinstance(crunch, dict):
        raise ValueError("Missing or invalid 'crunch_coms' object in config")

    try:
        cfg = {
            "ip": str(crunch["ip"]),
            "port": int(crunch["port"]),
            "size": int(crunch["size"]),
            "real_count": int(crunch["real_count"]),
            "real_stride": int(crunch["real_stride"]),
            "real_start": int(crunch["real_start"]),
        }
    except KeyError as exc:
        raise ValueError(f"Missing required crunch_coms key: {exc.args[0]}")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid crunch_coms value type: {exc}")

    if cfg["port"] <= 0:
        raise ValueError("crunch_coms.port must be greater than 0")
    if cfg["size"] <= 0:
        raise ValueError("crunch_coms.size must be greater than 0")
    if cfg["real_count"] < 0:
        raise ValueError("crunch_coms.real_count must be 0 or greater")
    if cfg["real_stride"] < 4:
        raise ValueError("crunch_coms.real_stride must be at least 4 bytes")
    if cfg["real_start"] < 0:
        raise ValueError("crunch_coms.real_start must be 0 or greater")

    return cfg


def generate_values(args: argparse.Namespace, real_count: int) -> list[float]:
    if args.pattern == "constant":
        return [float(args.value)] * real_count

    start = float(args.start)
    step = float(args.step)
    return [start + (index * step) for index in range(real_count)]


def build_payload(values: list[float], real_start: int, real_stride: int) -> bytes:
    total_size = real_start + (len(values) * real_stride)
    payload = bytearray(total_size)

    for index, value in enumerate(values):
        offset = real_start + (index * real_stride)
        payload[offset:offset + 4] = struct.pack(">f", float(value))

    return bytes(payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send a single synthetic TCP payload to crunch_comms using correction-cv/config.json."
    )
    parser.add_argument(
        "--config",
        default="correction-cv/config.json",
        help="Path to config.json with a crunch_coms section",
    )
    parser.add_argument(
        "--pattern",
        choices=("ramp", "constant"),
        default="ramp",
        help="Synthetic value pattern to generate",
    )
    parser.add_argument(
        "--start",
        type=float,
        default=0.0,
        help="Starting value for the ramp pattern",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=1.0,
        help="Increment for the ramp pattern",
    )
    parser.add_argument(
        "--value",
        type=float,
        default=0.0,
        help="Constant value used by the constant pattern",
    )
    parser.add_argument(
        "--preview-count",
        type=int,
        default=10,
        help="How many generated floats to print before sending",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        cfg = load_crunch_config(Path(args.config))
        values = generate_values(args, cfg["real_count"])
        payload = build_payload(values, cfg["real_start"], cfg["real_stride"])
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    preview_count = max(0, min(args.preview_count, len(values)))
    preview_values = values[:preview_count]

    print(f"[TARGET] {cfg['ip']}:{cfg['port']}")
    print(f"[CONFIG] real_count={cfg['real_count']} real_stride={cfg['real_stride']} real_start={cfg['real_start']} size={cfg['size']}")
    print(f"[PAYLOAD] sending {len(values)} floats in {len(payload)} bytes")
    print(f"[PATTERN] {args.pattern}")
    print(f"[PREVIEW] {preview_values}")

    try:
        with socket.create_connection((cfg["ip"], cfg["port"]), timeout=3.0) as sock:
            sock.sendall(payload)
    except OSError as exc:
        print(f"[ERROR] Failed to send payload: {exc}", file=sys.stderr)
        return 1

    print("[DONE] Payload sent successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
