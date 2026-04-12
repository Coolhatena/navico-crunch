import argparse
import json
import socket
import sys
from pathlib import Path


def load_tcp_config(config_path: Path) -> dict:
    try:
        with config_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise ValueError(f"Config file not found: {config_path}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in config file: {config_path} ({exc})")

    if not isinstance(data, dict):
        raise ValueError("Config root must be a JSON object")

    server = data.get("server") if isinstance(data.get("server"), dict) else data
    try:
        cfg = {
            "ip": str(server["ip"]),
            "port": int(server.get("port_recv", server.get("port", 2000))),
        }
    except KeyError as exc:
        raise ValueError(f"Missing required config key: {exc.args[0]}")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid TCP config value type: {exc}")

    if cfg["port"] <= 0:
        raise ValueError("port_recv must be greater than 0")

    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send console commands to pcb.py recv TCP server using correction-cv/config.json."
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("config.json")),
        help="Path to config.json with ip and port_recv",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Socket timeout in seconds",
    )
    return parser.parse_args()


def send_command(ip: str, port: int, command: str, timeout: float) -> bytes:
    payload = command.encode("utf-8")
    with socket.create_connection((ip, port), timeout=timeout) as sock:
        sock.settimeout(timeout)
        sock.sendall(payload)
        return sock.recv(1024)


def main() -> int:
    args = parse_args()

    try:
        cfg = load_tcp_config(Path(args.config))
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    print(f"[TARGET] {cfg['ip']}:{cfg['port']}")
    print("[INFO] Enter a TCP command. Empty input or 'exit' quits.")

    while True:
        try:
            command = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not command or command.lower() == "exit":
            return 0

        try:
            response = send_command(cfg["ip"], cfg["port"], command, args.timeout)
        except OSError as exc:
            print(f"[ERROR] Failed to send command: {exc}", file=sys.stderr)
            continue

        decoded = response.decode("utf-8", errors="replace").strip()
        print(f"[RESPONSE] bytes={response!r} text={decoded!r}")


if __name__ == "__main__":
    raise SystemExit(main())
