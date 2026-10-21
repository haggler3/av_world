"""A controller for av_world.sil, and the smallest client library: connect, receive a step, send a command.

    python -m av_world.sil_client --port 5555                 # the example controller: follow the leader, stop for red, keep the limit
    python -m av_world.sil_client --port 5555 --delay-ms 80   # the same, slow (to see deadline misses in realtime mode)

Your controller: replace `example` with anything that maps an ego dict (see av_world/sil.py, "Protocol") to [u0, u1]. On a hardware
target, run the same loop on the device, or a thin gateway that forwards the step to the ECU (CAN, serial, ROS) and the answer back.
"""
import argparse
import json
import socket
import time


def connect(host="127.0.0.1", port=5555, retries=100):
    for _ in range(retries):
        try:
            s = socket.create_connection((host, port))
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            return s
        except OSError:
            time.sleep(0.1)
    raise ConnectionError(f"no simulator on {host}:{port}")


def serve(sock, policy, delay_s=0.0):
    """Answer every step with policy(ego) -> [u0, u1]; return the simulator's summary."""
    f = sock.makefile("rb")
    try:
        return _loop(sock, f, policy, delay_s)
    except (BrokenPipeError, ConnectionError):
        return None  # the simulator closed the connection


def _loop(sock, f, policy, delay_s):
    for line in f:
        msg = json.loads(line)
        if msg["type"] == "step":
            if delay_s:
                time.sleep(delay_s)
            u = {str(e["id"]): policy(e) for e in msg["egos"]}
            sock.sendall((json.dumps({"type": "cmd", "step": msg["step"], "u": u}) + "\n").encode())
        elif msg["type"] == "bye":
            return msg["summary"]
    return None


def accel_to_u0(a):
    return max(-1.0, min(1.0, a / 3.0 if a >= 0 else a / 8.0))


def example(e):
    """A plain driver: IDM toward the speed limit, behind the leader, and toward the stop target (red light, stop line, queue)."""
    v, v0 = e["speed_mps"], max(e["speed_limit_mps"], 1.0)
    a = 1.5 * (1 - (v / v0) ** 4)
    for gap, vl in ((e["lead_gap_m"], e["lead_speed_mps"]), (e["stop_gap_m"], 0.0)):
        if gap is not None:
            s_star = 2.0 + max(0.0, v * 1.4 + v * (v - (vl or 0.0)) / (2 * (1.5 * 2.0) ** 0.5))
            a = min(a, 1.5 * (1 - (v / v0) ** 4 - (s_star / max(gap, 0.1)) ** 2))
    return [accel_to_u0(max(a, -8.0)), 0.0]


def brake(e):
    return [-1.0, 0.0]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--delay-ms", type=float, default=0.0)
    a = ap.parse_args()
    print(json.dumps(serve(connect(a.host, a.port), example, a.delay_ms / 1000), indent=1))
