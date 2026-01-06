import argparse, json, socket, time
import numpy as np
from stable_baselines3 import DQN
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

from src.config import Config


def run_inference(ip=None, port=None, model_path=None, deterministic=False, step_delay=0.0, max_steps=30):
    # 1. Configuration Setup
    target_ip = ip if ip else Config.ROBOT_IP
    target_model_path = model_path if model_path else Config.MODEL_PATH
    target_port = port if port else Config.ROBOT_PORT

    print(f"Connecting to Robot at {target_ip}:{target_port}")
    print(f"Loading Model: {target_model_path}")

    # 2. Load the Model
    if not os.path.exists(target_model_path):
        print(f"Error: Model file not found at {target_model_path}")
        return

    model = DQN.load(target_model_path, device="cpu")

    def send_recv(sock, payload, timeout=5.0):
        sock.settimeout(timeout)
        sock.sendall(json.dumps(payload).encode())
        data = sock.recv(4096) # small, single-response messages
        if not data:
            raise RuntimeError("Empty response from Pi server")
        return json.loads(data.decode())

    def main():

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((target_ip, target_port))
        print("Connected.")

        steps = 0
        try:
            while True:
                try:

                    resp = send_recv(sock, {"action": -1}, timeout=6.0)
                    obs = np.array([resp.get("obs", 0.5)], dtype=np.float32)

                    action, _ = model.predict(obs, deterministic=deterministic)
                    action = int(action)

                    resp2 = send_recv(sock, {"action": action}, timeout=6.0)
                    obs2 = float(resp2.get("obs", 0.5))
                    reward = float(resp2.get("reward", 0.0))
                    raw_cm = resp2.get("raw_cm", None)

                    steps += 1
                    if steps == 1 or steps % 5.0 == 0.0:
                        print(f"step={steps} raw_cm={raw_cm} obs={obs2:.3f} action={action} reward={reward:.3f}")

                    if (max_steps > 0) and (steps >= max_steps):
                        print("Reached max steps, stopping inference")
                        break
                    time.sleep(step_delay)

                except (socket.timeout, RuntimeError, json.JSONDecodeError) as e:
                    print(f"Comms issue: {e} - retrying ping...")
                    try:
                        _ = send_recv(sock, {"action": -1}, timeout=6.0)
                        continue
                    except Exception as e2:
                        print(f"Ping failed: {e2}")
                        break

        except KeyboardInterrupt:
            print("\nStopping (Ctrl+C).")
        finally:
            try: sock.close()
            except: pass
            print("Client closed.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pi_ip", help="Override Robot IP")
    ap.add_argument("--pi_port", type=int, default=target_port)
    ap.add_argument("--model", help="Override Model Path")
    ap.add_argument("--deterministic", action="store_true", help="Deterministic policy")
    ap.add_argument("--step_delay", type=float, default=0.0, help="Delay between steps (s)")
    ap.add_argument("--max_steps", type=int, default=30, help="0 = run until Ctrl+C")
    args = ap.parse_args()

    run_inference(ip=args.pi_ip, port=args.target_port, model_path=args.model, deterministic=args.deterministic,
                  step_delay=0.0, max_steps=30)
