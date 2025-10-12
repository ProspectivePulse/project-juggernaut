import argparse, json, socket, time
import numpy as np
from stable_baselines3 import DQN
import sys

def send_recv(sock, payload, timeout=5.0):
    sock.settimeout(timeout)
    sock.sendall(json.dumps(payload).encode())
    data = sock.recv(4096) # small, single-response messages
    if not data:
        raise RuntimeError("Empty response from Pi server")
    return json.loads(data.decode())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pi_ip", required=True, help="192.168.1.9")
    ap.add_argument("--pi_port", type=int, default=5000)
    ap.add_argument("--model", default="./dqn_checkpoints/dqn_robot_16500_steps.zip", help="C:/Users/reply/PycharmProjects/RL_Robot_Demo/")
    ap.add_argument("--deterministic", action="store_true", help="Deterministic policy")
    ap.add_argument("--step_delay", type=float, default=0.0, help="Delay between steps (s)")
    ap.add_argument("--max_steps", type=int, default=30, help="0 = run until Ctrl+C")
    args = ap.parse_args()

    print(f"Loading model: {args.model}")
    model = DQN.load(args.model, device="cpu")

    print(f"Connecting to Pi at {args.pi_ip}:{args.pi_port} ...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((args.pi_ip, args.pi_port))
    print("Connected.")

    steps = 0
    try:
        while True:
            try:

                resp = send_recv(sock, {"action":-1}, timeout=6.0)
                obs = np.array([resp.get("obs", 0.5)], dtype=np.float32)

                action, _ = model.predict(obs, deterministic=args.deterministic)
                action = int(action)

                resp2 = send_recv(sock, {"action": action}, timeout=6.0)
                obs2 = float(resp2.get("obs", 0.5))
                reward = float(resp2.get("reward", 0.0))
                raw_cm = resp2.get("raw_cm", None)

                steps += 1
                if steps == 1 or steps % 5.0 == 0.0:
                    print(f"step={steps} raw_cm={raw_cm} obs={obs2:.3f} action={action} reward={reward:.3f}")

                if args.max_steps > 0 and steps >= args.max_steps:
                    print("Reached max steps, stopping inference")
                    break
                time.sleep(args.step_delay)

            except (socket.timeout, RuntimeError, json.JSONDecodeError) as e:
                print(f" Comms issue: {e} - retrying ping...")
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
    main()
