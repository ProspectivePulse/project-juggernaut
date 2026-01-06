import os, glob
import json, socket, numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
import torch
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from collections import deque
import io, matplotlib.pyplot as plt
import PIL
import sys
import argparse


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)

from src.config import Config

class RobotEnv(gym.Env):
    metadata = {}

    def __init__(self, host=None, port=None):
        super().__init__()

        self.host = host if host else Config.ROBOT_IP
        self.port = port if port else Config.ROBOT_PORT

        print(f"Environment connecting to {self.host}:{self.port}")

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(10.0)
        self.sock.connect((self.host, self.port))

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.last_raw_cm = None
        self.last_obs = 0.5

    def _send(self, payload: dict) -> dict:
        self.sock.sendall(json.dumps(payload).encode())
        data = self.sock.recv(2048).decode()
        return json.loads(data)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        resp = self._send({"action": -1})
        self.last_raw_cm = resp.get("raw_cm", None)
        obs_val = float(resp.get("obs", 0.5))
        self.last_obs = obs_val
        return np.array([obs_val], dtype=np.float32), {}

    def step(self, action):
        resp = self._send({"action": int(action)})
        self.last_raw_cm = resp.get("raw_cm", None)
        obs_val = float(resp.get("obs", 0.5))
        self.last_obs = obs_val
        reward = float(resp.get("reward", 0.0))
        done = bool(resp.get("done", False))
        return np.array([obs_val], dtype=np.float32), reward, done, False, {}

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass
        super().close()


# --- TensorBoard callback (logs reward, raw distance, action histogram) --- #
class TBCallback(BaseCallback):
    def __init__(self, window=300, csv_path=None):
        super().__init__()
        self.window = window
        self.dist_buf = deque(maxlen=window)
        self.act_buf = deque(maxlen=window)
        self.rew_buf = deque(maxlen=window)
        self.csv_path = csv_path
        self.step_idx = 0
        if csv_path:
            with open(csv_path, "w", buffering=1, newline="") as f:
                f.write("step,raw_cm,obs,action,reward\n")

    def _fig_to_image(self, fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        import PIL.Image as Image
        arr = np.array(Image.open(buf)).transpose(2, 0, 1)
        return arr

    def _on_step(self) -> bool:
        # Get raw_cm from the underlying environment (Vectorized)
        env = self.training_env.envs[0]

        raw_cm = getattr(env, "last_raw_cm", None)
        obs = getattr(env, "last_obs", None)

        action = int(self.locals["actions"][0]) if "actions" in self.locals else None
        reward = float(self.locals.get("rewards", [0.0])[0])

        if self.csv_path:
            with open(self.csv_path, "a") as f:
                f.write(f"{self.step_idx},{'' if raw_cm is None else raw_cm}," f"{'' if obs is None else obs}, "
                        f"{'' if action is None else action}, {reward}\n")

        if raw_cm is not None:
            self.logger.record("robot/raw_distance_cm", float(raw_cm))

        if action is not None:
            self.logger.record("robot/action_index", action)
            for i in range(5):
                self.logger.record(f"robot/action_onehot/{i}", 1.0 if i == action else 0.0)

        if raw_cm is not None and action is not None:
            self.dist_buf.append(float(raw_cm))
            self.act_buf.append(action)
            self.rew_buf.append(reward)

            # Rolling time-series: distance + actions
            x = np.arange(len(self.dist_buf))
            fig = plt.figure(figsize=(7, 3))
            ax1 = plt.gca()
            ax1.plot(x, self.dist_buf, label="distance (cm)")
            ax1.set_ylim(0, 120)
            ax1.set_xlabel("recent steps"); ax1.set_ylabel("cm")

            ax2 = ax1.twinx()
            ax2.step(x, self.act_buf, where="post", alpha=0.5, label="action idx")
            ax2.set_ylabel("action (0..4)")
            fig.legend(loc="upper right")

            self.logger.record("robot/ts_dist_action", self._fig_to_image(fig))

            # Scatter: distance vs action
            fig2 = plt.figure(figsize=(5, 3))
            plt.scatter(self.dist_buf, self.act_buf, s=8)
            plt.xlabel("distance (cm)"); plt.ylabel("action index")

            self.logger.record("robot/scatter_dist_vs_action", self._fig_to_image(fig2))

        self.step_idx += 1
        return True


def find_latest_checkpoint(ckpt_dir, backup_model_path):
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "dqn_robot_*_steps.zip")))
    # Return latest checkpoint, or fallback to main model path if it exists
    if ckpts:
        return ckpts[-1]
    if os.path.exists(backup_model_path):
        return backup_model_path
    return None


# Main Training Function
def train_agent(ip=None, port=None, total_timesteps=10000, save_path=None, load_path=None):
    """
    Main Training Loop.
    :param ip:  Robot IP
    :param port: Robot Port
    :param total_timesteps: How long to train
    :param save_path: Where to save the final model
    :param load_path: Continue training from this model
    :return:
    """

    # 1. Resolve Configuration
    target_ip = ip if ip else Config.ROBOT_IP
    target_port = port if port else Config.ROBOT_PORT

    # Paths
    final_save_path = save_path if save_path else Config.MODEL_PATH
    checkpoint_dir = "./dqn_checkpoints"

    print(f"Training Configuration:")
    print(f"Target:    {target_ip}:{target_port}")
    print(f"Steps:     {total_timesteps}")
    print(f"Save Path: {final_save_path}")

    # 2. Initialize Environment
    # Pass the resolved IP/Port to the Env
    base_env = DummyVecEnv([lambda: RobotEnv(host=target_ip, port=target_port)])
    env = VecMonitor(base_env)

    # 3. Initialize Model
    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
        buffer_size=5000,
        learning_starts=1000,
        batch_size=256,
        learning_rate=3e-4,
        train_freq=4,
        # This value strikes a balance between learning from new experience and computational efficiency. Too low, such as 1, is computationally expensive, and might lead to chasing noisy, short-term patterns. Too high, such as 16, leads to less frequent policy updates, potentially leading to slowing down the learning.
        target_update_interval=1000,
        # This parameter copies the weight from the main Q-network to the stable target network every 1000 steps. The purpose of the target network is to provide a stable, slowly-changing target for the main network to learn towards. If network is updated too frequently, it becomes almost identical to the main network. This can lead
        # to unstable and divergent training.
        exploration_fraction=0.1,
        exploration_final_eps=0.05,
        gamma=0.99,
        tensorboard_log="./dqn_tb_runs",
    )

    # 4. Check for Existing Models
    # If explicit load_path provided, us it. Otherwise, search checkpoints.
    if load_path:
        start_model = load_path
    else:
        start_model = find_latest_checkpoint(checkpoint_dir, final_save_path)


    reset_timesteps = True
    if start_model:
        print(f"Resuming from: {start_model}")
        # Re-load into the model object
        model = DQN.load(start_model, env=env, device="cuda" if torch.cuda.is_available() else "cpu")
        reset_timesteps = False
    else:
        print("Starting fresh training session.")

    # 5. Callbacks
    ckpt_cb = CheckpointCallback(save_freq=500, save_path=checkpoint_dir, name_prefix="dqn_robot")

    eval_env = DummyVecEnv([lambda: Monitor(RobotEnv(host=target_ip, port=target_port))])
    eval_cb = EvalCallback(eval_env, best_model_save_path="./dqn_best", log_path="./dqn_tb_runs/eval", eval_freq=2000,
                           deterministic=True)

    # 6. Train
    print("Starting Learn Loop...")
    model.learn(total_timesteps=total_timesteps, tb_log_name="dqn_robot",
                callback=[TBCallback(window=300, csv_path="dqn_rollout_log.csv"), ckpt_cb],
                reset_num_timesteps=reset_num_timesteps, )

    model.save(final_save_path)
    print(f"Model saved to {final_save_path}")

    env.close()
    eval_env.close()


# Client Entry Point
if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    # Network Args
    ap.add_argument("--pi_ip", help="Override Robot IP")
    ap.add_argument("pi_port", type=int, help="Override Robot Port")

    # Training Args
    ap.add_argument("--timesteps", type=int, default=10000, help="Total training steps")
    ap.add_argument("--save_path", help="Path to save trained model")
    ap.add_argument("--load_model", help="Path to existing model to resume training")

    args = ap.parse_args()

    train_agent(
        ip=args.pi_ip,
        port=args.pi_port,
        total_timesteps=args.timesteps,
        save_path=args.save_path,
        load_path=args.load_model
    )









