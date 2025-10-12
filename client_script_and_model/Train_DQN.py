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
print(PIL.__version__)

PI_HOST = "192.168.1.9"
PI_PORT = 5000
CKPT_DIR = "./dqn_checkpoints"
MODEL_PATH = "dqn_robot.zip"


class RobotEnv(gym.Env):
    metadata = {}

    def __init__(self, host=PI_HOST, port=PI_PORT):
        super().__init__()
        self.host, self.port = host, port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(10.0)
        self.sock.connect((host, port))

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


# --- TensorBoard callback (logs reward, raw distance, action histogram --- #
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
                f.write(f"{self.step_idx},{'' if raw_cm is None else raw_cm}," f"{'' if obs is None else obs}, {'' if action is None else action}, {reward}\n")

        if raw_cm is not None:
            self.logger.record("robot/raw_distance_cm", float(raw_cm))
        if action is not None:
            self.logger.record("robot/action_index", action)
            for i in range(5):
                self.logger.record(f"robot/action_onehot/{i}", 1.0 if i==action else 0.0)

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


def find_latest_checkpoint():
    os.makedirs(CKPT_DIR, exist_ok=True)
    ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, "dqn_robot_*_steps.zip")))
    return ckpts[-1] if ckpts else (MODEL_PATH if os.path.exists(MODEL_PATH) else None)


if __name__ == "__main__":

    base_env = DummyVecEnv([lambda: RobotEnv()])
    env = VecMonitor(base_env)


    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
        buffer_size=5000,
        learning_starts=1000,
        batch_size=256,
        learning_rate=3e-4,
        train_freq=4, # This value strikes a balance between learning from new experience and computational efficiency. Too low, such as 1, is computationally expensive, and might lead to chasing noisy, short-term patterns. Too high, such as 16, leads to less frequent policy updates, potentially leading to slowing down the learning.
        target_update_interval=1000, # This parameter copies the weight from the main Q-network to the stable target network every 1000 steps. The purpose of the target network is to provide a stable, slowly-changing target for the main network to learn towards. If network is updated too frequently, it becomes almost identical to the main network. This can lead
                                     # to unstable and divergent training.
        exploration_fraction=0.1,
        exploration_final_eps=0.05,
        gamma=0.99,
        tensorboard_log="./dqn_tb_runs",
    )

    latest = find_latest_checkpoint()
    if latest:
        print(f"Resuming from: {latest}")
        model = DQN.load(latest, env=env, device="cuda")
        reset_num_timesteps = False
    else:
        print("Starting fresh")
        reset_num_timesteps = True

    ckpt_cb = CheckpointCallback(save_freq=500, save_path=CKPT_DIR, name_prefix="dqn_robot")

    eval_env = DummyVecEnv([lambda: Monitor(RobotEnv())])
    eval_cb = EvalCallback(eval_env, best_model_save_path="./dqn_best", log_path="./dqn_tb_runs/eval", eval_freq=2000, deterministic=True)

    model.learn(total_timesteps=5000, tb_log_name="dqn_robot", callback=[TBCallback(window=300, csv_path="dqn_rollout_log.csv"), ckpt_cb], reset_num_timesteps=reset_num_timesteps,)
    model.save(MODEL_PATH)
    env.close()
    eval_env.close()






