"""TensorBoard + console logger shared by both algorithms."""

import json
import os
import time
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, log_dir: str, algo_name: str, env_id: str, seed: int,
                 config: dict = None):
        run_name = f"{algo_name}_{env_id}_s{seed}_{int(time.time())}"
        self.tb_dir = os.path.join(log_dir, run_name)
        self.writer = SummaryWriter(self.tb_dir)
        self.algo_name = algo_name
        self.env_id = env_id
        self.ep_count = 0
        print(f"Logging to {self.tb_dir}")

        if config:
            config_str = json.dumps(config, indent=2, default=str)
            self.writer.add_text("config", f"```json\n{config_str}\n```", 0)
            config_path = os.path.join(self.tb_dir, "config.json")
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2, default=str)

    def log_scalar(self, tag: str, value: float, step: int):
        self.writer.add_scalar(tag, value, step)

    def log_episode(self, reward: float, length: int, step: int):
        self.ep_count += 1
        self.log_scalar("episode/reward", reward, step)
        self.log_scalar("episode/length", length, step)
        if self.ep_count % 10 == 0:
            print(
                f"  [{self.algo_name}] step {step:>7d} | "
                f"ep {self.ep_count:>5d} | "
                f"R={reward:+7.2f} | len={length}"
            )

    def log_losses(self, policy_loss, value_loss, entropy, step):
        self.log_scalar("loss/policy", policy_loss, step)
        self.log_scalar("loss/value", value_loss, step)
        self.log_scalar("loss/entropy", entropy, step)

    def close(self):
        self.writer.close()
