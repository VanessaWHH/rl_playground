import os
import sys
import random
import importlib.util

from sb3_contrib import QRDQN, MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from envs.snake_env import SnakeEnv


ALGOS = {
    "a2c": A2C,
    "dqn": DQN,
    "ppo": PPO,
    "ppo2": PPO,
    # SB3 Contrib,
    "qrdqn": QRDQN,
    "maskableppo": MaskablePPO,
    "maskableppo2": MaskablePPO,
}


def load_config(config_path: str, exp_name: str):
    spec = importlib.util.spec_from_file_location("configs", config_path)
    configs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(configs)
    return getattr(configs, exp_name)


def gen_random_seeds(num: int):
    """Generate a list of random seeds for each environment"""
    seed_set = set()
    while len(seed_set) < num:
        seed_set.add(random.randint(0, 1e9))
    return seed_set


def make_env(seed: int, level_map: str):
    def _init():
        env = SnakeEnv(seed=seed, level_map=level_map)
        env = ActionMasker(env, SnakeEnv.get_action_mask)
        env = Monitor(env)
        random.seed(seed)
        return env

    return _init


def make_checkpoint_callback(model, save_dir):
    checkpoint_interval = 15625
    return CheckpointCallback(
        save_freq=checkpoint_interval, save_path=save_dir, name_prefix=model
    )


def class_to_dict(src):
    params = {}
    black_keys = ["__module__", "__dict__", "__weakref__", "__doc__"]
    for k, v in src.__dict__.items():
        if k not in black_keys:
            params[k] = v
    return params


def filter_dict(src):
    black_keys = ["__module__", "__dict__", "__weakref__", "__doc__"]
    params = {k: v for k, v in src.items() if k not in black_keys}
    return params


def gen_ts():
    import time

    return time.strftime("%y%m%d-%H%M%S", time.localtime())


