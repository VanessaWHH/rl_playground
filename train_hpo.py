import importlib.util
import os
import random
import sys

import fire
import optuna

from torch import nn

from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import safe_mean

from rl_zoo3.utils import linear_schedule

from common import ALGOS, make_env, make_checkpoint_callback, class_to_dict, filter_dict


"""
python train_hpo.py configs_hpo/configs_hpo.py ppo
"""


def gen_ts():
    import time

    return time.strftime("%y%m%d-%H%M%S", time.localtime())


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


def objective(trial, config_path, exp_name): 
    config = load_config(config_path, exp_name)
    save_dir = config.save_dir_prefix + str(trial._trial_id - 1)
    os.makedirs(save_dir, exist_ok=True)

    seed_set = gen_random_seeds(config.n_envs)
    env = SubprocVecEnv(
        [make_env(seed=s, level_map=config.level_map) for s in seed_set]
    )

    params = class_to_dict(config.model_params)

    if hasattr(config, 'search_space'):
        for param, bounds in config.search_space.__dict__.items():
            value = None
            if isinstance(bounds, tuple) and len(bounds) == 2:
                value = trial.suggest_float(param, bounds[0], bounds[1], log=True)
            elif isinstance(bounds, list):
                value = trial.suggest_categorical(param, bounds)

            # If the hyperparameter needs linear scheduling, apply it
            if param == "learning_rate" or param == "clip_range":
                params[param] = linear_schedule(value)
            elif param == "activation_fn_name":
                activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}[value]
                params["policy_kwargs"] = dict(activation_fn = activation_fn)
            else:
                params[param] = value

    params = filter_dict(params)    

    model = ALGOS[config.algo](
        "CnnPolicy",
        env,
        device="cuda",
        verbose=1,
        tensorboard_log=save_dir,
        **params,
    )

    # checkpoint_interval * num_envs = total_steps_per_checkpoint
    ckpt_callback = make_checkpoint_callback(config.algo, save_dir)

    # start training and save stdout to log file
    original_stdout = sys.stdout
    log_file_path = os.path.join(save_dir, "training_log.txt")
    with open(log_file_path, "w") as log_file:
        sys.stdout = log_file

        model.learn(
            total_timesteps=int(config.n_timesteps), 
            #callback=[ckpt_callback]
        )
        env.close()

    # Restore stdout
    sys.stdout = original_stdout

    return safe_mean([info["r"] for info in model.ep_info_buffer])


def train_hpo(config_path, exp_name):
    study = optuna.create_study(
        storage=f"sqlite:///optuna_{exp_name}_{gen_ts()}.sqlite3",
        study_name=f"optuna_{exp_name}_{gen_ts()}",
        direction="maximize")
    study.optimize(lambda trial: objective(trial, config_path, exp_name), n_trials=100)


if __name__ == "__main__":
    fire.Fire(train_hpo)

