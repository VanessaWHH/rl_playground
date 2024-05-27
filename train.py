import os
import random
import sys
from torch import nn

import fire
from stable_baselines3.common.vec_env import SubprocVecEnv

from common import ALGOS, load_config, gen_random_seeds, make_env, make_checkpoint_callback, class_to_dict 

"""
python train.py configs/config_blank.py ppo
python train.py configs/config_walls.py ppo
"""


def train(config: str, exp_name: str):
    config = load_config(config, exp_name)
    seed_set = gen_random_seeds(config.n_envs)
    save_dir = config.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # Create the Snake environment.
    env = SubprocVecEnv(
        [make_env(seed=s, level_map=config.level_map) for s in seed_set]
    )

    # create model
    if config.pretrain:
        print("load pretrain: ", config.pretrain)
        model = ALGOS[config.algo].load(
            config.pretrain,
            env,
            device="cuda",
            verbose=1,
            tensorboard_log=save_dir,
            **class_to_dict(config.model_params)
        )
    else:
        model = ALGOS[config.algo](
            "CnnPolicy",
            env,
            device="cuda",
            verbose=1,
            tensorboard_log=save_dir,
            **class_to_dict(config.model_params)
        )

    # checkpoint_interval * num_envs = total_steps_per_checkpoint
    ckpt_callback = make_checkpoint_callback(config.algo, config.save_dir)

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

    # Save the final model
    model.save(os.path.join(save_dir, config.algo + "_final.zip"))


if __name__ == "__main__":
    fire.Fire(train)
