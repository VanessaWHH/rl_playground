import torch
from torch import nn

from common import gen_ts

from rl_zoo3.utils import linear_schedule


class a2c:
    algo = "a2c"
    n_envs = 32
    ## If killed when n_nevs=32, n_envs is set to 8
    # n_envs = 8
    level_map = "envs/levels/12x12-walls.json"
    pretrain = "./trained_models/trained_models_a2c_cnn_nenvs32_best/a2c_snake_final.zip"
    n_timesteps = 1e8

    time = gen_ts()
    save_dir = f"train_walls_{algo}_{n_envs}/" + f"{time}"

    class model_params:
        gae_lambda = 0.9
        gamma = 0.99
        n_steps = 8
        ent_coef = 0.01
        vf_coef = 0.4
        learning_rate = linear_schedule(8e-4)




