from rl_zoo3.utils import linear_schedule
from torch import nn


def gen_ts():
    import time

    return time.strftime("%y%m%d-%H%M%S", time.localtime())


def my_linear_schedule(initial_value, final_value=0.0):
    if isinstance(initial_value, str):
        initial_value = float(initial_value)
        final_value = float(final_value)
        assert initial_value > 0.0

    def scheduler(progress):
        return final_value + progress * (initial_value - final_value)

    return scheduler


class a2c:
    algo = "a2c"
    n_envs = 32
    # If killed when n_nevs=32, n_envs is set to 8
    # n_envs = 8
    level_map = "envs/levels/12x12-blank.json"
    pretrain = ""
    n_timesteps = 1e8

    time = gen_ts()
    save_dir = f"train_blank_{algo}_{n_envs}/" + f"{time}"

    class model_params:
        gae_lambda = 0.9
        gamma = 0.99
        n_steps = 8
        ent_coef = 0.01
        vf_coef = 0.4
        learning_rate = linear_schedule(8e-4)
        policy_kwargs = dict(activation_fn=nn.ReLU)


class dqn:
    algo = "dqn"
    n_envs = 8
    level_map = "envs/levels/12x12-blank.json"
    pretrain = ""
    n_timesteps = 1e8

    time = gen_ts()
    save_dir = f"train_blank_{algo}_{n_envs}/" + f"{time}"

    class model_params:
        batch_size = 64
        gamma = 0.99
        learning_rate = linear_schedule(2.5e-4)
        buffer_size = 1000
        exploration_final_eps = 0.01
        target_update_interval = 100


class ppo:
    algo = "ppo"
    n_envs = 32
    # If killed when n_nevs=32, n_envs is set to 8
    # n_envs = 8
    level_map = "envs/levels/12x12-blank.json"
    pretrain = ""
    n_timesteps = 1e8

    time = gen_ts()
    save_dir = f"train_blank_{algo}_{n_envs}/" + f"{time}"

    class model_params:
        n_steps = 2048
        batch_size = 512
        n_epochs = 4
        gamma = 0.94
        learning_rate = my_linear_schedule(2.5e-4, 2.5e-6)
        clip_range = my_linear_schedule(0.150, 0.025)


class maskableppo:
    algo = "maskableppo"
    n_envs = 32
    # If killed when n_nevs=32, n_envs is set to 8
    # n_envs = 8
    level_map = "envs/levels/12x12-blank.json"
    pretrain = ""
    n_timesteps = 1e8

    time = gen_ts()
    save_dir = f"train_blank_{algo}_{n_envs}/" + f"{time}"

    class model_params:
        n_steps = 2048
        batch_size = 512
        n_epochs = 4
        gamma = 0.94
        learning_rate = my_linear_schedule(2.5e-4, 2.5e-6)
        clip_range = my_linear_schedule(0.150, 0.025)