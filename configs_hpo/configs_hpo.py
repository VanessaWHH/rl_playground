from torch import nn 

from common import gen_ts

class a2c:
    algo = "a2c"
    n_envs = 32
    # If killed when n_nevs=32, n_envs is set to 8
    # n_envs = 8
    level_map = "envs/levels/12x12-blank.json"
    pretrain = ""
    n_timesteps = 1e6

    time = gen_ts()
    save_dir_prefix = f"optuna_{algo}_{n_envs}/" + f"{time}" + "/trial_"

    class search_space:
        gae_lambda = [0.9, 1.0]
        n_steps = [5, 8]
        ent_coef = [1e-9, 0.01]
        vf_coef = [0.25, 0.4]
        learning_rate = (7e-4, 1e-3)
        activation_fn_name = ["tanh", "relu"]

    class model_params:
        gae_lambda = None
        n_steps = None
        ent_coef = None
        vf_coef = None
        learning_rate = None
        policy_kwargs = dict(
            activation_fn = None,
            ortho_init = False,
        )


class dqn:
    algo = "dqn"
    n_envs = 8
    level_map = "envs/levels/12x12-blank.json"
    pretrain = ""
    n_timesteps = 1e6

    time = gen_ts()
    save_dir_prefix = f"optuna_{algo}_{n_envs}/" + f"{time}" + "/trial_"

    class search_space:
        learning_rate = (5e-7, 1e-3)
        batch_size = [32, 64, 128, 256, 512]
        buffer_size = [500, 1000, 1500, 2000, 4000]

    class model_params:
        learning_rate = None
        batch_size = None
        buffer_size = None


class ppo:
    algo = "ppo"
    n_envs = 32
    # If killed when n_nevs=32, n_envs is set to 8
    # n_envs = 8
    level_map = "envs/levels/12x12-blank.json"
    pretrain = ""
    n_timesteps = 1e6

    time = gen_ts()
    save_dir_prefix = f"optuna_{algo}_{n_envs}/" + f"{time}" + "/trial_"
    
    class search_space:
        batch_size = [8, 16, 32, 64, 128]
        n_steps = [8, 16, 32, 64]
        gamma = [0.9, 0.95, 0.98]
        learning_rate = (5e-5, 1e-4)
        ent_coef = (1e-9, 0.01)
        clip_range = [0.1, 0.2, 0.3, 0.4]
        n_epochs = [1, 4, 5]
        gae_lambda = [0.8, 0.9, 0.92, 0.95]
        max_grad_norm = [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        vf_coef = (1e-9, 1)
        activation_fn_name = ["tanh", "relu"]

    class model_params:
        batch_size = None
        n_steps = None
        gamma = None
        learning_rate = None
        ent_coef = None
        clip_range = None
        n_epochs = None
        gae_lambda = None
        max_grad_norm = None
        vf_coef = None
        policy_kwargs = dict(
            activation_fn = None,
            ortho_init = False,
        )


class maskableppo:
    algo = "maskableppo"
    n_envs = 32
    # If killed when n_nevs=32, n_envs is set to 8
    # n_envs = 8
    level_map = "envs/levels/12x12-blank.json"
    pretrain = ""
    n_timesteps = 1e6

    time = gen_ts()
    save_dir_prefix = f"optuna_{algo}_{n_envs}/" + f"{time}" + "/trial_"
    
    class search_space:
        batch_size = [8, 16, 32, 64, 128]
        n_steps = [8, 16, 32, 64]
        gamma = [0.9, 0.95, 0.98]
        learning_rate = (5e-5, 1e-4)
        ent_coef = (1e-9, 0.01)
        clip_range = [0.1, 0.2, 0.3, 0.4]
        n_epochs = [1, 4, 5]
        gae_lambda = [0.8, 0.9, 0.92, 0.95]
        max_grad_norm = [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        vf_coef = (1e-9, 1)
        activation_fn_name = ["tanh", "relu"]

    class model_params:
        batch_size = None
        n_steps = None
        gamma = None
        learning_rate = None
        ent_coef = None
        clip_range = None
        n_epochs = None
        gae_lambda = None
        max_grad_norm = None
        vf_coef = None
        policy_kwargs = dict(
            activation_fn = None,
            ortho_init = False,
        )
