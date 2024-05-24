import os
import time
import random

import fire
import imageio
import numpy as np

from sb3_contrib import MaskablePPO, QRDQN
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3 import A2C, PPO, DQN

from envs.snake_env import SnakeEnv


"""
1. eval trained model
python eval.py eval_agent a2c ../trained_models/trained_models_a2c_cnn_nenvs32_best \
    --level_map envs/levels/12x12-walls.json \
    --save_to_gif gif_test

2. compare models
python eval.py eval_all_blank gif_test_all
python eval.py eval_all_walls gif_walls_test_all

python eval.py eval_all_blank gif_test_all >eval_log.txt
python eval.py eval_all_walls gif_walls_test_all >eval_log.txt
"""


models = {
    "a2c": A2C,
    "ppo": PPO,
    "dqn": DQN,
    "maskableppo": MaskablePPO,
    "qrdqn": QRDQN,
}


def make_env(
    level_map: str,
    seed: int = 0,
    render_mode=None,
    with_action_mask=False,
):
    env = SnakeEnv(
        seed=seed, level_map=level_map, limit_step=True, render_mode=render_mode
    )
    if with_action_mask == "maskableppo":
        env = ActionMasker(env, SnakeEnv.get_action_mask)
    return env


def eval_agent(
    model: str, model_path: str, level_map: str, algo=None, save_to_gif: str = None
):
    print(f">>> start eval agent {model}, {model_path}")
    NUM_EPISODE = 10

    seed = random.randint(0, 1e9)
    print(f"Using seed = {seed} for testing.")

    with_action_mask = True if model == "maskableppo" else False
    render_mode = None if not save_to_gif else "rgb_array"
    env = make_env(
        seed=seed,
        level_map=level_map,
        render_mode=render_mode,
        with_action_mask=with_action_mask,
    )

    # Load the trained model
    if algo:
        agent = algo.load(model_path)
    elif model in models:
        agent = models[model].load(model_path)
    else:
        raise NameError(f"no such agent model: '{model}'")

    total_reward = 0
    total_score = 0
    min_score = 1e9
    max_score = 0

    for episode in range(NUM_EPISODE):
        if save_to_gif:
            images = []
            os.makedirs(os.path.join(save_to_gif, model), exist_ok=True)

        episode_reward = 0
        done = False
        num_step = 0
        sum_step_reward = 0
        print(f"=================== Episode {episode + 1} ==================")
        obs = env.reset()
        obs = obs[0]
        if save_to_gif:
            image = env.render()
        while not done:
            if save_to_gif:
                images.append(image)
            if with_action_mask:
                action, _ = agent.predict(obs, action_masks=env.get_action_mask())
                prev_mask = env.get_action_mask()
            else:
                action, _ = agent.predict(obs)

            num_step += 1
            obs, reward, done, _, info = env.step(action)

            if done:
                if info["snake_size"] == env.grid_size:
                    print(f"You are BREATHTAKING! Victory reward: {reward:.4f}.")
                else:
                    last_action = ["UP", "LEFT", "RIGHT", "DOWN"][action]
                    print(
                        f"Gameover Penalty: {reward:.4f}. Last action: {last_action}, Dead reason: {info['dead_reason']}"
                    )

            elif info["food_obtained"]:
                print(
                    f"Food obtained at step {num_step:04d}. Food Reward: {reward:.4f}. Step Reward: {sum_step_reward:.4f}"
                )
                sum_step_reward = 0

            else:
                sum_step_reward += reward

            episode_reward += reward
            if save_to_gif:
                image = env.render()
                # time.sleep(FRAME_DELAY)

        episode_score = env.score
        if episode_score < min_score:
            min_score = episode_score
        if episode_score > max_score:
            max_score = episode_score

        snake_size = info["snake_size"] + 1
        print(
            f"Episode {episode + 1}: Reward Sum: {episode_reward:.4f}, Score: {episode_score}, Total Steps: {num_step}, Snake Size: {snake_size}"
        )
        total_reward += episode_reward
        total_score += env.score

        if save_to_gif:
            gif_path = os.path.join(save_to_gif, model, f"{model}-ep{episode}.gif")
            fps = 30
            # speed[2x,3x]
            step = min(3, max(2, len(images) // 30))
            print(f"save gif to {gif_path}, fps: {fps}")
            imageio.mimsave(
                f"{gif_path}",
                [np.array(img) for i, img in enumerate(images) if i % step == 0],
                fps=30,
            )

    env.close()
    print(f"=================== Summary ==================")
    summary = {
        "Average Score": total_score / NUM_EPISODE,
        "Min Score": min_score,
        "Max Score": max_score,
        "Average reward": total_reward / NUM_EPISODE,
    }
    print(summary)
    
    return summary


def eval_all_blank(save_to_gif: str):
    import pandas as pd

    level_map = "envs/levels/12x12-blank.json"
    models = {
        "a2c": (A2C, "./trained_models/trained_models_a2c_cnn_nenvs32_best/a2c_snake_final.zip"),
        "a2c_baseline": (A2C, "./trained_models/trained_models_a2c_cnn_nenvs32/a2c_snake_final.zip"),
        "ppo": (PPO, "./trained_models/trained_models_ppo_cnn_nenvs32/ppo_snake_final.zip"),
        "maskableppo": (MaskablePPO, "./trained_models/trained_models_maskableppo_cnn_nenvs32/maskableppo_snake_final.zip"),
        "dqn": (DQN, "./trained_models/trained_models_dqn_cnn_nenvs8_best/dqn_snake_final.zip"),
        "dqn_baseline": (DQN, "./trained_models/trained_models_dqn_cnn_nenvs8/dqn_snake_final.zip"),
        # "qrdqn": (QRDQN, ""),
    }
    summaries = {}
    for model, agent in models.items():
        summaries[model] = eval_agent(
            model=model,
            algo=agent[0],
            model_path=agent[1],
            level_map=level_map,
            save_to_gif=save_to_gif,
        )
    print(pd.DataFrame.from_dict(summaries))


def eval_all_walls(save_to_gif: str):
    import pandas as pd

    # prefix = "./trained_models/"
    # no dqn
    models = {
        "a2c-walls": (
            A2C,
            "./trained_models/trained_models_walls_a2c_32/a2c_final.zip",
            "envs/levels/12x12-walls.json",
        ),
        "a2c-refresh": (
            A2C,
            "./trained_models/trained_models_walls_a2c_32/a2c_final.zip",
            "envs/levels/12x12-refresh.json",
        ),
        "ppo-walls": (
            PPO,
            "./trained_models/trained_models_walls_ppo_32/ppo_final.zip",
            "envs/levels/12x12-walls.json",
        ),
        "ppo-refresh": (
            PPO,
            "./trained_models/trained_models_walls_ppo_32/ppo_final.zip",
            "envs/levels/12x12-refresh.json",
        ),
        "maskableppo-walls": (
            MaskablePPO,
            "./trained_models/trained_models_walls_maskableppo_32/maskableppo_final.zip",
            "envs/levels/12x12-walls.json",
        ),
        "maskableppo-refresh": (
            MaskablePPO,
            "./trained_models/trained_models_walls_maskableppo_32/maskableppo_final.zip",
            "envs/levels/12x12-refresh.json",
        ),
    }
    summaries = {}
    for model, agent in models.items():
        summaries[model] = eval_agent(
            model=model,
            algo=agent[0],
            model_path=agent[1],
            level_map=agent[2],
            save_to_gif=save_to_gif,
        )
    print(pd.DataFrame.from_dict(summaries))


if __name__ == "__main__":
    fire.Fire()
