# rl_playground

## Greedy Snake Game

### Display

see the gif folder for displays

game environment without obstacles

![a2c-ep0-score1410](https://github.com/VanessaWHH/rl_playground/assets/94059478/5e81b7bf-694c-464e-b3b2-75c8996f1857)

at the beginning of each game, obstacles in the game environment are randomly generated, and their positions are fixed during the game

![a2c-walls-ep6-score730](https://github.com/VanessaWHH/rl_playground/assets/94059478/a59e44a5-b683-41d6-9a58-677ed566949b)

at the beginning of each game, obstacles in the game environment are randomly generated, and their positions also change randomly during the game

![a2c-refresh-ep8-score590](https://github.com/VanessaWHH/rl_playground/assets/94059478/4e262e0d-2984-47da-a1f6-4d0f74b7041f)

### Experimental Environment Configuration


```bash
pip install -r requirements.txt
```
```bash
python==3.10.9
pygame==2.5.2
gymnasium==0.29.1
stable-baselines3==2.3.2
sb3-contrib==2.3.0
rl-zoo3==2.3.0
optuna==3.6.1
optuna-dashboard==0.15.1
fire==0.6.0
imageio==2.34.1
tensorboard==2.14.0
```


### Architecture
![image](https://github.com/VanessaWHH/rl_playground/assets/94059478/15bd5288-4702-4206-bba9-365955a6361e)

```bash
├───envs
│   ├───levels
│       ├───12x12-blank
│       ├───12x12-walls
│       └───12x12-refresh
│   ├───snake_utils
│       └───entities
│   └───snake_env
├───configs
│   ├───config_blank
│   └───config_walls
├───configs_hpo
│   └───configs_hpo
├───common
├───train
├───train_hpo
├───trained_models
├───eval
└───gif
```

1. environment module
the environment module contains the implementation of different Snake game environments

```bash
envs
```
2. configuration module
the configuration module contains hyperparameter configurations for different deep reinforcement learning algorithms, such as hyperparameter search ranges and hyperparameter values

```bash
configs configs_hpo common
```
3. training module
the training module contains the complete process of creating models (loading pre-training), training models, logging, etc

```bash
train
```
4. hyperparameter optimization module
this module contains the complete process of hyperparameter optimization

```bash
train_hpo
```
5. evaluation module
the evaluation module contains complete processes such as loading models, environment interaction, and statistical data processes

```bash
eval.py trained_models
```

### Game Environment

1. without obstacles
2. with obstacles

### RL Algorithm Used

1. DQN
2. A2C
3. PPO
4. Maskable PPO

### Training Agents

train the Snake agent in an obstacle-free game environment

'ppo' can be changed to 'dqn', 'a2c', 'maskableppo'
```bash
python train.py configs/config_blank.py ppo
```
train the Snake agent in a game environment with obstacles
```bash
python train.py configs/config_walls.py ppo
```

### Hyperparameter Optimization

search hyperparameters based on optuna
```bash
python train_hpo.py configs_hpo/configs_hpo.py ppo
```

### Evaluating Agents

load the agent in trained_models for evaluation
1. eval trained model
```bash
python eval.py eval_agent a2c ../trained_models/trained_models_a2c_cnn_nenvs32_best \
    --level_map envs/levels/12x12-walls.json \
    --save_to_gif gif_test
```
2. compare models
```bash
# output to terminal
python eval.py eval_all_blank gif_test_all
python eval.py eval_all_walls gif_walls_test_all
# output to file
python eval.py eval_all_blank gif_test_all >eval_log.txt
python eval.py eval_all_walls gif_walls_test_all >eval_log.txt
```

### Training Curves

execute the following command in the home directory to view all training curves in tensorboard
```bash
tensorboard --logdir .
```

omparison of the training curves of the A2C Snake agent under optimal hyperparameters and under default hyperparameters

![image](https://github.com/VanessaWHH/rl_playground/assets/94059478/7df1a6c5-de57-4e19-b798-b2a5602f5cd2)

comparison of the training curves of the Snake agent corresponding to four deep reinforcement learning algorithms under optimal hyperparameters

![image](https://github.com/VanessaWHH/rl_playground/assets/94059478/320339aa-cedc-4900-be63-c5ce5dc15b87)

comparison of training curves of A2C Snake agent under different numbers of parallel environments

![image](https://github.com/VanessaWHH/rl_playground/assets/94059478/84f8e553-7f6f-4748-aa6a-b2a6d96a032b)

comparison of training curves of whether A2C Snake agent loads pre-training

![image](https://github.com/VanessaWHH/rl_playground/assets/94059478/9877a4c1-f05a-4224-924c-6eb3e5f87d59)




