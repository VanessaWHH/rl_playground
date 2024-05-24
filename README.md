# rl_playground

## Greedy Snake Game

### Display

see the gif folder for more displays.

![image](https://github.com/VanessaWHH/rl_playground/assets/94059478/a0fd846f-cc5e-488d-ba0c-9d472c245d66) ![image](https://github.com/VanessaWHH/rl_playground/assets/94059478/39daaf50-be11-4e16-8d57-6c9e9ef89b54) ![image](https://github.com/VanessaWHH/rl_playground/assets/94059478/1cf46c78-68ec-40a1-8e08-350a7ef56550)

### Experimental Environment Configuration

1. Ubuntu
2. GPU
3. external libraries
```bash
pip install -r requirements.txt
```

### Overall Architecture

![image](https://github.com/VanessaWHH/rl_playground/assets/94059478/2e151a34-3654-44f9-beb0-ffa7753ea32e)

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

1. 'Environment' module: envs
2. 'train' module: train
3. 'train_hpo' module: train_hpo
4. 'eval' module: eval.py trained_models
5. 'config' module: configs configs_hpo common.py

### Game Environment

1. without obstacles
2. with obstacles

### RL Algorithm Used

1. DQN
2. A2C
3. PPO
4. Maskable PPO

### Training Agents

to train the Snake agent in an obstacle-free game environment.
'ppo' can be changed to 'dqn', 'a2c', 'maskableppo'.
```bash
python train.py configs/config_blank.py ppo
```
to train the Snake agent in a game environment with obstacles.
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
![image](https://github.com/VanessaWHH/rl_playground/assets/94059478/b96e1777-652f-4b93-a3b9-f4c8cade53ef) ![image](https://github.com/VanessaWHH/rl_playground/assets/94059478/26be3a52-861d-49dc-93e8-0307d6447e24)
![image](https://github.com/VanessaWHH/rl_playground/assets/94059478/82f4ba84-ecc4-44a3-bf52-5d89de214e71) ![image](https://github.com/VanessaWHH/rl_playground/assets/94059478/7df1a6c5-de57-4e19-b798-b2a5602f5cd2)
![image](https://github.com/VanessaWHH/rl_playground/assets/94059478/1b6fd5a0-03fe-48f5-96e0-a1a8a4768dbc) ![image](https://github.com/VanessaWHH/rl_playground/assets/94059478/064f84ec-f00a-473f-a4bd-6bc773c2bf52)
![image](https://github.com/VanessaWHH/rl_playground/assets/94059478/404f83e5-2aa3-4b6d-9c2d-8df422fc4041) ![image](https://github.com/VanessaWHH/rl_playground/assets/94059478/6344a167-3583-435b-a221-018320bf9d35)
![image](https://github.com/VanessaWHH/rl_playground/assets/94059478/f9fcbaed-eb12-415a-b194-1fcca04c4475) 
![image](https://github.com/VanessaWHH/rl_playground/assets/94059478/78cd326d-a16d-42ce-bafa-1d27a43adcad)
![image](https://github.com/VanessaWHH/rl_playground/assets/94059478/9877a4c1-f05a-4224-924c-6eb3e5f87d59)




