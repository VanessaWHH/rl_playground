from typing import Dict, Optional, Any

import os
import random
import sys
import math
import numpy as np
import json

from PIL import Image
import gymnasium as gym

from .snake_utils.entities import (
    Colors,
    Snake,
    Field,
    CellType,
    ACTION_TO_SNAKE_DIRECTION,
    ALL_SNAKE_DIRECTIONS,
)


class SnakeEnv(gym.Env):

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(
        self,
        seed: int = 0,
        level_map: str = None,
        render_mode: Optional[str] = None,
        limit_step: bool = True,
        debug: bool = False,
    ):
        """
        Create a snake environments
        """
        random.seed(seed)
        # game related
        with open(level_map) as cfg:
            level_map = json.load(cfg)
        random_walls = level_map.get("random_walls", 0)
        self.refresh_wall = level_map.get("refresh_wall", False)
        self.field = Field(level_map=level_map["field"], random_walls=random_walls)
        self.snake = None
        self.initial_snake_length = 3
        self.foods = []
        self.board_size = self.field.size
        self.grid_size = self.board_size**2
        self.max_growth = self.grid_size - self.initial_snake_length

        # rl model related
        # 0: UP, 1: LEFT, 2: RIGHT, 3: DOWN
        self.action_space = gym.spaces.discrete.Discrete(4)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.board_size * 7, self.board_size * 7, 3),
            dtype=np.uint8,
        )
        # # More than enough steps to get the food or no limit
        self.step_limit = self.grid_size * 4 if limit_step else 1e9

        # render related
        self.render_mode = render_mode
        self.screen = None
        self.font = None
        self.isopen = True
        self.cell_size = 40
        self.width = self.height = self.board_size * self.cell_size
        self.border_size = 20
        self.display_width = self.width + 2 * self.border_size
        self.display_height = self.height + 2 * self.border_size + 40
        self.debug = debug
        self.k = 0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        """Reset the environment and begin a new episode."""
        super().reset(seed=seed)
        self.field.create_level()
        self.snake = Snake(
            self.field.find_snake_head(), length=self.initial_snake_length
        )
        self.field.place_snake(self.snake)

        self.foods = []
        self._generate_food()

        self.score = 0
        self.reward_step_counter = 0

        obs = self._generate_observation()

        return obs, {}

    def step(self, action):
        self.done, info = self._update_environment(action)
        obs = self._generate_observation()
        reward = 0.0
        self.reward_step_counter += 1

        # Snake fills up the entire board. Game over.
        if info["snake_size"] == self.grid_size:
            reward = self.max_growth * 0.1  # Victory reward
            self.done = True
            return obs, reward, self.done, self.done, info

        if self.reward_step_counter > self.step_limit:  # Step limit reached, game over.
            self.reward_step_counter = 0
            self.done = True
            info["dead_reason"] = "step_limit_reached" 

        # Snake bumps into wall or itself. Episode is over.
        if self.done:
            # Game Over penalty is based on snake size.
            reward = -math.pow(
                self.max_growth, (self.grid_size - info["snake_size"]) / self.max_growth
            )  # (-max_growth, -1)
            reward = reward * 0.1
            return obs, reward, self.done, self.done, info
        elif info["food_obtained"]:
            # Food eaten. Reward boost on snake size and reset reward step counter
            reward = info["snake_size"] / self.grid_size
            self.reward_step_counter = 0
        else:
            # Give a tiny reward/penalty to the agent based on whether it is heading towards the food or not.
            # Not competing with game over penalty or the food eaten reward.
            if np.linalg.norm(
                info["snake_head_pos"] - info["food_pos"]
            ) < np.linalg.norm(info["prev_snake_head_pos"] - info["food_pos"]):
                reward = 1 / info["snake_size"]
            else:
                reward = -1 / info["snake_size"]
            reward = reward * 0.1

        return obs, reward, self.done, self.done, info

    def render(self):
        """Draw the entire game frame."""
        import pygame

        self._init_pygame()
        self._render_background()

        for x in range(self.field.size):
            for y in range(self.field.size):
                self._render_cell(x, y)
        self._draw_score()

        if self.render_mode == "human":
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False

    def _init_pygame(self):
        if self.screen is None:
            import pygame
            from pygame import mixer

            pygame.init()
            if self.render_mode == "human":
                pygame.display.set_caption("GreedySnake")
                self.screen = pygame.display.set_mode(
                    (self.display_width, self.display_height)
                )
                # Load sound effects
                mixer.init()
                self.sound_eat = mixer.Sound("sound/eat.wav")
                self.sound_game_over = mixer.Sound("sound/game_over.wav")
                self.sound_victory = mixer.Sound("sound/victory.wav")
            else:  # mode = "rgb_array"
                self.screen = pygame.Surface((self.display_width, self.display_height))
            self.font = pygame.font.Font(None, 36)

    def _render_background(self):
        import pygame

        self.screen.fill((224, 255, 255))

        # Draw border
        pygame.draw.rect(
            self.screen,
            (255, 255, 255),
            (
                self.border_size - 2,
                self.border_size - 2,
                self.width + 4,
                self.height + 4,
            ),
            2,
        )

    def _draw_score(self):
        score_text = self.font.render(f"Score: {self.score}", True, (0, 0, 0))
        self.screen.blit(
            score_text, (self.border_size, self.height + 2 * self.border_size)
        )

    def _render_cell(self, x, y):
        """Draw the cell specified by the field coordinates."""
        import pygame

        cell_coords = pygame.Rect(
            y * self.cell_size + self.border_size,
            x * self.cell_size + self.border_size,
            self.cell_size,
            self.cell_size,
        )
        if self.field[x, y] == CellType.EMPTY:
            pygame.draw.rect(self.screen, Colors.SCREEN_BACKGROUND, cell_coords)
        else:
            color = Colors.CELL_TYPE[self.field[x, y]]
            pygame.draw.rect(self.screen, color, cell_coords, 1)

            internal_padding = self.cell_size // 6 * 2
            internal_square_coords = cell_coords.inflate(
                (-internal_padding, -internal_padding)
            )
            pygame.draw.rect(self.screen, color, internal_square_coords)
            if self.field[x, y] == CellType.SNAKE_HEAD:
                self._draw_eyes(cell_coords.left, cell_coords.top)

    def _draw_eyes(self, x, y):
        import pygame

        eye_size = 3
        eye_offset = self.cell_size // 4
        pygame.draw.circle(
            self.screen,
            (255, 255, 255),
            (x + eye_offset, y + eye_offset),
            eye_size,
        )
        pygame.draw.circle(
            self.screen,
            (255, 255, 255),
            (x + self.cell_size - eye_offset, y + eye_offset),
            eye_size,
        )

    # EMPTY: BLACK; SnakeBODY: GRAY; SnakeHEAD: GREEN; FOOD: RED;
    def _generate_observation(self):
        """generate obstavation"""
        obs = np.zeros((self.board_size, self.board_size), dtype=np.uint8)

        # Set the snake body to gray with linearly decreasing intensity from head to tail.
        snake = [(p.x, p.y) for p in self.snake.body]
        # print("snake: ", snake)
        obs[tuple(np.transpose(snake))] = np.linspace(
            200, 50, len(snake), dtype=np.uint8
        )
        # Stack single layer into 3-channel-image.
        obs = np.stack((obs, obs, obs), axis=-1)
        # Set the snake head to green and the tail to blue
        obs[tuple(snake[0])] = [0, 255, 0]
        obs[tuple(snake[-1])] = [255, 0, 0]
        # Set the food to red
        obs[self.foods[0]] = [0, 0, 255]
        # set walls to white
        walls = [(p.x, p.y) for p in self.field._wall_cells]
        # print("walls: ", walls)
        if len(walls) > 0:
            obs[tuple(np.transpose(walls))] = [255, 255, 255]

        # Enlarge the observation to 84x84
        obs = np.repeat(np.repeat(obs, 7, axis=0), 7, axis=1)

        if self.debug:
            os.makedirs("./gif/tmp", exist_ok=True)
            image = Image.fromarray(obs, "RGB")
            image.save("./gif/tmp/" + str(self.k) + ".jpg")
            self.k += 1

        return obs

    def _update_environment(self, action):
        direction = ACTION_TO_SNAKE_DIRECTION[int(action)]
        self.snake.check_and_update_direction(direction)

        old_head = self.snake.head
        old_tail = self.snake.tail
        next_move = self.snake.peek_next_move()

        dead = False
        dead_reason = ""

        # 1. Check if snake eats food.
        # If snake eats food, it won't pop the last cell. The food grid will be taken by snake later, no need to update board vacancy matrix.
        if next_move in self.foods:
            self.foods.remove(next_move)
            food_obtained = True
            self.snake.grow()
            self.score += 10
            old_tail = None
        else:
            food_obtained = False

        # 2. check if out level map
        if self._has_out_level_map(next_move):
            dead = True
            dead_reason = "out_level_map"
        else:
            if not food_obtained:
                self.snake.move()
            self.field.update_snake_footprint(old_head, old_tail, self.snake.head)
            # 2. Check if snake collided with itself or the wall
            if self._has_hit_own_body(next_move):
                dead = True
                dead_reason = "hit_own_body"
            elif self._has_hit_wall(next_move):
                dead = True
                dead_reason = "hit_wall"

        # 3. Add new food after snake movement completes.
        if food_obtained:
            self._generate_food()
            if self.refresh_wall:
                self.field.place_random_walls()

        info = {
            "snake_size": self.snake.length,
            "snake_head_pos": np.array(self.snake.body[0]),
            "prev_snake_head_pos": np.array(self.snake.body[1]),
            "food_pos": np.array(self.foods[0]),
            "food_obtained": food_obtained,
            "dead_reason": dead_reason,
        }
        return dead, info

    def get_action_mask(self):
        return np.array(
            [[self._check_action_if_valid(d) for d in ALL_SNAKE_DIRECTIONS]]
        )

    def _check_action_if_valid(self, d):
        pos = self.snake.head + d
        return (
            self.snake.check_direction(d)
            and not self._has_out_level_map(pos)
            and not self._has_hit_wall(pos)
            and not self._has_hit_own_body(pos)
        )

    def _generate_food(self, position=None):
        """Generate a new food at a random unoccupied cell."""
        if position is None:
            position = self.field.get_random_empty_cell()
        self.field[position] = CellType.FOOD
        self.foods.append(position)

    def _has_out_level_map(self, pos):
        row, col = pos
        return row < 0 or row >= self.board_size or col < 0 or col >= self.board_size

    def _has_hit_wall(self, pos):
        """True if the snake has hit a wall, False otherwise."""
        return self.field[pos] == CellType.WALL

    def _has_hit_own_body(self, pos):
        """True if the snake has hit its own body, False otherwise."""
        return self.field[pos] == CellType.SNAKE_BODY

    def _is_alive(self, pos):
        """True if the snake is still alive, False otherwise."""
        return not self._has_hit_wall(pos) and not self._has_hit_own_body(pos)
