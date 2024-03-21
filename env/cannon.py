from typing import Tuple, Optional
import gymnasium as gym
import numpy as np
import math
import random
from gymnasium import logger, spaces

class CannonEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self) -> None:
        super().__init__()
        self.gravity = 9.81
        self.min_action = -1.0
        self.max_action = 1.0
        self.min_speed = 0.1
        self.max_speed = 1000
        self.min_angle = 0.0
        self.max_angle = math.pi / 2.0
        self.min_distance = 1.0
        self.max_distance = 1000.0
        self.engage_reward = 100
        self.save_shot_reward = 10
        self.game_over_reward = 0
        self.shoot_cost = -10

        self.observation_space = spaces.Box(low=np.array([self.min_angle, self.min_distance]), 
                                            high=np.array([self.max_angle, self.max_distance]), 
                                            dtype=np.float32)
        self.action_space = spaces.Box(low=self.min_action, 
                                       high=self.max_action, 
                                       shape=(1, ), 
                                       dtype=np.float32)
        
    def step(self, action: np.ndarray) -> Tuple[np.array, float, bool, bool, dict]:
        speed = (self.min_speed +
                 (action[0] - self.min_action) / (self.max_action - self.min_action) *
                 (self.max_speed - self.min_speed))
        angle = self.state[0]
        distance = self.state[1]
        calculated_distance = (speed ** 2) * math.sin(math.radians(angle)) / self.gravity
        terminated = bool(abs(distance - calculated_distance) < 1)
        game_over = False
        self.n_shots -= 1

        reward = 0

        reward -=  abs(distance - calculated_distance) # more gap - more penalty

        if self.n_shots < 1:
            game_over = True
        
        if terminated:
            reward = self.engage_reward
            reward += self.n_shots * self.save_shot_reward
        if game_over:
            terminated = True
            reward = self.game_over_reward

        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        cannon_angle = random.uniform(self.min_angle, self.max_angle)
        distanse_to_target = random.uniform(self.min_distance, self.max_distance)
        self.state = np.array([cannon_angle, distanse_to_target], dtype=np.float32)
        self.n_shots = 10

        return np.array(self.state, dtype=np.float32), {}
    
    def render(self):
        print("----------")
        print(f"current cannon angle is {self.state[0]} current distance to target is {self.state[1]}")
        print(f"n shots left {self.n_shots}")

