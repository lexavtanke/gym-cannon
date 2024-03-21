from env.cannon import CannonEnv

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env

import numpy as np
import click
from statistics import mean 

def play_human():

    env = CannonEnv()
    env.reset()
    terminated = False
    total_reward = 0
    while not terminated:
        env.render()
        action = np.array([float(input("Enter a start velocity for the shot:"))], dtype=np.float32)
        observation, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        print(f"reward for the {action} speed is {reward}")
        if terminated:
            if reward >= 100:
                print("You engage the target")
            else:
                print("You miss the target")
    print(f"Total reward for the episode is {total_reward}")

def play_ppo(model_name):
    n_episodes = 1000
    eval_vec_env = make_vec_env(CannonEnv)
    eval_vec_env = VecNormalize(eval_vec_env, norm_obs=True, norm_reward=False,
                   clip_obs=10.)
    
    model = PPO.load(model_name)
    episode_reward_list = []
    wins = 0
    for i in range(n_episodes):
        episode_reward = 0
        reward_list = []
        obs = eval_vec_env.reset()
        terminated = False
        while not terminated:
            action, _states = model.predict(obs)
            obs, reward, terminated, info = eval_vec_env.step(action)
            episode_reward += reward
            reward_list.append(reward[0])
            if terminated and reward >= 100:
                wins +=1
                print(f"{i} episode reward is {episode_reward[0]}")
                print(f"episode reward list is {reward_list}")
            if terminated:
                episode_reward_list.append(episode_reward[0])
    mean_episode_reward = mean(episode_reward_list)
    print(f"Statistics for {n_episodes}:")
    print(f"{wins} targets engaged, mean_episode_reward is {mean_episode_reward}")

def play_sac(model_name):
    n_episodes = 1000
    eval_vec_env = make_vec_env(CannonEnv)
    eval_vec_env = VecNormalize(eval_vec_env, norm_obs=True, norm_reward=False,
                   clip_obs=10.)
    
    model = SAC.load(model_name)
    episode_reward_list = []
    wins = 0
    for i in range(n_episodes):
        episode_reward = 0
        reward_list = []
        obs = eval_vec_env.reset()
        terminated = False
        while not terminated:
            action, _states = model.predict(obs)
            obs, reward, terminated, info = eval_vec_env.step(action)
            episode_reward += reward
            reward_list.append(reward[0])
            if terminated and reward >= 100:
                wins +=1
                print(f"{i} episode reward is {episode_reward[0]}")
                print(f"episode reward list is {reward_list}")
            if terminated:
                episode_reward_list.append(episode_reward[0])
    mean_episode_reward = mean(episode_reward_list)
    print(f"Statistics for {n_episodes}:")
    print(f"{wins} targets engaged, mean_episode_reward is {mean_episode_reward}")

def play_random():
    n_episodes = 1000
    eval_vec_env = make_vec_env(CannonEnv)
    eval_vec_env.reset()
    terminated = False
    episode_reward_list = []
    wins = 0
    for i in range(n_episodes):
        episode_reward = 0
        reward_list = []
        obs = eval_vec_env.reset()
        terminated = False
        while not terminated:
            action = eval_vec_env.action_space.sample()
            obs, reward, terminated, info = eval_vec_env.step([action])
            episode_reward += reward
            reward_list.append(reward[0])
            if terminated and reward >= 100:
                wins +=1
                print(f"{i} episode reward is {episode_reward[0]}")
                print(f"episode reward list is {reward_list}")
            if terminated:
                episode_reward_list.append(episode_reward[0])
    mean_episode_reward = mean(episode_reward_list)
    print(f"Statistics for {n_episodes}:")
    print(f"{wins} targets engaged, mean_episode_reward is {mean_episode_reward}")


@click.command()
@click.option("--mode", default="human", help="ppo, human or random")
@click.option("--model_name", default="ppo_gunner")
def play(mode="human", model_name="ppo_gunner_normalize_env_5000k_save_shot"):

    if mode == "human":
        play_human()
    elif mode == "random":
        play_random()
    elif mode == "ppo":
        play_ppo(model_name)
    elif mode == "sac":
        play_sac(model_name)
        


if __name__ == "__main__":
    play()