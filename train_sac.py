import datetime

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement

from env.cannon import CannonEnv

env = CannonEnv()
check_env(env)
vec_env = make_vec_env(CannonEnv, n_envs=1000)
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True,
                   clip_obs=10.)

# Separate evaluation env
eval_vec_env = make_vec_env(CannonEnv)
eval_vec_env = VecNormalize(eval_vec_env, norm_obs=True, norm_reward=True,
                   clip_obs=10.)
# Use deterministic actions for evaluation
stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=5, min_evals=10, verbose=1)
eval_callback = EvalCallback(eval_vec_env, best_model_save_path="./logs/final/sac",
                             log_path="./logs/final/sac", eval_freq=4,
                            #  callback_after_eval=stop_train_callback, 
                             verbose=1,
                             deterministic=True, render=False)

model = SAC("MlpPolicy", vec_env, tensorboard_log="./ppo_gunner_tensorboard/", verbose=1)
start_time = datetime.datetime.now()
print(f"{datetime.datetime.now()} Start training.")
model.learn(total_timesteps=10000000, 
            tb_log_name="16th_sac_final_10mil", 
            callback=eval_callback,
            log_interval=1000)
print(f"Start time was {start_time}")
print(f"{datetime.datetime.now()} Fininsh training.")
model.save("sac_gunner_final_10mil")




