import retro
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
import torch




def main():
    env = SubprocVecEnv([lambda: retro.make(game='Celeste-GBA', state='Level1', scenario='scenario', record="records")])
    model = PPO('CnnPolicy', env, verbose=1)    
    model.learn(
        total_timesteps=int(100e6), 
        callback=None, 
        log_interval=1000,
        tb_log_name='PPO2',
        reset_num_timesteps=True    
    )
    

   


if __name__ == "__main__":
    main()