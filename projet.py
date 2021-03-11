import retro
from baselines.common.retro_wrappers import *

def main():
    env = retro.make(game='Celeste-GBA', state='Level1')
    obs = env.reset()
    while True:
        obs, rew, done, info =  env.step(env.action_space.sample())
        env.render()
        if done:
            obs = env.reset()
    env.close()
    


if __name__ == "__main__":
    main()