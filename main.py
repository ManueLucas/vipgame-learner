import environment
import numpy as np
import random
import time
if(__name__ == "__main__"):
    grid_map = np.loadtxt("map_presets/grid.csv", delimiter=",")
    env = environment.VipGame(grid_map=grid_map)
    done = False
    while not done:
        attacker_action = random.randint(0, env.attacker_defender_action_space - 1)
        defender_action = random.randint(0, env.attacker_defender_action_space - 1)
        vip_action = random.randint(0, env.vip_action_space - 1)
        state_reward, done = env.step((attacker_action, defender_action, vip_action))
        env.render()
        time.sleep(0.5)  # Wait for 100ms

    env.reset()
