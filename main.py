import environment
import numpy as np
import random
import time

if(__name__ == "__main__"):
    grid_map = np.loadtxt("map_presets/grid.csv", delimiter=",")
    env = environment.VipGame(grid_map=grid_map)
    done = False
    pygame_initialized = False
    screen = None

    while not done:
        attacker_action = random.randint(0, env.attacker_defender_action_space - 1)
        defender_action = random.randint(0, env.attacker_defender_action_space - 1)
        vip_action = random.randint(0, env.vip_action_space - 1)
        state_reward, done = env.step((attacker_action, defender_action, vip_action))
        vip_line_of_sight = env.line_of_sight([env.vip_pos])
        defender_line_of_sight = env.line_of_sight([env.defender_pos])
        attacker_line_of_sight = env.line_of_sight([env.attacker_pos])
        env.render()
        time.sleep(0.05)  # Wait for 100ms

    # env.render_line_of_sight(vip_line_of_sight)
    # env.render_line_of_sight(defender_line_of_sight)
    env.render_line_of_sight(attacker_line_of_sight)

    env.reset()
