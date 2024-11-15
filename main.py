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
    number_of_attackers = 2
    number_of_defenders = 1
    number_of_vips = 1    
    while not done:
        attacker_actions = [random.randint(0, env.attacker_defender_action_space - 1) for _ in range(number_of_attackers)]
        print(f'attacker actions: {attacker_actions}')
        defender_actions = [random.randint(0, env.attacker_defender_action_space - 1) for _ in range(number_of_defenders)]
        vip_actions = [random.randint(0, env.vip_action_space - 1) for _ in range(number_of_vips)]
        total_actions = [attacker_actions, defender_actions, vip_actions]
        
        state_reward, done = env.step(total_actions)
        vip_line_of_sight = env.line_of_sight(env.vip_positions)
        defender_line_of_sight = env.line_of_sight(env.defender_positions)
        attacker_line_of_sight = env.line_of_sight(env.attacker_positions)
        env.render()
        time.sleep(0.05)  # Wait for 100ms

    # env.render_line_of_sight(vip_line_of_sight)
    # env.render_line_of_sight(defender_line_of_sight)
    env.render_line_of_sight(attacker_line_of_sight)

    env.reset()
