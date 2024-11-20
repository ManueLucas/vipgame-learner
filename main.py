import environment
import numpy as np
import random
import time
import pygame


def individual_state(state, position, grid_width): #use this function to generate a state suitable for the agent to learn from. This function will take in a state and a position and return a state with the agent's position marked with a 5
    state = state.copy()
    state[position[0]*grid_width + position[1]] = 5
    state = state.astype(dtype="float32", order='K', casting='unsafe', subok=True, copy=True)
    return state

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
        defender_actions = [random.randint(0, env.attacker_defender_action_space - 1) for _ in range(number_of_defenders)]
        vip_actions = [random.randint(0, env.vip_action_space - 1) for _ in range(number_of_vips)]
        total_actions = [attacker_actions, defender_actions, vip_actions]
        fully_visible_state, team_states, team_rewards, team_positions, done = env.step(total_actions) #fully_visible is a single state, team_states is a list of states, team_rewards is a list of rewards, team_positions is a list of list of tuples, done is a boolean
        print(team_rewards)
        env.render()
        time.sleep(0.05)  # Wait for 100ms

    # env.render_line_of_sight(vip_line_of_sight)
    # env.render_line_of_sight(defender_line_of_sight)

    env.render_line_of_sight(team_states[0])
    env.render_line_of_sight(team_states[1])
    env.render_line_of_sight(individual_state(team_states[1], team_positions[1][0]))
        

    env.reset()
