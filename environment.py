import gymnasium as gym
import numpy as np

# Define actions for movement in the grid
ACTIONS = {
    'UP': (-1, 0),
    'DOWN': (1, 0),
    'LEFT': (0, -1),
    'RIGHT': (0, 1),
    'UPLEFT': (-1, -1),
    'UPRIGHT': (-1, 1),
    'DOWNLEFT': (1, -1),
    'DOWNRIGHT': (1, 1)
}

class VipGame(gym.Env):
    def __init__(self, grid_map, max_timesteps=100):
        # Initialize the grid, dimensions, max timesteps, and elapsed timesteps
        self.grid = np.copy(grid_map)
        self.grid_height, self.grid_width = self.grid.shape
        self.max_timesteps = max_timesteps
        self.timesteps_elapsed = 0

        # Initialize positions of agents (attacker, defender, VIP)
        self.attacker_pos = None
        self.defender_pos = None
        self.vip_pos = None

        # Find and set the initial positions of agents
        self._initialize_positions()

    def _initialize_positions(self):
        # Loop through the grid to find initial positions of agents
        for i in range(self.grid_height):
            for j in range(self.grid_width):
                if self.grid[i, j] == 2:
                    self.vip_pos = (i, j)  # VIP position
                elif self.grid[i, j] == 3:
                    self.defender_pos = (i, j)  # Defender position
                elif self.grid[i, j] == 4:
                    self.attacker_pos = (i, j)  # Attacker position

    def _move_agent(self, position, action, moveset):
        # If the action is out of bounds of the moveset, return negative reward
        if action >= len(moveset):
            return position, -1  # Invalid action, negative reward

        # Calculate the new position based on the action taken
        delta = moveset[action]
        new_position = (position[0] + delta[0], position[1] + delta[1])

        # Collision check to ensure the agent stays within grid bounds and avoids walls
        if (0 <= new_position[0] < self.grid_height and
                0 <= new_position[1] < self.grid_width and
                self.grid[new_position] != 1):  # Avoid walls
            return new_position, 0  # Successful move, neutral reward
        return position, -1  # Collision, negative reward

    def attacker_move(self, action):
        # Attacker has access to the full moveset (all 8 directions)
        moveset = list(ACTIONS.values())
        # Update attacker position and get reward for the move
        self.attacker_pos, reward = self._move_agent(self.attacker_pos, action, moveset)
        return reward

    def defender_move(self, action):
        # Defender has access to the full moveset (all 8 directions)
        moveset = list(ACTIONS.values())
        # Update defender position and get reward for the move
        self.defender_pos, reward = self._move_agent(self.defender_pos, action, moveset)
        return reward

    def vip_move(self, action):
        # VIP has a limited moveset (up, down, left, right)
        moveset = [ACTIONS['UP'], ACTIONS['DOWN'], ACTIONS['LEFT'], ACTIONS['RIGHT']]
        # Update VIP position and get reward for the move
        self.vip_pos, reward = self._move_agent(self.vip_pos, action, moveset)
        return reward

    def step(self, actions):
        # Perform actions for each agent (attacker, defender, VIP)
        attacker_action, defender_action, vip_action = actions

        # Execute each agent's move and get their respective rewards
        attacker_reward = self.attacker_move(attacker_action)
        defender_reward = self.defender_move(defender_action)
        vip_reward = self.vip_move(vip_action)

        # Increment the timestep counter
        self.timesteps_elapsed += 1
        # Check if the maximum number of timesteps has been reached
        done = self.timesteps_elapsed >= self.max_timesteps

        return (self.grid, attacker_reward, defender_reward, vip_reward), done

    def reset(self):
        # Reset the environment to its initial state
        self.timesteps_elapsed = 0
        self._initialize_positions()
        return self.grid

    def render(self):
        # Render the current state of the grid
        for i in range(self.grid_height):
            row = ''
            for j in range(self.grid_width):
                if (i, j) == self.vip_pos:
                    row += 'V '  # VIP
                elif (i, j) == self.defender_pos:
                    row += 'D '  # Defender
                elif (i, j) == self.attacker_pos:
                    row += 'A '  # Attacker
                elif self.grid[i, j] == 1:
                    row += '# '  # Wall
                else:
                    row += '. '  # Empty space
            print(row)
        print()
