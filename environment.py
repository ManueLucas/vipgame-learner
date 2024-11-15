import gymnasium as gym
import numpy as np
import pygame
import collections
from enum import Enum

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
UNSEEN = -1
OPEN = 0
WALL = 1
VIP = 2
DEFENDER = 3
ATTACKER = 4
SELF = 5 
  

class VipGame(gym.Env):
    def __init__(self, grid_map, max_timesteps=100):
        # Initialize the grid, dimensions, max timesteps, and elapsed timesteps
        self.defenderside_collision_set = []
        self.grid = np.copy(grid_map)
        self.grid_height, self.grid_width = self.grid.shape
        self.max_timesteps = max_timesteps
        self.timesteps_elapsed = 0
        self.attacker_defender_action_space = 8
        self.vip_action_space = 4
        self.attackerside_collision_set = [WALL, ATTACKER]
        self.defenderside_collision_set = [WALL, DEFENDER, VIP]
        
        self.num_cell_types = 7  # Number of cell types (-1, 0, 1, 2, 3, 4, 5)

        self.observation_space = gym.spaces.Box(
            low=-1,  # Minimum value (unseen open tile)
            high=5,  # Maximum value (self)
            shape=(self.grid_height, self.grid_width),  # Shape of the matrix
            dtype=np.int32  # Integer data type for the cell values
        )

        # Initialize positions of agents (attacker, defender, VIP) (these are tuples of row, column)
        self.attacker_positions = [] 
        self.defender_positions = []
        self.vip_positions = []
        
        # Pygame initialization flag
        self.pygame_initialized = False
        self.screen = None
        
        # Find and set the initial positions of agents
        self._initialize_positions()

    def _initialize_positions(self):
        # Loop through the grid to find initial positions of agents
        for i in range(self.grid_height):
            for j in range(self.grid_width):
                if self.grid[i, j] == 2:
                    self.vip_positions.append((i, j))  # VIP position
                elif self.grid[i, j] == 3:
                    self.defender_positions.append((i, j))  # Defender position
                elif self.grid[i, j] == 4:
                    self.attacker_positions.append((i, j))  # Attacker position

    def _move_agent(self, position, action, moveset, collisionset):
        # If the action is out of bounds of the moveset, return negative reward
        if action >= len(moveset):
            return position, -1  # Invalid action, negative reward

        # Calculate the new position based on the action taken
        delta = moveset[action]
        new_position = (position[0] + delta[0], position[1] + delta[1])

        # Collision check to ensure the agent stays within grid bounds and avoids walls
        if (0 <= new_position[0] < self.grid_height and
                0 <= new_position[1] < self.grid_width and
                self.grid[new_position] not in collisionset):  # disallow collision with same team and walls
            self.grid[new_position] = self.grid[position]
            self.grid[position] = 0  # Reset the current position to empty
            
            return new_position, 0  # Successful move, neutral reward
        return position, -1  # Collision, negative reward

    def attacker_move(self, action, agent_id):
        # Attacker has access to the full moveset (all 8 directions)
        moveset = list(ACTIONS.values())
        # Update attacker position and get reward for the move
        self.attacker_positions[agent_id], reward = self._move_agent(self.attacker_positions[agent_id], action, moveset, self.attackerside_collision_set)
        return reward

    def defender_move(self, action, agent_id): # we can merge the attacker and defender move functions into one later
        # Defender has access to the full moveset (all 8 directions)
        moveset = list(ACTIONS.values())
        # Update defender position and get reward for the move
        self.defender_positions[agent_id], reward = self._move_agent(self.defender_positions[agent_id], action, moveset, self.defenderside_collision_set)
        return reward

    def vip_move(self, action, agent_id):
        # VIP has a limited moveset (up, down, left, right)
        moveset = [ACTIONS['UP'], ACTIONS['DOWN'], ACTIONS['LEFT'], ACTIONS['RIGHT']]
        # Update VIP position and get reward for the move
        self.vip_positions[agent_id], reward = self._move_agent(self.vip_positions[agent_id], action, moveset, self.defenderside_collision_set)
        return reward

    def step(self, actions): #we expect a tuple of 3 lists of actions, one for each team
        # Perform actions for each agent (attacker, defender, VIP)
        attacker_actions, defender_actions, vip_actions = actions
        
        attacker_reward = 0
        defender_reward = 0
        vip_reward = 0
        # Execute each agent's move and get their respective rewards
        for i, attacker_action in enumerate(attacker_actions):
            attacker_reward += self.attacker_move(attacker_action, i)
            
        for i, defender_action in enumerate(defender_actions):
            defender_reward += self.defender_move(defender_action, i)
            
        for i, vip_action in enumerate(vip_actions):  
            vip_reward += self.vip_move(vip_action, i)

        # Increment the timestep counter
        self.timesteps_elapsed += 1
        # Check if the maximum number of timesteps has been reached
        done = self.timesteps_elapsed >= self.max_timesteps

        return (self.grid, attacker_reward, defender_reward, vip_reward), done
    
    def line_of_sight(self, agent_positions):
        #TODO: take multiple agent positions, reveal the map in their lines of sight and return the grid according to their team's lines of sight. -1: unseen tile, 0: seen tile,
        agent_view = np.where(self.grid == 0, -1, self.grid)

        for position in agent_positions:
            for delta in ACTIONS.values():
                current_position = position
                
                while True:
                    # new position in chosen direction
                    new_position = (current_position[0] + delta[0], current_position[1] + delta[1])

                    # Collision check to ensure vision stays within grid bounds and avoids walls
                    if (0 <= new_position[0] < self.grid_height and
                            0 <= new_position[1] < self.grid_width and
                            self.grid[new_position] != 1):  # Avoid walls
                        agent_view[new_position] = 0  # Successful move, mark as seen
                        current_position = new_position
                    else:
                        break
        return agent_view

        
    def reset(self):
        # Reset the environment to its initial state
        self.timesteps_elapsed = 0
        self._initialize_positions()
        return self.grid

    def render(self, cell_size=20):
        # Render the current state of the grid using Pygame
        if not self.pygame_initialized:
            # Initialize Pygame if not already initialized
            pygame.init()
            self.screen = pygame.display.set_mode((self.grid_width * cell_size, self.grid_height * cell_size))
            pygame.display.set_caption('VIP Game')
            self.pygame_initialized = True

        # Clear the screen
        self.screen.fill((0, 0, 0))

        # Draw the grid
        for i in range(self.grid_height):
            for j in range(self.grid_width):
                # Determine the color for each cell
                if self.grid[i,j] == 2:
                    color = (0, 255, 0)  # VIP - Green
                elif self.grid[i,j] == 3:
                    color = (0, 0, 255)  # Defender - Blue
                elif self.grid[i,j] == 4:
                    color = (255, 0, 0)  # Attacker - Red
                elif self.grid[i, j] == 1:
                    color = (0, 0, 0)  # Wall - Gray
                else:
                    color = (150, 150, 150)  # Empty space - Light gray

                # Draw the cell
                pygame.draw.rect(self.screen, color, pygame.Rect(j * cell_size, i * cell_size, cell_size, cell_size))
                pygame.draw.rect(self.screen, (0, 0, 0), pygame.Rect(j * cell_size, i * cell_size, cell_size, cell_size), 1)  # Draw borders

        # Update the display
        pygame.display.flip()

    def render_line_of_sight(self, agent_view, cell_size=20):
        if not self.pygame_initialized:
            # Initialize Pygame if not already initialized
            pygame.init()
            self.screen = pygame.display.set_mode((self.grid_width * cell_size, self.grid_height * cell_size))
            pygame.display.set_caption('VIP Game with Line of Sight')
            self.pygame_initialized = True

        running = True
        while running:
            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:  # Handle window close event
                    running = False

            # Clear the screen
            self.screen.fill((0, 0, 0))

            # Render the grid with line of sight differentiation
            for i in range(self.grid_height):
                for j in range(self.grid_width):
                    # Determine the color for each cell
                    if (i, j) == self.vip_positions:
                        color = (0, 255, 0)  # VIP - Green
                    elif (i, j) == self.defender_positions:
                        color = (0, 0, 255)  # Defender - Blue
                    elif (i, j) == self.attacker_positions:
                        color = (255, 0, 0)  # Attacker - Red
                    elif self.grid[i, j] == 1:
                        color = (0, 0, 0)  # Wall - Gray
                    else:
                        # Set color based on line of sight status
                        if agent_view[i, j] == 0:
                            color = (255, 255, 255)  # Seen tile - Light gray
                        else:
                            color = (150, 150, 150)  # Unseen tile - Dark gray

                    # Draw the cell with borders
                    pygame.draw.rect(self.screen, color, pygame.Rect(j * cell_size, i * cell_size, cell_size, cell_size))
                    pygame.draw.rect(self.screen, (0, 0, 0), pygame.Rect(j * cell_size, i * cell_size, cell_size, cell_size), 1)  # Draw borders

            # Update the display
            pygame.display.flip()

        pygame.quit()