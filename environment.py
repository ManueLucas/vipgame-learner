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
    def __init__(self, grid_map, max_timesteps=500):
        # Initialize the grid, dimensions, max timesteps, and elapsed timesteps
        self.defenderside_collision_set = []
        self.original_grid = np.copy(grid_map)
        self.grid = np.copy(grid_map)
        self.grid_height, self.grid_width = self.grid.shape
        self.max_timesteps = max_timesteps
        self.timesteps_elapsed = 0
        
        #we keep separate and redundant counts for the sake of statistics
        self.number_of_vip_dead = 0
        self.number_of_defender_dead = 0
        self.number_of_attacker_dead = 0
        self.number_of_vips = 0
        self.number_of_attackers = 0
        self.number_of_defenders = 0
        
        #which agents are still alive?
        self.live_vips = []
        self.live_attackers = []
        self.live_defenders = []
        
        self.attacker_defender_action_space = 8
        self.vip_action_space = 4
        
        self.attackerside_collision_set = [WALL, ATTACKER]
        self.defenderside_collision_set = [WALL, DEFENDER, VIP]
        self.vipside_collision_set = [WALL, VIP, DEFENDER, ATTACKER]
        
        self.attacker_kill_set = [DEFENDER, VIP]
        self.defender_kill_set = [ATTACKER]
        
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
        self.dead_cell = (-100, -100)
        
        # Pygame initialization flag
        self.pygame_initialized = False
        self.screen = None
        
        # Find and set the initial positions of agents
        self.reset()


    def _move_agent(self, position, action, agent_type, agent_id):
        # do nothing if the agent is dead
        if position == self.dead_cell:
            return position, 0
        
        moveset = self.get_moveset(agent_type)
        killset = self.get_killset(agent_type)
        collisionset = self.get_collisionset(agent_type)
        # If the action is out of bounds of the moveset, return negative reward
        if action >= len(moveset):
            return position, -1  # Invalid action, negative reward

        # Calculate the new position based on the action taken
        delta = moveset[action]
        new_position = (position[0] + delta[0], position[1] + delta[1])

        # Collision check to ensure the agent stays within grid bounds and avoids walls
        if (0 <= new_position[0] < self.grid_height and 0 <= new_position[1] < self.grid_width):  # disallow collision with same team and walls
            
            if (agent_type == ATTACKER and self.grid[new_position] == VIP):
                print("vip killed")
                self.grid[new_position] = self.grid[position]
                self.grid[position] = 0
                # Move VIP to the death cell
                vip_index = self.vip_positions.index(new_position)
                self.vip_positions[vip_index] = self.dead_cell
                self.number_of_vip_dead += 1
                return new_position, 10
            
            # when a defender and an attacker meet each other, both die
            if (self.grid[new_position] in killset):
                # remove them from the grid render
                self.grid[position] = 0
                self.grid[new_position] = 0
                self.number_of_attacker_dead += 1
                self.number_of_defender_dead += 1
                print("Attacker and Defender killed each other")
                if agent_type == ATTACKER:
                    # Move defender to the death cell
                    defender_index = self.defender_positions.index(new_position)
                    self.live_defenders[defender_index] = False
                    self.live_attackers[agent_id] = False
                    self.defender_positions[defender_index] = self.dead_cell
                    # Move attacker to the death cell
                    return self.dead_cell, 3  # high reward for killing defender

                elif agent_type == DEFENDER:
                    # Move attacker to the death cell
                    attacker_index = self.attacker_positions.index(new_position)
                    self.live_defenders[agent_id] = False
                    self.live_attackers[attacker_index] = False
                    self.attacker_positions[attacker_index] = self.dead_cell
                    # Move defender to the death cell
                    return self.dead_cell, 3 # high reward for killing attacker

                return new_position, 1
            
            if (self.grid[new_position] not in collisionset):
                self.grid[new_position] = self.grid[position]
                self.grid[position] = 0
                return new_position, 0  # Successful move, neutral reward

        return position, -1  # Collision or out of bounds, negative reward

    def attacker_move(self, action, agent_id):
        # Update attacker position and get reward for the move
        self.attacker_positions[agent_id], reward = self._move_agent(self.attacker_positions[agent_id], action, ATTACKER, agent_id)
        return reward

    def defender_move(self, action, agent_id): # we can merge the attacker and defender move functions into one later
        # Update defender position and get reward for the move
        self.defender_positions[agent_id], reward = self._move_agent(self.defender_positions[agent_id], action, DEFENDER, agent_id)
        return reward

    def vip_move(self, action, agent_id):
        # Update VIP position and get reward for the move
        self.vip_positions[agent_id], reward = self._move_agent(self.vip_positions[agent_id], action, VIP, agent_id)
        return reward

    def step(self, actions): #we expect a tuple of 3 lists of actions, one for each team
        # Perform actions for each agent (attacker, defender, VIP)
        attacker_actions, defender_actions, vip_actions = actions
        
        attacker_reward = []
        defender_reward = []
        vip_reward = []
        # Execute each agent's move and get their respective rewards
        for i, attacker_action in enumerate(attacker_actions):
            if self.live_attackers[i]: # only allow the agent to move if they are alive
                attacker_reward.append(self.attacker_move(attacker_action, i))
            else:
                attacker_reward.append(0)
            
        for i, defender_action in enumerate(defender_actions):
            if self.live_defenders[i]:
                defender_reward.append(self.defender_move(defender_action, i))
            else:
                defender_reward.append(0)
            
        for i, vip_action in enumerate(vip_actions):  
            if self.live_vips[i]:
                vip_reward.append(self.vip_move(vip_action, i))
            else:
                vip_reward.append(0)
        # Increment the timestep counter
        self.timesteps_elapsed += 1
        # Check if the maximum number of timesteps has been reached
        truncated = self.timesteps_elapsed >= self.max_timesteps
        terminated = False
        # Check if the game is over (either the VIP is dead or all attackers are dead)
        if self.number_of_vip_dead == self.number_of_vips:
            terminated = True
            attacker_reward = [x + 3 for x in attacker_reward] #give entire team a huge reward
        elif self.number_of_attacker_dead == self.number_of_attackers:
            terminated = True
            defender_reward = [x + 3 for x in defender_reward]
            vip_reward = [x + 3 for x in vip_reward] 
        
        defenderside_vision = self.line_of_sight(self.defender_positions + self.vip_positions)
        attackerside_vision = self.line_of_sight(self.attacker_positions)
        return self.grid, (defenderside_vision, attackerside_vision), (defender_reward, attacker_reward, vip_reward), (self.defender_positions, self.attacker_positions, self.vip_positions), truncated, terminated
    
    def line_of_sight(self, agent_positions):
        #TODO: take multiple agent positions, reveal the map in their lines of sight and return the grid according to their team's lines of sight. -1: unseen tile, 0: seen tile,
        agent_view = np.where(self.grid != 1, -1, self.grid)
        

        for position in agent_positions:
            if position == self.dead_cell:
                continue # do nothing if the agent is dead
            agent_view[position] = self.grid[position]  # reveal where they are standing
            for delta in ACTIONS.values():
                current_position = position
                
                while True:
                    # new position in chosen direction
                    new_position = (current_position[0] + delta[0], current_position[1] + delta[1])

                    # Collision check to ensure vision stays within grid bounds and avoids walls
                    if (0 <= new_position[0] < self.grid_height and
                            0 <= new_position[1] < self.grid_width and
                            self.grid[new_position] != 1):  # Avoid walls
                        agent_view[new_position] = self.grid[new_position]  # Successful move, mark as seen
                        current_position = new_position
                    else:
                        break
        return agent_view
    
    def get_killset(self, agent_type):
        if agent_type == ATTACKER:
            return self.attacker_kill_set
        elif agent_type == DEFENDER:
            return self.defender_kill_set
        else:
            return []
        
    def get_collisionset(self, agent_type):
        if agent_type == ATTACKER:
            return self.attackerside_collision_set
        elif agent_type == DEFENDER:
            return self.defenderside_collision_set
        else:
            return self.vipside_collision_set

    def get_moveset(self, agent_type):
        # Defender has access to the full moveset (all 8 directions)
        # Attacker has access to the full moveset (all 8 directions)
        if agent_type == ATTACKER or agent_type == DEFENDER:
            return list(ACTIONS.values())
        # VIP has a limited moveset (up, down, left, right)
        else:
            return [ACTIONS['UP'], ACTIONS['DOWN'], ACTIONS['LEFT'], ACTIONS['RIGHT']]

    def reset(self):
        # Reset the environment to its initial state
        self.timesteps_elapsed = 0
        
        self.number_of_vip_dead = 0
        self.number_of_attacker_dead = 0
        self.number_of_defender_dead = 0
        
        self.number_of_attackers = 0
        self.number_of_defenders = 0
        self.number_of_vips = 0
        
        self.attacker_positions = []
        self.defender_positions = []
        self.vip_positions = []
        
        self.grid = np.copy(self.original_grid)
        # Loop through the grid to find initial positions of agents
        for i in range(self.grid_height):
            for j in range(self.grid_width):
                if self.grid[i, j] == VIP:
                    self.vip_positions.append((i, j))  # VIP position
                    self.number_of_vips += 1
                elif self.grid[i, j] == DEFENDER:
                    self.defender_positions.append((i, j))  # Defender position
                    self.number_of_defenders += 1
                elif self.grid[i, j] == ATTACKER:
                    self.attacker_positions.append((i, j))  # Attacker position
                    self.number_of_attackers += 1
        self.live_vips = [True] * self.number_of_vips
        self.live_attackers = [True] * self.number_of_attackers
        self.live_defenders = [True] * self.number_of_defenders
        return self.grid
            

    def render(self, grid, cell_size=20):
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
                if grid[i,j] == VIP:
                    color = (0, 255, 0)  # VIP - Green
                elif grid[i,j] == DEFENDER:
                    color = (0, 0, 255)  # Defender - Blue
                elif grid[i,j] == ATTACKER:
                    color = (255, 0, 0)  # Attacker - Red
                elif grid[i, j] == WALL:
                    color = (0, 0, 0)  # Wall - Gray
                elif grid[i, j] == SELF:
                    color = (255, 255, 0)
                elif grid[i, j] == UNSEEN:
                    color = (150, 150, 150)  # Empty space - Light gray
                else:
                    color = (255, 255, 255)

                # Draw the cell
                pygame.draw.rect(self.screen, color, pygame.Rect(j * cell_size, i * cell_size, cell_size, cell_size))
                pygame.draw.rect(self.screen, (0, 0, 0), pygame.Rect(j * cell_size, i * cell_size, cell_size, cell_size), 1)  # Draw borders

        # Update the display
        pygame.display.flip()
        
    def render_line_of_sight(self, agent_view, cell_size=20):
        pygame.init()
        los_screen = pygame.display.set_mode((self.grid_width * cell_size, self.grid_height * cell_size))
        pygame.display.set_caption('VIP Game with Line of Sight')

        running = True
        while running:
            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:  # Handle window close event
                    running = False

            # Clear the screen
            los_screen.fill((0, 0, 0))

            # Render the grid with line of sight differentiation
            for i in range(self.grid_height):
                for j in range(self.grid_width):
                    # Determine the color for each cell
                    if agent_view[i, j] == VIP:
                        color = (0, 255, 0)  # VIP - Green
                    elif agent_view[i, j] == DEFENDER:
                        color = (0, 0, 255)  # Defender - Blue
                    elif agent_view[i, j] == ATTACKER:
                        color = (255, 0, 0)  # Attacker - Red
                    elif agent_view[i, j] == WALL:
                        color = (0, 0, 0)  # Wall - Gray
                    elif agent_view[i, j] == SELF:
                        color = (255, 255, 0)  # Self - Yellow
                    elif agent_view[i, j] == OPEN:
                        color = (255, 255, 255)  # Empty space - Light gray
                    else:
                        color = (150, 150, 150)  # Unseen tile - Dark gray

                    # Draw the cell with borders
                    pygame.draw.rect(los_screen, color, pygame.Rect(j * cell_size, i * cell_size, cell_size, cell_size))
                    pygame.draw.rect(los_screen, (0, 0, 0), pygame.Rect(j * cell_size, i * cell_size, cell_size, cell_size), 1)  # Draw borders

            # Update the display
            pygame.display.flip()

        pygame.quit()