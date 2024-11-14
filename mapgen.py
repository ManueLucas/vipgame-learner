import mmaze
import numpy as np
import pygame

def generate_maze(width, height, symmetry="horizontal"):
    m = mmaze.generate(width=width, height=height, symmetry=symmetry)
    lines = m.tostring().splitlines()

    # Initialize a NumPy array of the correct size
    rows, cols = len(lines), len(lines[0])//2
    maze_array = np.zeros((rows, cols), dtype=int)

    # Fill the NumPy array based on the character mapping
    for i, line in enumerate(lines):
        for j in range(0, len(line)-1, 2):
            char_pair = line[j:j+2]
            if char_pair == '||':  # Wall character
                maze_array[i, j//2] = 1
            elif char_pair == '  ':  # Open path character
                maze_array[i, j//2] = 0

    return maze_array


# Define constants for colors
WALL_COLOR = (0, 0, 0)  # Black for walls
PASSAGE_COLOR = (255, 255, 255)  # White for passages
VIP_COLOR = (255, 0, 0)  # Red for VIP
DEFENDER_COLOR = (0, 255, 0)  # Green for Defender
ATTACKER_COLOR = (0, 0, 255)  # Blue for Attacker

def render_game_environment(grid, vip_pos, defender_pos, attacker_pos, cell_size=30):
    """
    Render the game environment using Pygame.

    :param grid: numpy array representing the game grid
    :param vip_pos: position of the VIP agent as a tuple (row, column)
    :param defender_pos: position of the defender agent as a tuple (row, column)
    :param attacker_pos: position of the attacker agent as a tuple (row, column)
    :param cell_size: size of each grid cell in pixels (default is 30)
    """
    pygame.init()

    # Get grid dimensions
    grid_height, grid_width = grid.shape

    # Set up the display
    screen_width = grid_width * cell_size
    screen_height = grid_height * cell_size
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Game Environment")

    # Run the Pygame loop
    running = True
    while running:
        screen.fill(PASSAGE_COLOR)  # Fill the screen with the background color (passages)

        # Loop through the grid to draw cells
        for row in range(grid_height):
            for col in range(grid_width):
                rect = pygame.Rect(col * cell_size, row * cell_size, cell_size, cell_size)
                if grid[row, col] == 1:  # Wall
                    pygame.draw.rect(screen, WALL_COLOR, rect)
                elif grid[row, col] == 0:  # Passage
                    pygame.draw.rect(screen, PASSAGE_COLOR, rect, 1)  # Border for passages

        # Handle events (e.g., quit the game)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Update the display
        pygame.display.flip()

    pygame.quit()

# Example usage
if __name__ == "__main__":
    # Example grid and agent positions
    grid_height = 7
    grid_width = 15
    vip_pos = (1, 1)
    defender_pos = (3, 3)
    attacker_pos = (5, 5)

    # Generate the grid
    try:
        grid = generate_maze(grid_width, grid_height)
        np.savetxt("map_presets/grid.csv", grid, delimiter=",", fmt='%d')
        # Render the grid in Pygame
        render_game_environment(grid, vip_pos, defender_pos, attacker_pos)
    except ValueError as e:
        print(e)

    
