import environment
import numpy as np
if(__name__ == "__main__"):
    grid_map = np.loadtxt("map_presets/grid.csv", delimiter=",")
    env = environment.VipGame(grid_map=grid_map)
    env.render()
    env.reset()