# Generate grid for DG Method

import numpy as np
import matplotlib.pyplot as plt

def generateGrid(grid_level_lo, grid_level_hi, uniformgrid_ratio, x_left=-5, x_right=5): 
    n = np.random.randint(grid_level_lo, high=grid_level_hi + 1, size = 1)
    random_draw = np.random.random_sample()
    if random_draw < uniformgrid_ratio: 
        grid, grid_spacing = uniformGrid(n, x_left=x_left, x_right=x_right)
    else: 
        grid, grid_spacing = nonuniformGrid(n, x_left=x_left, x_right=x_right)

    return grid, grid_spacing 

def uniformGrid(n, x_left=-5, x_right=5):
    dx = (x_right-x_left)/(2**n)
    num_grid_pts = int(2**n)
    uniform_grid = np.linspace(x_left, x_right-dx, num_grid_pts)
    grid_spacing = dx * np.ones(num_grid_pts)

    return uniform_grid, grid_spacing 

def nonuniformGrid(n, x_left=-5, x_right=5):
    xi, h_vals = uniformGrid(n, x_left=x_left, x_right=x_right)

    # Generate nonuniform grid from uniform grid
    nonuniform_grid = xi + 0.25*np.sin(2*np.pi/(x_right-x_left)* (xi-x_left*np.ones((len(xi), 1))))
    nonuniform_grid = np.sort(nonuniform_grid, axis=0)
    grid_spacing = 0 * h_vals
    for i in range(len(xi)-1):
        grid_spacing[i] = nonuniform_grid[i+1] - nonuniform_grid[i]
    grid_spacing[-1] = x_right - nonuniform_grid[-1]

    return nonuniform_grid, grid_spacing 
