import numpy as np


def set_up_occ_grid(params):
    """Sets up an occupancy grid based on the provided params."""

    N = params["N"]
    M = params["M"]
    
    if params["border"]:
        grid = np.ones((N, M))
        grid[1: -1, 1: -1] = 0
    else:
        grid = np.zeros((N, M))

    # inserting the obstacles into it:
    for key in params["obstacles"]:
        i = params["obstacles"][key]["i"]
        j = params["obstacles"][key]["j"]
        grid[i[0]: i[1], j[0]: j[1]] = 1
    
    return grid
