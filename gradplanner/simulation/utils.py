import numpy as np
import matplotlib.pyplot as plt
import json


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


if __name__ == "__main__":

    param_file = "src/params/sim_1.json"
    with open(param_file) as f:
        params = json.load(f)
    
    occ_grid = set_up_occ_grid(params)

    f, ax = plt.subplots(1, 1, figsize=(16, 16))
    ax.matshow(occ_grid)
    plt.show()