import numpy as np
import matplotlib.pyplot as plt


def plot_grad_field(field, occupancy_grid):
    """Plots the gradient field of the np.array of pixels given by field"""
    eps = 1e-6
    N, M = field.shape
    
    # matrix for the x and y coordinates in every point:
    x, y = np.zeros((N, M)), np.zeros((N, M))  
    for i in range(N):
        for j in range(M):
            x[i, j] = field[i, j].grad[0] + eps
            y[i, j] = field[i, j].grad[1] + eps
    
    # plotting:
    plt.figure(figsize=(16, M / N * 16))
    f, ax = plt.subplots(1, 1, figsize=(16, M / N * 16))
    ax.quiver(x.T, -y.T, scale=1, scale_units='xy')
    ax.matshow(occupancy_grid.T)
    plt.show()