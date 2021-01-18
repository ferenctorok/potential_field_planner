import numpy as np
from gradplanner.field_utils import Pixel


class AttractorField():
    """Attractor gradient field."""

    def __init__():
        """Initializes an AttractorField."""

        pass


def get_attractor_field(occupancy_grid):
    """
    Wavefront planner started from the goal.
    Input:
        - occupancy_grid: np.array(N, M)
    Output:
        - attractor_field: np.array(N, M), The repulsive field that was produced
    """
    
    N, M = occupancy_grid.shape
    occ_shape = np.array([N, M])
    # queue for carrying out the expansion around the obstacles:
    queue = []
    # np.array of pixels:
    attractor_field = np.ndarray(occupancy_grid.shape, dtype=np.object)
    for i in range(occupancy_grid.shape[0]):
        for j in range(occupancy_grid.shape[1]):
            attractor_field[i, j] = Pixel(i, j)
            if occupancy_grid[i, j] == 1:
                attractor_field[i, j].value = 1
            elif occupancy_grid[i, j] == -1:
                attractor_field[i, j].value = -1
                queue.append(np.array([i, j]))
                
    # carrying out the expansion while the queue is not empty:
    #search_directions = [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]]
    search_directions = [[1, 0], [0, 1], [-1, 0], [0, -1]]
    while queue:
        ind = queue.pop(0)
        pix = attractor_field[ind[0], ind[1]]
        # iterate over the neighboring pixels:
        for direction in search_directions:
            new_ind = ind + direction
            if (new_ind >= 0).all() and (new_ind < occ_shape).all():
                new_pix = attractor_field[new_ind[0], new_ind[1]]
                # if the goal is free space or goal, calculate the value and the gradient of the new pixel:
                if (new_pix.value == 0):
                    set_new_pixel_attr(pix, new_pix, attractor_field, occ_shape)
                    queue.append(new_ind)
        
    
    return attractor_field


def set_new_pixel_attr(pix, new_pix, attractor_field, occ_shape):
    """Sets up a new pixels value and gradient for the attractor field."""
    
    # setting the value of the pixel:
    new_pix.value = pix.value - 1
    new_pix.grad = np.array([0, 0])
    
    # summing up the gradients which point to points which are closer to the goal:
    ind = np.array([new_pix.x, new_pix.y])
    search_directions = [[1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]]
    
    for direction in search_directions:
        old_ind = ind + direction
        if (old_ind >= 0).all() and (old_ind < occ_shape).all():
            old_pix = attractor_field[old_ind[0], old_ind[1]]
            if (old_pix.value < 0) and (old_pix.value > new_pix.value):
                new_pix.grad += np.array(direction)
                
    new_pix.normalize_grad()
                
    # if the sum is accidentaly zero, or if it points to an obstacle: 
    # set grad to point to the first direction that it finds feasible:
    # The search_directions list is structured so, that pixels which are touching this
    # pixel with a side are taken first.
    grad_is_zero = (new_pix.grad[0] == 0) and (new_pix.grad[1] == 0)
    # check if neighbour is opccupied:
    neighbour_indices = np.floor(ind + np.array([0.5, 0.5]) + new_pix.grad)
    neighbour_is_occupied = (attractor_field[int(neighbour_indices[0]), int(neighbour_indices[1])].value == 1)
    
    if grad_is_zero or neighbour_is_occupied:
        for direction in search_directions:
            old_ind = ind + direction
            if (old_ind >= 0).all() and (old_ind < occ_shape).all():
                old_pix = attractor_field[old_ind[0], old_ind[1]]
                if (old_pix.value < 0) and (old_pix.value > new_pix.value):
                    new_pix.grad = np.array(direction)
                    break
        new_pix.normalize_grad()