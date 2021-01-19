import numpy as np
from gradplanner.field_utils import Pixel


class AttractorField():
    """Attractor gradient field."""

    def __init__(self,
                 occupancy_grid=None,   # np.array(N, M), 1: obstacle, 0: freespace
                 goal=None              # np.array(2,): The goal position. 
                 ):
        """Initializes an AttractorField."""
        self._occupancy_grid = occupancy_grid
        self._goal = goal

        if self._occupancy_grid is not None:
            self._grid_shape = self._occupancy_grid.shape
            self._grid_shape_arr = np.array([self._grid_shape[0], self._grid_shape[1]])
            if self._goal is not None:
                self._init_field()

    
    def _init_field(self):
        """Initializes the attractor field based on the available occupancy_grid"""
        assert self._occupancy_grid is not None, "Empty or not provided occupancy grid."
        assert self._goal is not None, "Goal is not set."
        self._field = get_attractor_field(self._occupancy_grid, self._goal)


    def set_new_goal(self, goal):
        """Sets a new goal. If an occupancy map has already been provided, it initializes the attractive field."""
        assert goal.shape == (2,), "Expected goal shape (2,) but received {}".format(goal.shape)
        self._goal = goal
        if self._occupancy_grid is not None:
            self._init_field()


    def update_occupancy_grid(self, new_grid):
        """Updates the occupancy grid based on a new grid.
        It creates a list of indices where there has been a change in the occupancy grid.
        self._diff_grid: 0: no change , 1: new obstacle, -1: obstacle disappeared 
        """
        if self._occupancy_grid is not None:
            diff_grid = new_grid - self._occupancy_grid
            self._occupancy_grid = new_grid.copy()
            if diff_grid.any():
                self._changed_indices = list(np.argwhere(diff_grid != 0))
                self._update_field
        else:
            self._occupancy_grid = new_grid.copy()
            self._grid_shape = self._occupancy_grid.shape
            self._grid_shape_arr = np.array([self._grid_shape[0], self._grid_shape[1]])


    def _update_field(self):
        """updates the attractor field if there is something new in the occupancy grid.
        It uses the list of indices of changed grid points.
        """
        
        expandable_indices = self._list_expandable_indices()


    def _list_expandable_indices(self):
        """Lists all neighboring indices around changed areas. 
        It also sorts them, so that the returned list will have the indices
        of the pixel with the smalles value at the first place.
        """

        values = np.array([])
        indices = np.array([])

        directions = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
        for index in self._changed_indices:
            for direction in directions:
                new_ind = index + direction
                if (new_ind >= 0).all() and (new_ind < self._grid_shape_arr).all():
                    if not self._occupancy_grid[new_ind[0], new_ind[1]]:
                        indices.append(new_ind)
                        values.append(self._field[new_ind[0], new_ind[1]].value)
        
        # there are probably multiplicities in the indices, so first we sort them:
        indices, returned_inds = np.unique(indices, return_index=True, axis=0)
        values = values[returned_inds]

        # sorting it according to values.
        sorted_ind = np.argsort(values)

        return indices[sorted_ind]
            



        


def get_attractor_field(occupancy_grid, goal):
    """
    Wavefront planner started from the goal.
    Input:
        - occupancy_grid: np.array(N, M)
        - goal: np.array(2,), goal position
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
            attractor_field[i, j] = Pixel(i, j, occupancy_grid[i, j])

    # set the goal position pixel to -1 and add its index to the queue.
    goal_floor = np.floor(goal)
    assert (goal_floor >= 0).all() and (goal_floor < occ_shape).all(), "Goal is out of map." 
    goal_i, goal_j = int(goal_floor[0]), int(goal_floor[1])
    attractor_field[goal_i, goal_j].value = -1
    queue.append(np.array([goal_i, goal_j]))
                   
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