import numpy as np
import matplotlib.pyplot as plt

from gradplanner.potential_field import PotentialField
from gradplanner.field_utils import Pixel, get_values_from_field
from gradplanner.utils import plot_grad_field


class AttractorField(PotentialField):
    """Attractor gradient field."""

    def __init__(self,
                 occupancy_grid=None,   # np.array(N, M), 1: obstacle, 0: freespace
                 goal=None              # np.array(2,): The goal position. 
                 ):
        """Initializes an AttractorField."""
        self._goal = goal.copy() if goal is not None else None
        super().__init__(occupancy_grid)

    
    def _init_field(self):
        """Initializes the attractor field based on the available occupancy_grid"""
        assert self._occupancy_grid is not None, "Empty or not provided occupancy grid."
        assert self._goal is not None, "Goal is not set."
        self._field = get_attractor_field(self._occupancy_grid, self._goal)


    def set_new_goal(self, goal):
        """Sets a new goal. If an occupancy map has already been provided, it initializes the attractive field."""
        assert goal.shape == (2,), "Expected goal shape (2,) but received {}".format(goal.shape)
        self._goal = goal.copy()
        if self._occupancy_grid is not None:
            self._init_field()


    def _update_field(self):
        """updates the attractor field if there is something new in the occupancy grid.
        It uses the list of indices of changed grid points.
        """
        # neighboring indices of the changed gridpoints sorted by value:
        expandable_indices = self._list_expandable_indices()
        
        # carry out the expansion from every expandable pixel:
        for index in expandable_indices:
            self._expand_pixel(index)
            self._update_gradient(self._field[index[0], index[1]])


    def _list_expandable_indices(self):
        """Lists all neighboring indices around changed areas. 
        It also sorts them, so that the returned list will have the indices
        of the pixel with the smalles value at the first place.
        """

        indices = []

        directions = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
        for index in self._changed_indices:
            for direction in directions:
                new_ind = index + direction
                if (new_ind >= 0).all() and (new_ind < self._grid_shape_arr).all():
                    if self._field[new_ind[0], new_ind[1]].value < 0:
                        indices.append(new_ind)

        # checking for infeasible values which would lead to local minimas:
        indices, values = self._sort_infeasible_indices(indices)

        indices, values = np.array(indices), np.array(values)

        # there are probably multiplicities in the indices, so first we sort them:
        indices, returned_inds = np.unique(indices, return_index=True, axis=0)
        values = values[returned_inds]

        # sorting it according to values.
        sorted_ind = np.argsort(-values)

        return indices[sorted_ind]


    def _sort_infeasible_indices(self, indices):
        """Looks for pixels which have infeasable small values. This can occure when a new obstacle is
        placed in the grid and hence on the other side of this there are pixels which had been accessible
        befopreviously through the place obstacle. Now these values have too small values and would lead
        to local minimas on the further side of a newly placed obstacle.
        """

        indices_out, values_out = [], []
        
        directions = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
        while indices:
            index = indices.pop()
            feasible = False
            for direction in directions:
                new_ind = index + direction
                if (new_ind >= 0).all() and (new_ind < self._grid_shape_arr).all():
                    new_val = self._field[new_ind[0], new_ind[1]].value
                    if (new_val < 0) and (new_val > self._field[index[0], index[1]].value):
                        feasible = True
                        break
            
            if feasible:
                indices_out.append(index)
                values_out.append(self._field[index[0], index[1]].value)
            else:
                # push all smaller neighbours back to the queue, since that is also a possibly infeasible point:
                for direction in directions:
                    new_ind = index + direction
                    if (new_ind >= 0).all() and (new_ind < self._grid_shape_arr).all():
                        new_val = self._field[new_ind[0], new_ind[1]].value
                        if (new_val < 0) and (new_val < self._field[index[0], index[1]].value):
                            indices.append(new_ind)

                # finally setting the value as free space as it has to be rediscovered to avoid local minimas:
                self._field[index[0], index[1]].value = 0

        # return the feasible indices and their values:
        for index in indices_out:
            values_out.append(self._field[index[0], index[1]].value)

        return indices_out, values_out


    def _expand_pixel(self, index):
        """Carries out a wavefront expansion starting from a pixel at 'index' until it has got an effect.
        Input:
            - index: np.array((2,)), the index from where to start the expansion.
        """

        queue = [index]
        search_directions = [[1, 0], [0, 1], [-1, 0], [0, -1]]

        while queue:
            ind = queue.pop(0)

            #######################
            # for debugging:
            #oldval = self._field[ind[0], ind[1]].value
            #self._field[ind[0], ind[1]].value = 10
            #self.plot_potential()
            #self._field[ind[0], ind[1]].value = oldval
            ######################

            pix = self._field[ind[0], ind[1]]
            # iterate over the neighboring pixels:
            for direction in search_directions:
                new_ind = ind + direction
                if (new_ind >= 0).all() and (new_ind < self._grid_shape_arr).all():
                    new_pix = self._field[new_ind[0], new_ind[1]]
                    # if the new_pixel has smaller value or is free space:
                    if (new_pix.value < pix.value) or (new_pix.value == 0):
                        value_orig = new_pix.value
                        new_pix = self._update_pixel(new_pix)
                        if value_orig != new_pix.value:
                            queue.append(new_ind)


    def _set_new_pixel(self, pix, new_pix):
        """Sets up a new pixels value and gradient."""

        # setting the value of the pixel:
        new_pix.value = pix.value - 1
        # setting the gradient of the pixel:
        return self._update_gradient(new_pix)


    def _update_pixel(self, new_pix):
        """Updates a pixels value and gradient."""
        
        # setting the value of the pixel:
        ind = np.array([new_pix.x, new_pix.y])
        search_directions = [[1, 0], [0, 1], [-1, 0], [0, -1]]
        for direction in search_directions:
            old_ind = ind + direction
            if (old_ind >= 0).all() and (old_ind < self._grid_shape_arr).all() and \
                (self._field[old_ind[0], old_ind[1]].value < 0):
                old_pix = self._field[old_ind[0], old_ind[1]]
                if (new_pix.value < old_pix.value - 1) or (new_pix.value == 0):
                    new_pix.value = old_pix.value - 1

        # updating the gradient of the pixel:
        return self._update_gradient(new_pix)


    def _update_gradient(self, new_pix):
        """Updates the gradient of the pixel."""

        new_pix.grad = np.array([0, 0])
        
        # summing up the gradients which point to points which are closer to the goal:
        ind = np.array([new_pix.x, new_pix.y])
        search_directions = [[1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]]
        
        for direction in search_directions:
            old_ind = ind + direction
            if (old_ind >= 0).all() and (old_ind < self._grid_shape_arr).all():
                old_pix = self._field[old_ind[0], old_ind[1]]
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
        neighbour_is_occupied = (self._field[int(neighbour_indices[0]), int(neighbour_indices[1])].value == 1)
        
        if grad_is_zero or neighbour_is_occupied:
            for direction in search_directions:
                old_ind = ind + direction
                if (old_ind >= 0).all() and (old_ind < self._grid_shape_arr).all():
                    old_pix = self._field[old_ind[0], old_ind[1]]
                    if (old_pix.value < 0) and (old_pix.value > new_pix.value):
                        new_pix.grad = np.array(direction)
                        break
            new_pix.normalize_grad()

        return new_pix


    def plot_grad(self):
        """plots the gradient field."""
        occ_grid_to_plot = self._occupancy_grid.copy()
        goal_floor = np.floor(self._goal)
        goal_i, goal_j = int(goal_floor[0]), int(goal_floor[1])
        occ_grid_to_plot[goal_i, goal_j] = -1

        plot_grad_field(self._field, occ_grid_to_plot)

    
    @property
    def _goal_is_set(self):
        """Returns whether the goal is set or not."""
        return self._goal is not None

    
    @property
    def _everything_is_set_for_init(self):
        """True, if everything is set for initializing a potential field."""
        return self._occupancy_grid_is_set and self._goal_is_set


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
    goal_i, goal_j = int(goal_floor[0]), int(goal_floor[1])
    assert (goal_floor >= 0).all() and (goal_floor < occ_shape).all(), "Goal is out of map." 
    assert (occupancy_grid[goal_i, goal_j] == 0), "Goal is not in free space."
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
                # if the pixel is free space, calculate the value and the gradient of the new pixel:
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