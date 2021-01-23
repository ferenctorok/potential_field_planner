import numpy as np
import matplotlib.pyplot as plt

from gradplanner.planner.field_utils import Pixel, get_values_from_field
from gradplanner.planner.utils import plot_grad_field


class PotentialField:
    """Base class for Potential Fields."""

    def __init__(self,
                 occupancy_grid=None):
        """Initializes a PotentialField"""
        self._occupancy_grid = occupancy_grid.copy() if occupancy_grid is not None else None

        if self._occupancy_grid_is_set:
            self._grid_shape = self._occupancy_grid.shape
            self._grid_shape_arr = np.array([self._grid_shape[0], self._grid_shape[1]])
            if self._everything_is_set_for_init:
                self._init_field()


    def _init_field(self):
        """Initializes the potential field.
        Has to be implemented in child class.
        """
        raise(NotImplementedError)


    def update_occupancy_grid(self, new_grid):
        """Updates the attractor field based on a new grid.
        It creates a list of indices where there has been a change in the occupancy grid.
        self._diff_grid: 0: no change , 1: new obstacle, -1: obstacle disappeared 
        """
        if self._everything_is_set_for_init:
            assert new_grid.shape == self._grid_shape, \
                "New grid shape does not match previous grid shapes. Expected {}, recieved {}".format(self._grid_shape, new_grid.shape)
            diff_grid = new_grid - self._occupancy_grid
            self._occupancy_grid = new_grid.copy()
            if diff_grid.any():
                self._changed_indices = list(np.argwhere(diff_grid != 0))
                # updating the values of the pixels where there was a change:
                for index in self._changed_indices:
                    if self._field[index[0], index[1]].value == 1:
                        self._field[index[0], index[1]].value = 0
                    else:
                        self._field[index[0], index[1]].value = 1
                    self._field[index[0], index[1]].grad = np.array([0, 0])
                    self._field[index[0], index[1]].parent = None
                        
                self._update_field()
        else:
            self._occupancy_grid = new_grid.copy()
            self._grid_shape = self._occupancy_grid.shape
            self._grid_shape_arr = np.array([self._grid_shape[0], self._grid_shape[1]])
            if self._everything_is_set_for_init:
                self._init_field()


    def _update_field(self):
        """Updates the potential field.
        Has to be implemented in child class.
        """
        raise(NotImplementedError)


    def plot_grad(self):
        """Plots the gradient field."""
        occ_grid_to_plot = self._occupancy_grid.copy()
        plot_grad_field(self._field, occ_grid_to_plot)


    def plot_potential(self):
        """Plots the potential values of the grid"""
        values = get_values_from_field(self._field)
        M, N = self._grid_shape

        f, ax = plt.subplots(1, 1, figsize=(16, M / N * 16))
        ax.matshow(values.T)
        plt.show()


    @property
    def _occupancy_grid_is_set(self):
        """Returns whether the occupancy grid is set or not."""
        return self._occupancy_grid is not None


    @property
    def _everything_is_set_for_init(self):
        """True, if everything is set for initializing a potential field.
        Has to be set in child class.
        """
        raise(NotImplementedError)
