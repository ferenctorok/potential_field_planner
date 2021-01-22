import numpy as np
import matplotlib.pyplot as plt

from gradplanner.field_utils import Pixel, get_values_from_field
from gradplanner.utils import plot_grad_field


class PotentialField:
    """Base class for Potential Fields."""

    def __init__(self,
                 occupancy_grid=None):
        """Initializes a PotentialField"""
        self._occupancy_grid = occupancy_grid.copy() if occupancy_grid is not None else None


    def _init_field(self):
        """Initializes the potential field.
        Has to be implemented in child class.
        """
        raise(NotImplementedError)


    def update_occupancy_grid(self, occupancy_grid):
        """Updates the potential field based on a new occupancy grid.
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

