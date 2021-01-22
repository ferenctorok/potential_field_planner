import numpy as np

from gradplanner.field_utils import Pixel
from gradplanner.potential_field import PotentialField


class RepulsiveField(PotentialField):
    """Class for the repulsive gradient field."""

    def __init__(self,
                 occupancy_grid=None,
                 R=5
                 ):
        """Initialize the Repulsive field object."""
        assert isinstance(R, int), "The provided R is not of type int."
        self._R = R
        super().__init__(occupancy_grid)

        if self._occupancy_grid is not None:
            self._grid_shape = self._occupancy_grid.shape
            self._grid_shape_arr = np.array([self._grid_shape[0], self._grid_shape[1]])
            self._init_field()


    def _init_field(self):
        """Initializes the repulsive field based on the available occupancy_grid"""
        assert self._occupancy_grid is not None, "Empty or not provided occupancy grid."
        self._field = get_repulsive_field(self._occupancy_grid, self._R)


    def _update_field(self):
        """updates the attractor field if there is something new in the occupancy grid.
        It uses the list of indices of changed grid points.
        """
        
        # indices from which an expansion has to be carried out.
        expandable_indices = self._list_expandable_indices()


    def _list_expandable_indices(self):
        """Lists the indices from which updating expansion should be carried out.
        It also zeros out pixels to which the neares obstacle was previously one
        that disappeared.
        """
        
        indices = []
        
        for index in self._changed_indices:
            if self._field[index[i], index[j]].value == 1:
                indices.append(index)
            else:
                for ind in self._get_first_not_influenced_pixels(index):
                    indices.append(ind)

        return indices


    def _get_first_not_influenced_pixels(self, index):
        """Sets all pixels to zero, which were influenced by the obstacle that disappeared
        and returns the indices (if any) which are neighboring them and by further expanding them,
        they will influence the reset pixels.
        """

        indices_out = []

        queue = [index]
        parent = (index[0], index[1])
        search_directions = [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]]

        while queue:
            ind = queue.pop(0)
            pix = self._field[ind[0], ind[1]]
            # iterate over the neighboring pixels:
            for direction in search_directions:
                new_ind = ind + direction
                if (new_ind >= 0).all() and (new_ind < self._grid_shape_arr).all():
                    new_pix = self._field[new_ind[0], new_ind[1]]
                    # if the new_pix's parent was the disappeared obstacle:
                    if new_pix.parent == parent:
                        # searching among the surrounding pixels for a possible new pixel to expand.
                        to_expand = self._search_surrounding_for_expandable(new_pix)
                        if to_expand is not None:
                            indices_out.append(to_expand)

                        # reseting the pixel:
                        new_pix.value = 0
                        new_pix.parent = None
                        
                        queue.append(new_ind)

        return list(np.unique(np.array(indices_out), axis=0))                
    

    def _search_surrounding_for_expandable(self, pix):
        """Searches around a pixel for other pixels which are influenced by another obstacle
        and can be further expanded. It returns the one of them, which has got the smallest value.
        """

        ind_out = None
        # R + 1 is the max value a cell can have. It is ok to set val_out to 6 and only look for cells
        # with smaller values, because a value 6 pixel could not be further expanded.
        val_out = self._R + 1

        ind = np.array([pix.x, pix.y])
        search_directions = [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]]

        for direction in search_directions:
            new_ind = ind + direction
            if (new_ind >= 0).all() and (new_ind < self._grid_shape_arr).all():
                new_pix = self._field[new_ind[0], new_ind[1]]
                if (0 < new_pix.value < val_out) and (new_pix.parent != pix.parent):
                    ind_out = new_ind
                    val_out = new_pix.value

        return ind_out


    @property
    def _everything_is_set_for_init(self):
        """True, if everything is set for initializing a potential field."""
        return self._occupancy_grid_is_set


def get_repulsive_field(occupancy_grid, R):
    """
    Function to obtain the repulsive potential field based on an occupancy grid.
    Input:
        - occupancy_grid: np.array(N, M), the occupancy grid
        - R: scalar, max distrance to propagate from any obstacle
    Output:
        - rep_field: np.array(N, M) of pixels: The repulsive field that was produced
    """
    N, M = occupancy_grid.shape
    occ_shape = np.array([N, M])
    # queue for carrying out the expansion around the obstacles:
    queue = []
    # np.array of pixels:
    rep_field = np.ndarray(occupancy_grid.shape, dtype=np.object)
    for i in range(occupancy_grid.shape[0]):
        for j in range(occupancy_grid.shape[1]):
            rep_field[i, j] = Pixel(i, j)
            if occupancy_grid[i, j] == 1:
                rep_field[i, j].value = 1
                # adding the indices of obstacles to the queue:
                queue.append(np.array([i, j]))
    
    
    # carrying out the expansion while the queue is not empty:
    search_directions = [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]]
    while queue:
        ind = queue.pop(0)
        pix = rep_field[ind[0], ind[1]]
        # iterate over the neighboring pixels:
        for direction in search_directions:
            new_ind = ind + direction
            if (new_ind >= 0).all() and (new_ind < occ_shape).all():
                new_pix = rep_field[new_ind[0], new_ind[1]]
                # if the goal is free space or goal, calculate the value and the gradient of the new pixel:
                if (new_pix.value == 0):
                    set_new_pixel_rep(pix, new_pix, rep_field, occ_shape)
                    scale_gradient_rep(new_pix, R)
                    # at a distance of R from a boundary the gradient should be already zero, so it is 
                    # not necessary to further expand a node, which's child will already be over R.
                    # new_pix - 1 is needed since the value of an obstacle is already 1
                    if (new_pix.value - 1) < R - 1:
                        queue.append(new_ind)
    
    return rep_field


def set_new_pixel_rep(pix, new_pix, rep_field, occ_shape):
    """Sets up the value and the gradient of a new pixel for the repulsive field."""
    
    # setting the value of the pixel:
    new_pix.value = pix.value + 1
    new_pix.parent = (pix.x, pix.y)
    new_pix.grad = np.array([0, 0])
    
    # the gradient of the pixel is the sum of directions which point from neigboring pixels
    # with smaller values to this point. The summed vector is normalized at the end.
    ind = np.array([new_pix.x, new_pix.y])
    search_directions = [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]]
    
    for direction in search_directions:
        old_ind = ind + direction
        if (old_ind >= 0).all() and (old_ind < occ_shape).all():
            old_pix = rep_field[old_ind[0], old_ind[1]]
            if (old_pix.value != 0) and (old_pix.value < new_pix.value):
                new_pix.grad -= direction
    
    new_pix.normalize_grad()
    
    
def scale_gradient_rep(new_pix, R):
    """
    Scales the gradient of the pixel. The repulsive gradients have lenght 1 at
    an obstacle boarder and and 0 if they are further away from the boarder then R. 
    Between them the length varies linearly.
    """
    new_pix.grad = new_pix.grad * (1 - (new_pix.value - 2) / R)