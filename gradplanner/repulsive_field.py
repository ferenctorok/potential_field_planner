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

        # expanding the expandable indices:
        self._expand_pixels(expandable_indices)

        # refresh the gradients of the expandable indices, because they might have been
        # influenced by the changes:
        for index in expandable_indices:
            pix = self._field[index[0], index[1]]
            set_gradient(pix, self._field, self._grid_shape_arr)
            scale_gradient_rep(pix, self._R)            


    def _list_expandable_indices(self):
        """Lists the indices from which updating expansion should be carried out.
        It also zeros out pixels to which the neares obstacle was previously one
        that disappeared.
        """
        
        indices, values = [], []
        
        for index in self._changed_indices:
            if self._field[index[0], index[1]].value == 1:
                indices.append(index)
                values.append(1)
            else:
                new_indices, new_values = self._get_first_not_influenced_pixels(index)
                indices += new_indices
                values += new_values

        # aranging the indices in a growing order according to their corresponding values:
        indices, values = np.array(indices), np.array(values)
        sorted_ind = np.argsort(values)
        return list(indices[sorted_ind])


    def _get_first_not_influenced_pixels(self, index):
        """Sets all pixels to zero, which were influenced by the obstacle that disappeared
        and returns the indices (if any) which are neighboring them and by further expanding them,
        they will influence the reset pixels.
        """

        indices_out, values_out = [], []

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
                        to_expand_list = self._search_surrounding_for_expandable(new_pix)
                        for to_expand in to_expand_list: 
                            indices_out.append(to_expand)
                            values_out.append(self._field[to_expand[0], to_expand[1]].value)

                        # reseting the pixel:
                        new_pix.value = 0
                        new_pix.parent = None
                        new_pix.grad = np.array([0, 0])
                        
                        queue.append(new_ind)

        # returning only the unique indices:
        if indices_out != []:
            indices_out, values_out = np.array(indices_out), np.array(values_out)
            indices_out, inds = np.unique(indices_out, return_index=True, axis=0)
            values_out = values_out[inds]

        return list(indices_out), list(values_out) 
    

    def _search_surrounding_for_expandable(self, pix):
        """Searches around a pixel for other pixels which are influenced by another obstacle
        and can be further expanded.
        """

        ind_out = []
        ind = np.array([pix.x, pix.y])
        search_directions = [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]]

        for direction in search_directions:
            new_ind = ind + direction
            if (new_ind >= 0).all() and (new_ind < self._grid_shape_arr).all():
                new_pix = self._field[new_ind[0], new_ind[1]]
                if (0 < new_pix.value < self._R + 1) and (new_pix.parent != pix.parent):
                    ind_out.append(new_ind)

        return ind_out


    def _expand_pixels(self, indices):
        """Expands a pixel until R."""

        queue = indices.copy()
        search_directions = [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]]

        while queue:
            ind = queue.pop(0)

            #######################
            # for debugging: It alwas plots, which pixel is popped
            #oldval = self._field[ind[0], ind[1]].value
            #self._field[ind[0], ind[1]].value = 10
            #self.plot_potential()
            #self._field[ind[0], ind[1]].value = oldval
            ######################

            pix = self._field[ind[0], ind[1]]
            # it is possible, that since then the obstacle of this pixel has been deleted.
            # Then it has got a value of 0 and then the expansion does not have to be carried out:
            if pix.value == 0:
                continue
            # iterate over the neighboring pixels:
            for direction in search_directions:
                new_ind = ind + direction
                if (new_ind >= 0).all() and (new_ind < self._grid_shape_arr).all():
                    new_pix = self._field[new_ind[0], new_ind[1]]
                    # if the new_pix is free space, calculate the value and the gradient of it:
                    if (new_pix.value == 0) or (new_pix.value > pix.value):
                        val_orig = new_pix.value 
                        set_new_pixel_rep(pix, new_pix, self._field, self._grid_shape_arr)
                        scale_gradient_rep(new_pix, self._R)
                        # at a distance of R from a boundary the gradient should be already zero, so it is 
                        # not necessary to further expand a node, which's child will already be over R.
                        # new_pix - 1 is needed since the value of an obstacle is already 1
                        if ((new_pix.value - 1) < self._R - 1) and\
                            (new_pix.value != val_orig):
                            queue.append(new_ind)


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
                # if the new_pix is free space, calculate the value and the gradient of it:
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
    
    # setting the parent of the pixel:
    if pix.parent is not None:
        new_pix.parent = pix.parent
    else:
        new_pix.parent = (pix.x, pix.y)

    # setting the gradient:
    set_gradient(new_pix, rep_field, occ_shape)
    
    
def scale_gradient_rep(new_pix, R):
    """
    Scales the gradient of the pixel. The repulsive gradients have lenght 1 at
    an obstacle boarder and and 0 if they are further away from the boarder then R. 
    Between them the length varies linearly.
    """
    new_pix.grad = new_pix.grad * (1 - (new_pix.value - 2) / R)


def is_special_case(new_pix, field, occ_shape):
    """There is a special case, where we would like to return grad = np.array([0, 0])
    no matter, how the other pixels are. This is the case, when on 2 opposite sides
    of the pixel are obstacles. This of course can only happen, if the pixel is
    a neighbour of obstacles, so if its value is 2.
    """

    if new_pix.value == 2:
        ind = np.array([new_pix.x, new_pix.y])
        ind1, ind2 = ind + [1, 0], ind - [1, 0]
        ind3, ind4 = ind + [0, 1], ind - [0, 1]

        if ((ind1 >= 0).all() and (ind1 < occ_shape).all()) and\
           ((ind2 >= 0).all() and (ind2 < occ_shape).all()) and\
           (field[ind1[0], ind1[1]].value == 1) and\
           (field[ind2[0], ind2[1]].value == 1):
           return True
        elif ((ind3 >= 0).all() and (ind3 < occ_shape).all()) and\
             ((ind4 >= 0).all() and (ind4 < occ_shape).all()) and\
             (field[ind3[0], ind3[1]].value == 1) and\
             (field[ind4[0], ind4[1]].value == 1):
            return True

    return False


def set_gradient(new_pix, rep_field, occ_shape):
    """Sets the gradient of a pixel."""

    # zeroing out the gradient initially
    new_pix.grad = np.array([0, 0])

    # there is a special case, where we need grad = np.array([0, 0]). It is detailed in
    # the function is_special_case():
    if is_special_case(new_pix, rep_field, occ_shape):
        return
    
    # If the pixel has got a value 0 or 1, it has to have a gradient of 0:
    if new_pix.value <= 1:
        return
    
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