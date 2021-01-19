import unittest2
import numpy as np

from gradplanner.attractor_field import AttractorField
from gradplanner.utils import array_is_in_list


class AttractorFieldTests(unittest2.TestCase):
    """Tests the AttractorField class."""

    def setUp(self):
        """Sets up the tests."""
        self.N, self.M = 10, 12
        self.occupancy_grid = np.ones((self.N, self.M))
        self.occupancy_grid[1: -1, 1: -1] = 0
        self.goal = np.array([5, 5])

    
    def test_init(self):
        """Tests the __init__() function of the AttractorField"""
        # testing without args:
        field = AttractorField()
        self.assertIsNone(field._occupancy_grid)
        self.assertIsNone(field._goal)

        # testing with provided occupancy_grid:
        field = AttractorField(occupancy_grid=self.occupancy_grid)
        self.assertTrue((field._occupancy_grid == self.occupancy_grid).all())
        self.assertIsNone(field._goal)
        self.assertEqual((self.N, self.M), field._grid_shape)
        self.assertTrue((np.array([self.N, self.M]) == field._grid_shape_arr).all())

        # testing with provided goal:
        field = AttractorField(goal=self.goal)
        self.assertIsNone(field._occupancy_grid)
        self.assertTrue((field._goal == self.goal).all())

        # testing with provided occupancy grid and goal:
        field = AttractorField(occupancy_grid=self.occupancy_grid, goal=self.goal)
        self.assertTrue((field._occupancy_grid == self.occupancy_grid).all())
        self.assertTrue((field._goal == self.goal).all())
        self.assertEqual((self.N, self.M), field._grid_shape)
        self.assertTrue((np.array([self.N, self.M]) == field._grid_shape_arr).all())
        self.assertIsNotNone(field._field)


    def test_init_field(self):
        """Tests the _init_field method of the AttractorField"""
        # testing without args:
        field = AttractorField()
        with self.assertRaises(AssertionError):
            field._init_field()

        # testing with provided occupancy grid:
        field = AttractorField(occupancy_grid=self.occupancy_grid)
        with self.assertRaises(AssertionError):
            field._init_field()

        # testing with provided goal:
        field = AttractorField(goal=self.goal)
        with self.assertRaises(AssertionError):
            field._init_field()

        # testing with everything provided:
        field = AttractorField(occupancy_grid=self.occupancy_grid, goal=self.goal)
        field._init_field()
        self.assertIsNotNone(field._field)


    def test_set_new_goal(self):
        """Tests the set_new_goal method of the AttractorField"""

        field = AttractorField()
        # checking assertion errors:
        with self.assertRaises(AssertionError):
            field.set_new_goal(np.array([1, 2, 3]))
            field.set_new_goal(np.array([[1, 1]]))
        
        # checking goal setting without occupancy_grid:
        field.set_new_goal(self.goal)
        self.assertTrue((field._goal == self.goal).all())
        self.assertIsNone(field._occupancy_grid)

        # checking goal setting with occupancy grid:
        field = AttractorField()
        field._occupancy_grid = self.occupancy_grid
        field.set_new_goal(self.goal)
        self.assertTrue((field._goal == self.goal).all())
        self.assertIsNotNone(field._field)


    def test_update_occupancy_grid(self):
        """Tests the update_occupancy_grid method of the AttractorField"""
        # testing without original occupancy grid:
        field = AttractorField(goal=self.goal)
        field.update_occupancy_grid(self.occupancy_grid)
        self.assertTrue((field._occupancy_grid == self.occupancy_grid).all())
        self.assertEqual((self.N, self.M), field._grid_shape)
        self.assertTrue((np.array([self.N, self.M]) == field._grid_shape_arr).all())

        # testing with original occupancy grid:
        field = AttractorField(occupancy_grid=self.occupancy_grid, goal=self.goal)

        # test wrong shape assertion:
        with self.assertRaises(AssertionError):
            field.update_occupancy_grid(np.zeros((self.N - 1, self.M)))

        # check if nothing has changed:
        new_grid = self.occupancy_grid.copy()
        field.update_occupancy_grid(new_grid)
        self.assertTrue((field._occupancy_grid == new_grid).all())
        with self.assertRaises(AttributeError):
            a = field._changed_indices

        # check if something has changed:
        new_grid[5, 5] = 1
        new_grid[0, 3] = 0
        etalon_changes = np.sort(np.array([[5, 5], [0, 3]]), axis=0)
        field.update_occupancy_grid(new_grid)
        changes = np.sort(np.array(field._changed_indices), axis=0)
        self.assertTrue((field._occupancy_grid == new_grid).all())
        self.assertEqual(len(field._changed_indices), 2)
        i = 0
        for ind in changes:
            self.assertTrue((ind == etalon_changes[i]).all())
            i += 1


    def test_list_expandable_indices(self):
        """Tests the _list_expandable_indices method of the AttractorField"""
        
        field = AttractorField(occupancy_grid=self.occupancy_grid, goal=self.goal)
        field._changed_indices = [np.array([0, 5]), np.array([5, 5])]
        # expected list:
        etalon_indices = [np.array([1, 5]), np.array([4, 5]), np.array([6, 5]), np.array([5, 4]), np.array([5, 6])]

        # run function:
        indices = field._list_expandable_indices()

        # check if the list members are the same:
        self.assertEqual(len(etalon_indices), len(indices))
        for index in etalon_indices:
            self.assertTrue(array_is_in_list(index, indices))

        # check the order of the list:
        for i, index in enumerate(indices[: -1]):
            next_index = indices[i + 1]
            value = field._field[index[0], index[1]].value
            next_value = field._field[next_index[0], next_index[1]].value
            self.assertTrue(value >= next_value)


    def test_update_pixel(self):
        """Tests the _update_pixel method of the AttractorField"""

        field = AttractorField(occupancy_grid=self.occupancy_grid, goal=self.goal)
        # setting some pixels:
        for ind in [(6, 5), (6, 6), (5, 6), (4, 6), (4, 5), (4, 4), (6, 4)]:
            field._field[ind].value = -3
        field._field[5, 4].value = -2

        # the pixel value is already bigger:
        field._field[5, 5].value = -1
        pix = field._field[4, 5]
        new_pix = field._field[5, 5]
        new_pix = field._update_pixel(pix, new_pix)
        self.assertEqual(new_pix.value, -1)
        self.assertTrue((new_pix.grad == np.array([0, 0])).all())

        # the pixel value has to be changed:
        field._field[5, 5].value = -4
        pix = field._field[4, 5]
        new_pix = field._field[5, 5]
        new_pix = field._update_pixel(pix, new_pix)
        self.assertEqual(new_pix.value, -3)
        self.assertTrue((new_pix.grad == np.array([0, -1])).all())

        


if __name__ == "__main__":
    unittest2.main()