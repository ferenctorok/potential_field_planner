import unittest2
import numpy as np

from gradplanner.attractor_field import AttractorField


class AttractorFieldTests(unittest2.TestCase):
    """Tests the AttractorField class."""

    def setUp(self):
        """Set up the tests."""
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
        """Tests _init_field method of the AttractorField"""
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
        """Tests set_new_goal method of the AttractorField"""

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
        """Tests update_occupancy_grid method of the AttractorField"""
        # testing without original occupancy grid:
        field = AttractorField()
        field.update_occupancy_grid(self.occupancy_grid)
        self.assertTrue((field._occupancy_grid == self.occupancy_grid).all())
        self.assertEqual((self.N, self.M), field._grid_shape)
        self.assertTrue((np.array([self.N, self.M]) == field._grid_shape_arr).all())

        # testing with original occupancy grid:
        field = AttractorField(occupancy_grid=self.occupancy_grid)

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
        

        


if __name__ == "__main__":
    unittest2.main()