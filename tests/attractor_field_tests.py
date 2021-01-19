import unittest2
import numpy as np

from gradplanner.attractor_field import AttractorField


class AttractorFieldTests(unittest2.TestCase):
    """Tests the AttractorField class."""

    def setUp(self):
        """Set up the tests."""
        pass

    
    def test_init(self):
        """Tests the __init__() function of the AttractorField"""

        N, M = 10, 12
        occupancy_grid = np.ones((N, M))
        occupancy_grid[1: -1, 1: -1] = 0
        goal = np.array([5, 5])

        # testing without args:
        field = AttractorField()
        self.assertIsNone(field._occupancy_grid)
        self.assertIsNone(field._goal)

        # testing with provided occupancy_grid:
        field = AttractorField(occupancy_grid=occupancy_grid)
        self.assertTrue((field._occupancy_grid == occupancy_grid).all())
        self.assertIsNone(field._goal)
        self.assertEqual((N, M), field._grid_shape)
        self.assertTrue((np.array([N, M]) == field._grid_shape_arr).all())

        # testing with provided goal:
        field = AttractorField(goal=goal)
        self.assertIsNone(field._occupancy_grid)
        self.assertTrue((field._goal == goal).all())

        # testing with provided occupancy grid and goal:
        field = AttractorField(occupancy_grid=occupancy_grid, goal=goal)
        self.assertTrue((field._occupancy_grid == occupancy_grid).all())
        self.assertTrue((field._goal == goal).all())
        self.assertEqual((N, M), field._grid_shape)
        self.assertTrue((np.array([N, M]) == field._grid_shape_arr).all())
        self.assertIsNotNone(field._field)


    def test_init_field(self):
        """Tests _init_field method of the AttractorField"""
        N, M = 10, 12
        occupancy_grid = np.ones((N, M))
        occupancy_grid[1: -1, 1: -1] = 0
        goal = np.array([5, 5])

        # testing without args:
        field = AttractorField()
        with self.assertRaises(AssertionError):
            field._init_field()

        # testing with provided occupancy grid:
        field = AttractorField(occupancy_grid=occupancy_grid)
        with self.assertRaises(AssertionError):
            field._init_field()

        # testing with provided goal:
        field = AttractorField(goal=goal)
        with self.assertRaises(AssertionError):
            field._init_field()

        # testing with everything provided:
        field = AttractorField(occupancy_grid=occupancy_grid, goal=goal)
        field._init_field()
        self.assertIsNotNone(field._field)
        

if __name__ == "__main__":
    unittest2.main()