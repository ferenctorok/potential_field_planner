import unittest2
import numpy as np

from gradplanner.repulsive_field import RepulsiveField
from gradplanner.utils import array_is_in_list
from gradplanner.field_utils import get_values_from_field

class RepulsiveFieldTests(unittest2.TestCase):
    """Tests of te RepulsiveField class."""

    def setUp(self):
        """Sets up the tests."""
        self.N, self.M = 10, 12
        self.occupancy_grid = np.ones((self.N, self.M))
        self.occupancy_grid[1: -1, 1: -1] = 0
        self.goal = np.array([5, 5])


    def test_init(self):
        """Tests the __init__() function of the RepulsiveField"""
        # testing without args:
        field = RepulsiveField()
        self.assertIsNone(field._occupancy_grid)
        self.assertEqual(field._R, 5)

        # testing with provided occupancy_grid:
        field = RepulsiveField(occupancy_grid=self.occupancy_grid)
        self.assertTrue((field._occupancy_grid == self.occupancy_grid).all())
        self.assertEqual(field._R, 5)
        self.assertEqual((self.N, self.M), field._grid_shape)
        self.assertTrue((np.array([self.N, self.M]) == field._grid_shape_arr).all())

        # testing with provided R:
        field = RepulsiveField(R=7)
        self.assertIsNone(field._occupancy_grid)
        self.assertEqual(field._R, 7)

        # testing with non int R:
        with self.assertRaises(AssertionError):
            field = RepulsiveField(R=7.2)

        # testing with provided occupancy grid and goal:
        field = RepulsiveField(occupancy_grid=self.occupancy_grid, R=7)
        self.assertTrue((field._occupancy_grid == self.occupancy_grid).all())
        self.assertEqual(field._R, 7)
        self.assertEqual((self.N, self.M), field._grid_shape)
        self.assertTrue((np.array([self.N, self.M]) == field._grid_shape_arr).all())
        self.assertIsNotNone(field._field)


    def test_init_field(self):
        """Tests the _init_field method of the RepulsiveField"""
        # testing without args:
        field = RepulsiveField()
        with self.assertRaises(AssertionError):
            field._init_field()
        with self.assertRaises(AttributeError):
            field._field

        # testing with provided occupancy grid:
        field = RepulsiveField(occupancy_grid=self.occupancy_grid)
        field._init_field()
        self.assertIsNotNone(field._field)


    def test_search_surrounding_for_expandable(self):
        """Tests the _search_surrounding_for_expandable method of the RepulsiveField"""

        field = RepulsiveField(occupancy_grid=self.occupancy_grid)

        # there is no neighbour to expand:
        for i in range(4, 7):
            for j in range(4, 7):
                field._field[i, j].value = 0
                field._field[i, j].parent = None
        field._field[5, 5].parent = (8, 8)
        field._field[6, 5].parent = (8, 8)
        field._field[6, 6].parent = (8, 8)
        field._field[6, 6].value = 2

        out = field._search_surrounding_for_expandable(field._field[5, 5])
        self.assertIsNone(out)

        # there is a neighbor to return:
        # there is no neighbour to expand:
        for i in range(4, 7):
            for j in range(4, 7):
                field._field[i, j].value = 3
                field._field[i, j].parent = (9, 9)
        field._field[5, 5].parent = (8, 8)
        field._field[6, 5].parent = (8, 8)
        field._field[6, 6].value = 0
        field._field[4, 5].parent = (8, 8)
        field._field[4, 5].value = 1
        field._field[4, 4].value = 2

        out = field._search_surrounding_for_expandable(field._field[5, 5])
        self.assertTrue((out == np.array([4, 4])).all())





if __name__ == "__main__":
    unittest2.main()