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

        field = RepulsiveField(occupancy_grid=self.occupancy_grid, R=4)

        # there is no neighbour to expand:
        for i in range(4, 7):
            for j in range(4, 7):
                field._field[i, j].value = 0
                field._field[i, j].parent = None
        field._field[5, 5].parent = (8, 8)
        field._field[6, 5].parent = (8, 8)
        field._field[6, 6].parent = (8, 8)
        field._field[6, 6].value = 2
        field._field[4, 5].value = 5
        field._field[4, 5].parent = (9, 9)

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


    def test_get_first_not_influenced_pixels(self):
        """Tests the _get_first_not_influenced_pixels method of the RepulsiveField"""

        field = RepulsiveField(occupancy_grid=self.occupancy_grid)

        # first check it if there is nothing to return:
        # As if an obstacle in (5, 5) would have disappeared, but there is no other obstacle
        # in the neighbourhood.
        for i in range(1, self.N - 1):
            for j in range(1, self.M - 1):
                field._field[i, j].value = 0
                field._field[i, j].parent = None

        for i in range(4, 7):
            for j in range(4, 7):
                field._field[i, j].value = 2
                field._field[i, j].parent = (5, 5)
        
        out = field._get_first_not_influenced_pixels(np.array([5, 5]))
        self.assertEqual(out, [])

        # second, check if there is something to return:
        # As if an obstacle in (5, 5) would have disappeared and there are some other
        # pixels which are occupied with different parents.
        for i in range(4, 7):
            for j in range(4, 7):
                field._field[i, j].value = 2
                field._field[i, j].parent = (5, 5)

        field._field[7, 7].parent = (9, 9)
        field._field[7, 7].value = 3
        field._field[7, 6].parent = (9, 9)
        field._field[7, 6].value = 4
        field._field[7, 5].parent = (9, 9)
        field._field[7, 5].value = 5
        field._field[7, 4].parent = (9, 9)
        field._field[7, 4].value = 1
        field._field[7, 3].parent = (9, 9)
        field._field[7, 3].value = 0

        out = field._get_first_not_influenced_pixels(np.array([5, 5]))
        print(out)

        self.assertEqual(len(out), 2)
        for ind in [np.array([7, 7]), np.array([7, 4])]:
            self.assertTrue(array_is_in_list(ind, out))






if __name__ == "__main__":
    unittest2.main()