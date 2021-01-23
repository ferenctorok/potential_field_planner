import unittest2
import numpy as np

from gradplanner.planner.repulsive_field import RepulsiveField
from gradplanner.planner.utils import array_is_in_list
from gradplanner.planner.field_utils import get_values_from_field

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
        self.assertEqual(out, [])

        # there are neighbors to return:
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
        
        self.assertEqual(len(out), 5)
        for ind in [np.array([5, 6]), np.array([4, 6]), np.array([4, 4]), np.array([5, 4]), np.array([6, 4])]:
            self.assertTrue(array_is_in_list(ind, out))


    def test_get_first_not_influenced_pixels(self):
        """Tests the _get_first_not_influenced_pixels method of the RepulsiveField.
        An obstacle has disappeared from (5, 5).
        """

        field = RepulsiveField(occupancy_grid=self.occupancy_grid, R=4)

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
        
        indices, values = field._get_first_not_influenced_pixels(np.array([5, 5]))
        self.assertEqual(indices, [])
        self.assertEqual(values, [])

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

        indices, values = field._get_first_not_influenced_pixels(np.array([5, 5]))

        self.assertEqual(len(indices), 3)
        self.assertEqual(len(values), 3)
        for ind in [np.array([7, 7]), np.array([7, 6]), np.array([7, 4])]:
            self.assertTrue(array_is_in_list(ind, indices))


    def test_list_expandable_indices(self):
        """Tests the _list_expandable_indices method of the RepulsiveField.
        There is going to be a disappeared obstacle in (5, 5) and a 
        new obstacle in (2, 2).
        """

        field = RepulsiveField(occupancy_grid=self.occupancy_grid, R=4)

        # the setup from test_get_first_not_influenced_pixels:
        for i in range(1, self.N - 1):
            for j in range(1, self.M - 1):
                field._field[i, j].value = 0
                field._field[i, j].parent = None

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

        # also placing an extra obstacle in the map, which is a new one:
        field._field[2, 2].value = 1
        
        # setting up the changed indices and run the method:
        field._changed_indices = [np.array([5, 5]), np.array([2, 2])]
        out = field._list_expandable_indices()

        self.assertEqual(len(out), 4)
        for ind in [np.array([7, 7]), np.array([7, 6]), np.array([7, 4]), np.array([2, 2])]:
            self.assertTrue(array_is_in_list(ind, out))

        # check wether they are really in growing order:
        last = 0
        for index in out:
            self.assertGreaterEqual(field._field[index[0], index[1]].value, last)
            last = field._field[index[0], index[1]].value


    def test_update_occupancy_grid(self):
        """Tests the update_occupancy_grid method of the RepulsiveField."""

        def carry_out_update_test(occ_old, occ_new):
            """Carries out the testing of the update_occupancy_grid method.
            Sets up 2 fields with occ_old and occ_new. Then the one which was set
            up with occ_old is updated with occ_new. This should result in two
            identical potential and gradient fields.
            """

            field1 = RepulsiveField(occupancy_grid=occ_new)
            field2 = RepulsiveField(occupancy_grid=occ_old)

            field2.update_occupancy_grid(occ_new)

            # testing the values:
            result_vals1 = get_values_from_field(field1._field)
            result_vals2 = get_values_from_field(field2._field)
            self.assertTrue((result_vals1 == result_vals2).all())

            # testing the grads:
            for i in range(self.N):
                for j in range(self.M):
                    self.assertTrue((field1._field[i, j].grad == field2._field[i, j].grad).all())

        ### 1: a new obstacle is inserted. ###
        occ_old = self.occupancy_grid.copy()
        occ_new = self.occupancy_grid.copy()

        # inserting u shaped obstacle in occ_new:
        occ_new[6, 4: 7] = 1
        occ_new[7, 4] = 1
        occ_new[7, 6] = 1

        # carrying out the tests:
        carry_out_update_test(occ_old, occ_new)

        ### 2: Obstacle is deleted. ###
        occ_old = self.occupancy_grid.copy()
        occ_new = self.occupancy_grid.copy()

        # inserting u shaped obstacle in occ_old:
        occ_old[6, 4: 7] = 1
        occ_old[7, 4] = 1
        occ_old[7, 6] = 1

        # carrying out the tests:
        carry_out_update_test(occ_old, occ_new)
        
        ### 3: one obstacle is inserted and one is deleted:
        occ_old = self.occupancy_grid.copy()
        occ_new = self.occupancy_grid.copy()

        # inserting an obstacle in occ_old:
        occ_old[3, 2] = 1

        # inserting u shaped obstacle in occ_new:
        occ_new[6, 4: 7] = 1
        occ_new[7, 4] = 1
        occ_new[7, 6] = 1

        # carrying out the tests:
        carry_out_update_test(occ_old, occ_new)

        ### 4: inserting 2 obstacles at once. ###
        occ_old = self.occupancy_grid.copy()
        occ_new = self.occupancy_grid.copy()

        # inserting u shaped obstacle in occ_new:
        occ_new[6, 4: 7] = 1
        occ_new[7, 4] = 1
        occ_new[7, 6] = 1
        # inserting an extra obstacle in occ_new:
        occ_new[3, 2] = 1

        # carrying out the tests:
        carry_out_update_test(occ_old, occ_new)

        ### 5: Removing 2 obstacles at once. ###
        occ_old = self.occupancy_grid.copy()
        occ_new = self.occupancy_grid.copy()

        # inserting u shaped obstacle in occ_old:
        occ_old[6, 4: 7] = 1
        occ_old[7, 4] = 1
        occ_old[7, 6] = 1
        # inserting an extra obstacle in occ_old:
        occ_old[3, 2] = 1

        # carrying out the tests:
        carry_out_update_test(occ_old, occ_new)


        

if __name__ == "__main__":
    unittest2.main()