import unittest2
import numpy as np

from gradplanner.controller.grad_controller import GradController
from gradplanner.planner.attractor_field import AttractorField
from gradplanner.planner.repulsive_field import RepulsiveField

class GradControllerTests(unittest2.TestCase):
    """Tests of the GradController class."""

    def setUp(self):
        """Sets up the tests."""
        self.param_file = "params/test_params.json"

        self.N, self.M = 10, 12
        self.occupancy_grid = np.ones((self.N, self.M))
        self.occupancy_grid[1: -1, 1: -1] = 0
        self.goal = np.array([5, 5])
        self.R = 4


    def test_init(self):
        """Tests the __init__ function of the GradController"""

        controller = GradController(occupancy_grid=self.occupancy_grid,
            goal= self.goal,
            R=self.R,
            param_file=self.param_file)
        
        self.assertTrue((controller._occupancy_grid == self.occupancy_grid).all())
        self.assertTrue((controller._goal == self.goal).all())
        self.assertEqual(controller._R, self.R)

        self.assertIsInstance(controller._attractor, AttractorField)
        self.assertIsInstance(controller._repulsive, RepulsiveField)
        
        # params which are set from the json file:
        self.assertEqual(controller._pos_tolerance, 0.1)
        self.assertEqual(controller._ang_tolerance, 0.2)
        self.assertEqual(controller._max_trans_vel, 1)
        self.assertEqual(controller._max_trans_acc, 1)
        self.assertEqual(controller._max_ang_vel, 1.5708)
        self.assertEqual(controller._max_ang_acc, 1.5708)
        self.assertEqual(controller._K_grad, 0.5)
        self.assertEqual(controller._max_ang_error, 1.0472)
        self.assertEqual(controller._grad_vel_scaling, False)
        self.assertEqual(controller._K_pos, 0.5)
        self.assertEqual(controller._K_ang, 0.5)
        self.assertEqual(controller._K_ang_end, 0.5)


if __name__ == "__main__":
    unittest2.main()