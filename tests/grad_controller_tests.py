import unittest2
import numpy as np
import json

from gradplanner.controller.grad_controller import GradController
from gradplanner.planner.attractor_field import AttractorField
from gradplanner.planner.repulsive_field import RepulsiveField

class GradControllerTests(unittest2.TestCase):
    """Tests of the GradController class."""

    def setUp(self):
        """Sets up the tests."""
        param_file = "tests/params/test_params.json"
        with open(param_file) as f:
            self.params = json.load(f)

        self.N, self.M = 10, 12
        self.occupancy_grid = np.ones((self.N, self.M))
        self.occupancy_grid[1: -1, 1: -1] = 0
        self.goal_pos = np.array([5.2, 5.4])
        self.goal_ang = 0
        self.R = 4


    def test_init(self):
        """Tests the __init__ function of the GradController"""

        controller = GradController(occupancy_grid=self.occupancy_grid,
            goal_pos=self.goal_pos,
            goal_ang=self.goal_ang,
            R=self.R,
            params=self.params)
        
        self.assertTrue((controller._occupancy_grid == self.occupancy_grid).all())
        self.assertTrue((controller._goal_pos == self.goal_pos).all())
        self.assertEqual(controller._goal_ang, self.goal_ang)
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
        self.assertEqual(controller._boundar_error_grad, 0.2)
        self.assertEqual(controller._max_error_grad, 1.0472)
        self.assertEqual(controller._grad_vel_scaling, False)

        self.assertEqual(controller._K_direct, 0.5)
        self.assertEqual(controller._boundar_error_direct, 0.2)
        self.assertEqual(controller._max_error_direct, 1.0472)

        self.assertEqual(controller._K_end, 0.5)


    def test_set_pose(self):
        """Tests the _set_pose function of the GradController"""

        controller = GradController(occupancy_grid=self.occupancy_grid,
            goal_pos=self.goal_pos,
            goal_ang=self.goal_ang,
            R=self.R,
            params=self.params)

        pose = np.array([4.3, 8.7, -0.3])
        controller._set_pose(pose)

        self.assertTrue((controller._pos == np.array([4.3, 8.7])).all())
        self.assertEqual(controller._x, 4.3)
        self.assertEqual(controller._y, 8.7)
        self.assertEqual(controller._psi, -0.3)
        self.assertEqual(controller._i, 4)
        self.assertEqual(controller._j, 8)


    def test_goal_is_visible(self):
        """Tests the _goal_is_visible function of the GradController"""

        controller = GradController(occupancy_grid=self.occupancy_grid,
            goal_pos=self.goal_pos,
            goal_ang=self.goal_ang,
            R=self.R,
            params=self.params)

        # there is an obstacle between the position and the goal:
        controller._occupancy_grid[6, 5] = 1
        pose = np.array([8.3, 5.6, 0.3])
        controller._set_pose(pose)

        self.assertFalse(controller._goal_is_visible())

        # there is no obstacle between the position and the goal:
        pose = np.array([5.6, 8.3, 0.3])
        controller._set_pose(pose)


    def test_get_ang_diff(self):
        """Tests the _get_ang_diff function of the GradController"""

        controller = GradController(occupancy_grid=self.occupancy_grid,
            goal_pos=self.goal_pos,
            goal_ang=self.goal_ang,
            R=self.R,
            params=self.params)

        pi = np.pi

        desired, real = 5. / 6. * pi, 1. / 6. * pi
        self.assertTrue(np.isclose(controller._get_ang_diff(desired, real), -4. / 6. * pi))

        desired, real = 1. / 6. * pi, 5. / 6. * pi 
        self.assertTrue(np.isclose(controller._get_ang_diff(desired, real), 4. / 6. * pi))

        desired, real = -5. / 6. * pi, -1. / 6. * pi
        self.assertTrue(np.isclose(controller._get_ang_diff(desired, real), 4. / 6. * pi))

        desired, real = -1. / 6. * pi, -5. / 6. * pi 
        self.assertTrue(np.isclose(controller._get_ang_diff(desired, real), -4. / 6. * pi))

        desired, real = 1. / 6. * pi, -3. / 6. * pi
        self.assertTrue(np.isclose(controller._get_ang_diff(desired, real), -4. / 6. * pi))

        desired, real = -1. / 6. * pi, 3. / 6. * pi
        self.assertTrue(np.isclose(controller._get_ang_diff(desired, real), 4. / 6. * pi))

        desired, real = 5. / 6. * pi, -3. / 6. * pi
        self.assertTrue(np.isclose(controller._get_ang_diff(desired, real), 4. / 6. * pi))

        desired, real = -5. / 6. * pi, 3. / 6. * pi
        self.assertTrue(np.isclose(controller._get_ang_diff(desired, real), -4. / 6. * pi))

        desired, real = 3. / 6. * pi, -5. / 6. * pi
        self.assertTrue(np.isclose(controller._get_ang_diff(desired, real), 4. / 6. * pi))

        desired, real = -3. / 6. * pi, 5. / 6. * pi
        self.assertTrue(np.isclose(controller._get_ang_diff(desired, real), -4. / 6. * pi))


    def test_get_ang_vel(self):
        """Tests the _get_ang_vel function of the GradController"""

        controller = GradController(occupancy_grid=self.occupancy_grid,
            goal_pos=self.goal_pos,
            goal_ang=self.goal_ang,
            R=self.R,
            params=self.params)

        controller._max_ang_vel = 0.1

        out = controller._get_ang_vel(0.16, 0.5)
        self.assertEqual(out, -0.08)
        self.assertFalse(controller._goal_ang_is_reached)

        out = controller._get_ang_vel(-0.16, 0.5)
        self.assertEqual(out, 0.08)
        self.assertFalse(controller._goal_ang_is_reached)

        out = controller._get_ang_vel(0.3, 0.5)
        self.assertEqual(out, -0.1)
        self.assertFalse(controller._goal_ang_is_reached)

        out = controller._get_ang_vel(-0.3, 0.5)
        self.assertEqual(out, 0.1)
        self.assertFalse(controller._goal_ang_is_reached)


    def test_get_cmd_vel_end(self):
        """Tests the _get_cmd_vel_end function of the GradController"""

        controller = GradController(occupancy_grid=self.occupancy_grid,
            goal_pos=self.goal_pos,
            goal_ang=self.goal_ang,
            R=self.R,
            params=self.params)
        
        controller._max_ang_vel = 0.1
        controller._ang_tolerance = 0.05

        pose = np.array([0, 0, 0.16])
        controller._set_pose(pose)
        out = controller._get_cmd_vel_end()
        self.assertTrue((out == np.array([0, -0.08])).all())
        self.assertFalse(controller._goal_ang_is_reached)

        pose = np.array([0, 0, -0.16])
        controller._set_pose(pose)
        out = controller._get_cmd_vel_end()
        self.assertTrue((out == np.array([0, 0.08])).all())
        self.assertFalse(controller._goal_ang_is_reached)

        pose = np.array([0, 0, 1.6])
        controller._set_pose(pose)
        out = controller._get_cmd_vel_end()
        self.assertTrue((out == np.array([0, -0.1])).all())
        self.assertFalse(controller._goal_ang_is_reached)

        pose = np.array([0, 0, -1.6])
        controller._set_pose(pose)
        out = controller._get_cmd_vel_end()
        self.assertTrue((out == np.array([0, 0.1])).all())
        self.assertFalse(controller._goal_ang_is_reached)

        pose = np.array([0, 0, 0.04])
        controller._set_pose(pose)
        out = controller._get_cmd_vel_end()
        self.assertTrue((out == np.array([0, 0])).all())
        self.assertTrue(controller._goal_ang_is_reached)

        pose = np.array([0, 0, -0.04])
        controller._set_pose(pose)
        out = controller._get_cmd_vel_end()
        self.assertTrue((out == np.array([0, 0])).all())
        self.assertTrue(controller._goal_ang_is_reached)


    def test_get_trans_vel(self):
        """Tests the _get_trans_vel function of the GradController"""

        controller = GradController(occupancy_grid=self.occupancy_grid,
            goal_pos=self.goal_pos,
            goal_ang=self.goal_ang,
            R=self.R,
            params=self.params)

        out = controller._get_trans_vel(0.19, 0.2, 0.6)
        self.assertTrue(np.isclose(out, controller._max_trans_vel))

        out = controller._get_trans_vel(-0.19, 0.2, 0.6)
        self.assertTrue(np.isclose(out, controller._max_trans_vel))

        out = controller._get_trans_vel(0.2, 0.2, 0.6)
        self.assertTrue(np.isclose(out, controller._max_trans_vel))

        out = controller._get_trans_vel(-0.2, 0.2, 0.6)
        self.assertTrue(np.isclose(out, controller._max_trans_vel))

        out = controller._get_trans_vel(0.4, 0.2, 0.6)
        self.assertTrue(np.isclose(out, controller._max_trans_vel / 2))

        out = controller._get_trans_vel(-0.4, 0.2, 0.6)
        self.assertTrue(np.isclose(out, controller._max_trans_vel / 2))

        out = controller._get_trans_vel(0.6, 0.2, 0.6)
        self.assertEqual(out, 0)

        out = controller._get_trans_vel(-0.6, 0.2, 0.6)
        self.assertEqual(out, 0)

        out = controller._get_trans_vel(0.61, 0.2, 0.6)
        self.assertEqual(out, 0)

        out = controller._get_trans_vel(-0.61, 0.2, 0.6)
        self.assertEqual(out, 0)


if __name__ == "__main__":
    unittest2.main()