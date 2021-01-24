import numpy as np
import json

from gradplanner.planner.attractor_field import AttractorField
from gradplanner.planner.repulsive_field import RepulsiveField

class GradController:
    """Gradient field based controller."""

    def __init__(self,
                 occupancy_grid,
                 goal_pos,
                 goal_ang,
                 R,
                 params):
        """Initializes the GradController object."""

        self._occupancy_grid = occupancy_grid
        self.set_new_goal(goal_pos, goal_ang)
        self._R = R

        # creating the attractive and repulsive gradient fields:
        self._attractor = AttractorField(occupancy_grid=self._occupancy_grid,
            goal=self._goal_pos)
        self._repulsive = RepulsiveField(occupancy_grid=self._occupancy_grid,
            R=self._R)
        
        # setting up some params based on the json file if provided:
        self._set_from_params(params)


    def set_new_goal(self, goal_pos, goal_ang):
        """Sets a new goal."""

        self._goal_pos = goal_pos
        self._goal_ang = goal_ang
        self._goal_pos_is_reached = False
        self._goal_ang_is_reached = False


    def get_cmd_vel(self, pose):
        """Gets the cmd_vel to send to the low level controller of the robot."""

        self._set_pose(pose)

        if self._goal_pos_is_reached:
            if self._goal_ang_is_reached:
                return np.array([0, 0])
            else:
                print("In End mode.")
                return self._get_cmd_vel_end()
        if self._goal_is_visible():
            print("In Direct mode.")
            return self._get_cmd_vel_direct()
        else:
            print("In Grad mode.")
            return self._get_cmd_vel_grad()


    def _set_pose(self, pose):
        """sets pose related variables."""

        self._pos = pose[0: 2]
        self._x = pose[0]
        self._y = pose[1]
        self._psi = pose[2]
        self._i = int(np.floor(pose[0]))
        self._j = int(np.floor(pose[1]))


    def _goal_is_visible(self):
        """Returns true if there is no obstacle between the robot and the goal and
        the robot is at least self._min_obst_dist away from the nearest obstacle.
        """
        
        val = self._repulsive.get_val(self._i, self._j)
        if (0 < val < self._min_obst_dist):
            return False
        else:
            # carry out a raytracing:
            vect = self._goal_pos - self._pos
            distance = np.linalg.norm(vect)
            vect = vect / distance
            N = int(np.floor(distance))
            point = self._pos

            for _ in range(N):
                point += vect
                i, j = int(np.floor(point[0])), int(np.floor(point[1]))
                if self._occupancy_grid[i, j] == 1:
                    return False
            
            return True


    def _get_cmd_vel_end(self):
        """Controller for the case when the robot has already reached the goal 
        position, only the orientation has to be changed.
        """

        ang_diff = self._get_ang_diff(self._goal_ang, self._psi)
        if abs(ang_diff) > self._ang_tolerance:
            return np.array([0, self._get_ang_vel(ang_diff, self._K_end)])
        else:
            self._goal_ang_is_reached = True
            return np.array([0, 0])


    def _get_cmd_vel_direct(self):
        """Controller for the case when the goal is visible from the current 
        position of the robot.
        """

        vect = self._goal_pos - self._pos
        desired_direction = np.arctan2(vect[1], vect[0])
        ang_diff = self._get_ang_diff(desired_direction, self._psi)

        # calculating the desired angular velocity:
        des_ang_vel = self._get_ang_vel(ang_diff, self._K_direct)

        # calculating the desired translational velocity:
        des_trans_vel = self._get_trans_vel(ang_diff,
            self._boundar_error_direct, self._max_error_direct)
        
        return np.array([des_trans_vel, des_ang_vel])


    def _get_cmd_vel_grad(self):
        """Controller for the case when the robot is in the influenced
        area of obstacles. It uses 3 pixels from the occupancy grid:
        - 1: The pixel it is in.
        - 2 and 3: neares neighboring pixels. 
        """

        grad1 = self._attractor.get_grad(self._i, self._j) +\
            self._repulsive.get_grad(self._i, self._j)

        i1, j1 = np.round(self._i), np.round(self._j)
        if (j1 >= 0) and (j1 < self._occupancy_grid.shape[1]):
            grad2 = self._attractor.get_grad(self._i, j1) +\
                self._repulsive.get_grad(self._i, j1)
        else:
            grad2 = np.array([0, 0])

        if (i1 >= 0) and (i1 < self._occupancy_grid.shape[0]):
            grad3 = self._attractor.get_grad(i1, self._j) +\
                self._repulsive.get_grad(i1, self._j)
        else:
            grad3 = np.array([0, 0])

        #grad = self._norm_grad(grad1) + self._norm_grad(grad2) +\
        #    self._norm_grad(grad3)

        grad = grad1 + grad2 + grad3

        # getting the desired angular and translational velocity:
        desired_direction = np.arctan2(grad[1], grad[0])
        ang_diff = self._get_ang_diff(desired_direction, self._psi)

        # calculating the desired angular velocity:
        des_ang_vel = self._get_ang_vel(ang_diff, self._K_grad)

        # calculating the desired translational velocity:
        des_trans_vel = self._get_trans_vel(ang_diff,
            self._boundar_error_grad, self._max_error_grad)

        return np.array([des_trans_vel, des_ang_vel])


    def _norm_grad(self, grad):
        """Normalizes a gradient"""

        eps = 1e-6
        length = np.linalg.norm(grad)
        if length > eps:
            return grad / length
        else:
            return np.array([0, 0])

    
    def _get_ang_vel(self, ang_diff, K):
        """Gets the desired velocity based on a simple proportional
        relationship with the error. It also respects the max angular velocity."""

        des_ang_vel = - self._K_direct * ang_diff
        if abs(des_ang_vel) > self._max_ang_vel:
            des_ang_vel = np.sign(des_ang_vel) * self._max_ang_vel
        
        return des_ang_vel


    def _get_trans_vel(self, ang_diff, boundary_error, max_error):
        """Gets the desired translational velocity for the robot.
        - if abs(ang_diff) < boundary_error: max velocity
        - if boundary_error < abs(ang_diff) < max_error: linearly decreasing
            velocity between max velocity and 0
        - else: 0 translational velocity.
        """

        if abs(ang_diff) < boundary_error:
            return self._max_trans_vel
        elif abs(ang_diff) < max_error:
            ratio = (abs(ang_diff) - boundary_error) / (max_error - boundary_error)
            return self._max_trans_vel * (1 - ratio)
        else:
            return 0

    def _get_ang_diff(self, desired, real):
        """gets the orientation difference between the desired
        and the real orientation. The value is always in the range [-pi, pi]
        """

        diff = real - desired
        if abs(diff) < np.pi:
            return diff
        else:
            return diff - np.sign(diff) * 2 * np.pi

    
    @property
    def goal_is_reached(self):
        """Returns true if the goal is reached."""
        return self._goal_pos_is_reached and self._goal_ang_is_reached


    def _set_from_params(self, params):
        """Sets up some values based on params."""

        # general
        self._pos_tolerance = params["GradController"]["general"]["pos_tolerance"]
        self._ang_tolerance = params["GradController"]["general"]["ang_tolerance"]
        self._max_trans_vel = params["GradController"]["general"]["max_trans_vel"]
        self._max_trans_acc = params["GradController"]["general"]["max_trans_acc"]
        self._max_ang_vel = params["GradController"]["general"]["max_ang_vel"]
        self._max_ang_acc = params["GradController"]["general"]["max_ang_acc"]

        # grad_mode
        self._K_grad = params["GradController"]["grad_mode"]["K"]
        self._boundar_error_grad = params["GradController"]["grad_mode"]["boundary_error"]
        self._max_error_grad = params["GradController"]["grad_mode"]["max_error"]
        self._grad_vel_scaling = params["GradController"]["grad_mode"]["grad_vel_scaling"]

        # direct_mode:
        self._min_obst_dist = params["GradController"]["direct_mode"]["min_obst_dist"]
        self._K_direct = params["GradController"]["direct_mode"]["K"]
        self._boundar_error_direct = params["GradController"]["direct_mode"]["boundary_error"]
        self._max_error_direct = params["GradController"]["direct_mode"]["max_error"]

        # end_mode:
        self._K_end = params["GradController"]["end_mode"]["K_end"]
