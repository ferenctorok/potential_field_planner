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
                return self._get_cmd_vel_end()
        if self._goal_is_visible():
            return self._get_cmd_vel_visible()
        else:
            return self._get_cmd_vel_grad(pose)


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
        the robot is not in the effected area of any obstacle.
        """
        
        if self._repulsive.is_influenced(self._i, self._j):
            return False
        else:
            # carry out a raytracing:
            vect = self._goal_pos - self._pos
            distance = np.linalg.norm(vect)
            vect = vect / distance
            N = np.floor(distance)
            point = self._pos

            for _ in range(N):
                point += vect
                i, j = np.floor(point[0]), np.floor(point[1])
                if self._occupancy_grid[i, j] == 1:
                    return False
            
            return True


    def _get_cmd_vel_end(self):
        """Controller for the case when the robot has already reached the goal 
        position, only the orientation has to be changed.
        """

        ang_diff = self._get_ang_diff(self._goal_ang, self._psi)
        if abs(ang_diff) > self._ang_tolerance:
            des_ang_vel = - self._K_end * ang_diff
            if abs(des_ang_vel) > self._max_ang_vel:
                des_ang_vel = np.sign(des_ang_vel) * self._max_ang_vel
            return np.array([0, des_ang_vel])
        else:
            self._goal_ang_is_reached = True
            return np.array([0, 0])


    def _get_cmd_vel_visible(self):
        """Controller for the case when the goal is visible from the current 
        position of the robot.
        """

        vect = self._goal_pos - self._pos
        desired_direction = np.arctan2(vect[1], vect[0])
        ang_diff = self._get_ang_diff(desired_direction, self._psi)

        # calculating the desired angular velocity:
        des_ang_vel = - self._K_direct * ang_diff
        if abs(des_ang_vel) > self._max_ang_vel:
            des_ang_vel = np.sign(des_ang_vel) * self._max_ang_vel

        # calculating the desired translational velocity:
        des_trans_vel = self._get_trans_vel(ang_diff,
            self._boundar_error_direct, self._max_error_direct)
        
        return np.array([des_trans_vel, des_ang_vel])
        

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
    

    def _set_from_params(self, params):
        """Sets up some values based on params."""

        # general
        self._pos_tolerance = params["general"]["pos_tolerance"]
        self._ang_tolerance = params["general"]["ang_tolerance"]
        self._max_trans_vel = params["general"]["max_trans_vel"]
        self._max_trans_acc = params["general"]["max_trans_acc"]
        self._max_ang_vel = params["general"]["max_ang_vel"]
        self._max_ang_acc = params["general"]["max_ang_acc"]

        # grad_mode
        self._K_grad = params["grad_mode"]["K"]
        self._boundar_error_grad = params["grad_mode"]["boundary_error"]
        self._max_error_grad = params["grad_mode"]["max_error"]
        self._grad_vel_scaling = params["grad_mode"]["grad_vel_scaling"]

        # direct_mode:
        self._K_direct = params["direct_mode"]["K"]
        self._boundar_error_direct = params["direct_mode"]["boundary_error"]
        self._max_error_direct = params["direct_mode"]["max_error"]

        # end_mode:
        self._K_end = params["end_mode"]["K_end"]
