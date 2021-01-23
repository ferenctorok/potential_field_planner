import numpy as np
import json

from gradplanner.planner.attractor_field import AttractorField
from gradplanner.planner.repulsive_field import RepulsiveField

class GradController:
    """Gradient field based controller."""

    def __init__(self,
                 occupancy_grid,
                 goal,
                 R,
                 param_file):
        """Initializes the GradController object."""

        self._occupancy_grid = occupancy_grid
        self._goal = goal
        self._R = R

        # creating the attractive and repulsive gradient fields:
        self._attractor = AttractorField(occupancy_grid=self._occupancy_grid,
            goal=self._goal)
        self._repulsive = RepulsiveField(occupancy_grid=self._occupancy_grid,
            R=self._R)
        
        # setting up some params based on the json file if provided:
        self._set_from_params(param_file)


    def _set_from_params(self, param_file):
        """Sets up some values based on params."""

        with open(param_file) as f:
            params = json.load(f)

        # general
        self._pos_tolerance = params["general"]["pos_tolerance"]
        self._ang_tolerance = params["general"]["ang_tolerance"]
        self._max_trans_vel = params["general"]["max_trans_vel"]
        self._max_trans_acc = params["general"]["max_trans_acc"]
        self._max_ang_vel = params["general"]["max_ang_vel"]
        self._max_ang_acc = params["general"]["max_ang_acc"]

        # grad_mode
        self._K_grad = params["grad_mode"]["K_grad"]
        self._max_ang_error = params["grad_mode"]["max_ang_error"]
        self._grad_vel_scaling = params["grad_mode"]["grad_vel_scaling"]

        # direct_mode:
        self._K_pos = params["direct_mode"]["K_pos"]
        self._K_ang = params["direct_mode"]["K_ang"]

        # end_mode:
        self._K_ang_end = params["end_mode"]["K_ang_end"]
