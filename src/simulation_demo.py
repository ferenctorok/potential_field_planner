import numpy as np
import json

from gradplanner.controller.grad_controller import GradController
from gradplanner.controller.low_level_controller import LowLevelController
from gradplanner.simulation.environment import PointRobotEnv
from gradplanner.simulation.utils import set_up_occ_grid


# loading the params:
param_file = "src/params/params.json"
with open(param_file) as f:
    params = json.load(f)

sim_param_file = "src/params/sim_1.json"
with open(sim_param_file) as f:
    sim_params = json.load(f)

# generating occupancy grid:
occ_grid = set_up_occ_grid(params=sim_params)

# Environment:
env = PointRobotEnv(occ_grid=occ_grid,
    Ts=params["general"]["Ts"],
    sim_params=sim_params
    )

env.visualize()

