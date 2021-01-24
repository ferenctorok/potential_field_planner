import numpy as np
import json
import time

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

# High level controller:
grad_controller = GradController(occupancy_grid=occ_grid,
    goal_pos=np.array([sim_params["goal"]["x"], sim_params["goal"]["y"]]),
    goal_ang=sim_params["goal"]["psi"],
    R=params["GradController"]["general"]["R"],
    params=params
    )

# Low level controller:
low_level_controller = LowLevelController(params=params)

# main simulation loop:
collision = False
state = env.get_state()
while (not grad_controller.goal_is_reached) and (not collision):
    start_time = time.time()

    # getting command velocities from the high level controller:
    pose = np.array([state[0], state[1], state[3]])
    cmd_vel = grad_controller.get_cmd_vel(pose)

    # getting inputs from the low level controller:
    u = low_level_controller.get_inputs(state, cmd_vel)

    # carrying out a timestep in the environment:
    state, collision = env.step(u)

    print("Control loop duration: {}".format(time.time() - start_time))
    print("state: {}".format(state))

    # visualizing:
    env.visualize()

plt.show()



