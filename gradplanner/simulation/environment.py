import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from gradplanner.simulation.point_robot import PointRobot


class PointRobotEnv:
    """Class for the environment of a pointrobot."""

    def __init__(self,
                 occ_grid,
                 Ts,
                 sim_params):
        """Initializes the PointRobotEnv."""

        self._occ_grid = occ_grid
        self._Ts = Ts
        self._fig = None
        self._state = np.zeros(5)
        self._robot = PointRobot(Ts=Ts, params=sim_params)
        self._set_from_params(sim_params)


    def step(self, u):
        """Carries out a step in the environment."""

        state = self._robot.step(u)
        collision = False

        # check for collision:
        i, j = int(np.floor(state[0])), int(np.floor(state[1]))
        if self._occ_grid[i, j] == 1:
            state = np.array([state[0], state[1], 0, state[3], 0])
            self._robot.set_state(state)
            collision = True

        self._state = state

        return state, collision


    def visualize(self):
        """Visualizes the environment with the robot."""

        if self._fig is None:
            self._fig = plt.figure(1)
            self._ax = plt.gca()
            plt.xlabel("y")
            plt.ylabel("x")
        
        # clearing the axis:
        self._ax.cla()
        
        # setting the rectangles of the robot and the goal:
        self._robo_patch = Rectangle((self._state[1], self._state[0]),
            2, 1, angle= -np.rad2deg(self._state[3] - np.pi / 2), color='r') 
        self._ax.add_patch(self._robo_patch)

        self._goal_patch = Rectangle((self._goal_y, self._goal_x),
            2, 1, angle= -np.rad2deg(self._goal_psi - np.pi / 2))
        self._ax.add_patch(self._goal_patch)

        # plotting the occupancy_grid:
        self._ax.matshow(self._occ_grid)

        plt.pause(self._Ts)


    def get_state(self):
        """Returns the actual state."""
        return self._robot.get_state()


    def _set_from_params(self, params):
        """Sets some variables from the params."""

        self._goal_x = params["goal"]["x"]
        self._goal_y = params["goal"]["y"]
        self._goal_psi = params["goal"]["psi"]
