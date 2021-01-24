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
        elf._set_from_params(sim_params)


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
            self._robo_artist = Rectangle((self._state[0], self._state[1]),
                2, 1, angle=self.state[3], color='#EC66BA') 
            self._ax.add_artist(self._robo_artist)
            self._goal_artist = Rectangle((self._goal_x, self._goal_y),
                2, 1, angle=self._goal_psi, color="#37EC52")
            self._ax.add_artist(self._goal_artist)

        self._ax.cla()
        self._ax.add_artist(self._robo_artist)
        self._ax.add_artist(self._goal_artist)
        cmap = ListedColormap(['#240B3B', '#81BEF7'])
        self._ax.matshow(self._occ_grid, cmap=cmap)
        self._robo_artist.set_xy((self._state[0], self._state[1]))
        self._robo_artist.set_alpha(self._state[3])
        self._goal_artist.set_xy((self._goal_x[0], self._goal_psi[1]))
        self._goal_artist.set_alpha(self._goal_psi)

        plt.pause(self._Ts)


    def _set_from_params(self, params):
        """Sets some variables from the params."""

        self._goal_x = params["goal"]["x"]
        self._goal_y = params["goal"]["y"]
        self._goal_psi = params["goal"]["psi"]
