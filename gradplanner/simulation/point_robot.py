import numpy as np


class PointRobot:
    """class for modeling the kinematical behavior of the point-mass robot.
    The kinematical model of the robot is as follows.
    
    x_{k+1} = x_k + v_k * Ts * cos(psi_k)
    y_{k+1} = y_k + v_k * Ts * sin(psi_k)
    v_{k+1} = v_k + Ts * a_k
    psi_{k+1} = psi_k + Ts * omega_k
    omega_{k+1} = omega_k + Ts * epsilon_k
    """

    def __init__(self,
                 Ts,
                 params):
        """Initializes a PointRobot."""

        self._Ts = Ts
        self._init_from_params(params)

    
    def step(self, u):
        """carries out a timestep and returns the resulting state."""

        a, epsilon = u[0], u[1]

        self._x = self._x + self._v * self._Ts * np.cos(self._psi)
        self._y = self._y + self._v * self._Ts * np.sin(self._psi)
        self._v = self._v + self._Ts * a
        self._psi = self._psi + self._Ts * self._omega
        self._omega = self._omega + self._Ts * epsilon

        # limiting psi between pi and -pi
        if self._psi > np.pi:
            self._psi -= (2 * np.pi)
        elif self._psi < -np.pi:
            self._psi += (2 * np.pi)

        return np.array([self._x, self._y, self._v, self._psi, self._omega])


    def set_state(self, state):
        """Sets the state of the robot."""

        self._x = state[0]
        self._y = state[1]
        self._v = state[2]
        self._psi = state[3]
        self._omega = state[4]


    def get_state(self):
        """Returns the actual state."""
        return np.array([self._x, self._y, self._v, self._psi, self._omega])


    def _init_from_params(self, params):
        """Initializes some variables from the params.""" 

        self._x = params["init_state"]["x"]
        self._y = params["init_state"]["y"]
        self._v = params["init_state"]["v"]
        self._psi = params["init_state"]["psi"]
        self._omega = params["init_state"]["omega"]
