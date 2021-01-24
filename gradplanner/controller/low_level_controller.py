import numpy as np


class LowLevelController:
    """Low level controller of a point mass robot with dynamics:

    x_{k+1} = x_k + v_k * Ts * cos(psi_k)
    y_{k+1} = y_k + v_k * Ts * sin(psi_k)
    v_{k+1} = v_k + Ts * a_k
    psi_{k+1} = psi_k + Ts * omega_k
    omega_{k+1} = omega_k + Ts * epsilon_k

    Where a_k and epsilon_k are the inputs and are the translational and rotational
    accelerations respectively.

    For now we assume, that it is a perfect controller which is able to produce
    the exact commanded outputs if they are reachable with the provided
    input constraints.
    """

    def __init__(self,
                 params):
        """Initializes a LowLevelController."""

        self._init_from_params(params)

    
    def get_inputs(self, state, cmd_vel):
        """produces control inputs based on the actual state and the commanded
        velocities in cmd_vel = np.array([v_des, omega_des])"""

        v_des = cmd_vel[0]
        omega_des = cmd_vel[1]
        v_k = state[2]
        omega_k = state[4]

        # translational acceleration:
        a_k = (v_des - v_k) / self._Ts
        if a_k > self._acc_max:
            a_k = self._acc_max
        elif a_k < self._acc_min:
            a_k = self._acc_min

        # angular acceleration:
        epsilon_k = (omega_des - omega_k) / self._Ts
        if epsilon_k > self._epsilon_max:
            a_epsilon_kk = self._epsilon_max
        elif epsilon_k < self._epsilon_min:
            epsilon_k = self._epsilon_min

        return np.array([a_k, epsilon_k])


    def _init_from_params(self, params):
        """Initializes some variables from the params."""

        self._Ts = params["genera"]["Ts"]
        self._acc_min = params["LowLevelController"]["acc_min"]
        self._acc_max = params["LowLevelController"]["acc_max"]
        self._epsilon_min = params["LowLevelController"]["epsilon_min"]
        self._epsilon_max = params["LowLevelController"]["epsilon_max"]
