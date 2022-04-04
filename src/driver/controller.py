import numpy as np
from dataclasses import dataclass
from dataclasses import field

@dataclass
class PIDController:
    """
    Generalized vector-based PID controller with anti-windup protection
    Note: e = x - xdes and u = Ke. You can specify K to be negative yourself
    TODO: if needed, make dt possible to specify
    """
    Kp: np.array
    Kd: np.array
    Ki: np.array
    dt: float
    Emin: np.array
    Emax: np.array
    m: int  # number of states in error vector
    E: np.array = field(init=False)

    def __post_init__(self):
        # make sure that all gain matrices have appropriate length
        self.Kp = np.atleast_2d(self.Kp)
        self.Kd = np.atleast_2d(self.Kd)
        self.Ki = np.atleast_2d(self.Ki)

        self.E = np.zeros(self.m)  # set error integral to start at 0
        try:
            assert (self.m == np.atleast_2d(self.Kp).shape[1])
            assert (self.m == np.atleast_2d(self.Kd).shape[1])
            assert (self.m == np.atleast_2d(self.Ki).shape[1])
            print(self.m, np.atleast_1d(self.Emin))
            assert (self.m == np.atleast_1d(self.Emin).shape[0])
            assert (self.m == np.atleast_1d(self.Emax).shape[0])

        except ValueError:
            print("Dimension of gain matrices needs to match m")
            raise



    def feedback(self, e: np.array, edot: np.array) -> \
            np.array:
        e = np.atleast_1d(e)
        edot = np.atleast_1d(edot)
        u = self.Kp@e + self.dt*(self.Kd@edot) + self.Ki@self.E
        self.integrate(e)
        return u

    def integrate(self, e):
        # if ((self.E-self.Emax) >= 0).any():
        #     log.warning(f"Error integral maximum limit! {self.E}")
        # if ((self.E-self.Emin) <= 0).any():
        #     log.warning(f"Error integral at minimum limit! {self.E}")
        self.E = np.clip(self.E+e*self.dt, self.Emin, self.Emax)
        return self.E

    def reset_error(self):
        self.E = np.zeros(self.E.shape)