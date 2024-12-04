import abc
import numpy as np


class OneDimPotential(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, x):
        pass

    def derivative(self, x):
        raise NotImplementedError()


class HOPotential(OneDimPotential):
    def __init__(self, omega):
        self.omega = omega

    def __call__(self, x):
        return 0.5 * self.omega**2 * x**2

    def derivative(self, x):
        return self.omega**2 * x


class DWPotential(HOPotential):
    def __init__(self, omega, l):
        super().__init__(omega)
        self.l = l

    def __call__(self, x):
        return super().__call__(x) + 0.5 * self.omega**2 * (
            0.25 * self.l**2 - self.l * abs(x)
        )

    def derivative(self, x):
        """
        Uses Heaviside function to avoid division by zero. Is ill defined in x=0
        anyways
        """
        return super().derivative(x) - self.l * self.omega**2 * (
            np.heaviside(x, 0.5) - 0.5
        )


class DWPotentialSmooth(OneDimPotential):
    """
    This is the double-well potential used by J. Kryvi and S. Bøe in their
    thesis work. See Eq. [13.11] in Bøe: https://www.duo.uio.no/handle/10852/37170
    """

    def __init__(self, a=4):
        self.a = a

    def __call__(self, x):
        return (
            (1.0 / (2 * self.a**2))
            * (x + 0.5 * self.a) ** 2
            * (x - 0.5 * self.a) ** 2
        )

    def derivative(self, x):
        a = self.a
        return (
            1
            / a**2
            * (
                (x + 0.5 * a) * (x - 0.5 * a) ** 2
                + (x - 0.5 * a) * (x + 0.5 * a) ** 2
            )
        )


class SymmetricDWPotential(OneDimPotential):
    """
    This is a generalization of the symmetric double-well potential
    used by Wadehra et. al: https://aip.scitation.org/doi/10.1063/1.1589481
    """

    def __init__(self, a=0.5, b=1, c=-7):
        self.a = a
        self.b = b
        self.c = c

    def __call__(self, x):
        return self.a * x**6 + self.b * x**4 + self.c * x**2

    def derivative(self, x):
        return 6 * self.a * x**5 + 3 * self.b * x**3 + 2 * self.c * x


class AsymmetricDWPotential(OneDimPotential):
    """
    This is a generalization of the asymmetric double-well potential
    used by Wadehra et. al: https://aip.scitation.org/doi/10.1063/1.1589481
    """

    def __init__(self, a=1, b=1, c=-2.5):
        self.a = a
        self.b = b
        self.c = c

    def __call__(self, x):
        return self.a * x**4 + self.b * x**3 + self.c * x**2

    def derivative(self, x):
        return 4 * self.a * x**3 + 3 * self.b * x**2 + 2 * self.c * x


class GaussianPotential(OneDimPotential):
    def __init__(self, weight, center, deviation, np):
        self.weight = weight
        self.center = center
        self.deviation = deviation
        self.np = np

    def __call__(self, x):
        return -self.weight * self.np.exp(
            -((x - self.center) ** 2) / (2.0 * self.deviation**2)
        )

    def derivative(self, x):
        return -(x - self.center) / self.deviation**2 * self(x)


class GaussianPotentialHardWall(OneDimPotential):
    """
    A hard wall is introduced in order to force the
    higher lying states (unbound states) to go to zero at the end of the grid.
    """

    def __init__(self, weight, center, deviation, x_wall):
        self.weight = weight
        self.center = center
        self.deviation = deviation
        self.x_wall = x_wall

    def __call__(self, x):
        wall = np.zeros(len(x))
        for i in range(len(x)):
            if abs(x[i]) > self.x_wall:
                wall[i] = 1e5

        return (
            -self.weight
            * np.exp(-((x - self.center) ** 2) / (2.0 * self.deviation**2))
            + wall
        )


class AtomicPotential(OneDimPotential):
    def __init__(self, Za=2, c=0.54878464):
        self.Za = Za
        self.c = c

    def __call__(self, x):
        return -self.Za / np.sqrt(x**2 + self.c)

    def derivative(self, x):
        return self.Za * x / (x**2 + self.c) ** (3 / 2)


class MorsePotentialDW(OneDimPotential):
    """
    A double-well potential based on Morse potentials
    used by Igeh, Jonny A. in his thesis work.
    Morse potential based on: https://doi.org/10.1103/PhysRev.34.57, 
    but the zero-point has been moved to prevent multiple (unecessary) evaluations of the exponential function.

    params:
    D_a: Depth of the left well
    D_b: Depth of the right well
    a: Width of the left well
    b: Width of the right well
    d: Distance between center of the wells
    """
    def __init__(self, D_a=20, D_b=20, k_a=1.0, k_b=1.0, d=4):
        self.D_a = D_a 
        self.D_b = D_b
        self.a = np.sqrt(k_a / (2 * D_a))
        self.b = np.sqrt(k_b / (2 * D_b))
        self.d = d
        self.x_a = -d/2
        self.x_b = d/2
        self.left_pot = lambda x: self.D_a * (1 - np.exp(-self.a * (x - self.x_a)))**2
        self.right_pot = lambda x: + self.D_b * (1 - np.exp(self.b * (x - self.x_b)))**2

    def __call__(self, x):
        return (
            self.D_a * (1 - np.exp(-self.a * (x - self.x_a)))**2
             + self.D_b * (1 - np.exp(self.b * (x - self.x_b)))**2
        )
    
    def derivative(self, x):
        return (
            2 * self.D_a * self.a * (1 - np.exp(-self.a * (x - self.x_a))) * np.exp(-self.a * (x - self.x_a))
            - 2 * self.D_b * self.b * (1 - np.exp(self.b * (x - self.x_b))) * np.exp(self.b * (x - self.x_b))
        )