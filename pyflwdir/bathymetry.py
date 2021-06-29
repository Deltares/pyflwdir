import numpy as np
from numba import njit
from scipy.integrate import solve_ivp


@njit
def froude(h, q, w, g=9.81):
    """Returns the froude number for a rectangular channel."""
    return q / (w * np.sqrt(g * h))


@njit
def dhdx_gfv(x, h, n, q, s, w, g=9.81, eps=1e-7):
    """Gradually varied flow differential equation assuming a rectangular cross section.
    NOTE Froude(h) should be < 1 (subcritical flow).
    """
    sf = (
        lambda h: n ** 2 * (q / (w * h)) ** 2 * ((w * h) / (2 * h + w)) ** (-4 / 3)
    )  # friction slope
    frh = min(1 - eps, froude(h, q, w, g=g))  # froude limited to subcritical flow
    dhdx = (s - sf(h)) / (1 - frh ** 2)  # gradually varied flow
    return -dhdx


def h_man(n, q, s, w):
    """Returns depth based on the manning formula assuming a rectangular cross section.
    NOTE: assumes uniform flow

    Parameters
    ----------
    n: float
        manning rougness [s.m^(-1/3)]
    q: float
        bankfull discharge [m.s-1]
    s: float
        slope [m.m-1]
    w: float
        width [m]

    Returns
    -------
    h: float
        flow depth
    """
    return ((n * q) / (np.sqrt(s) * w)) ** (3 / 5)


def h_gvf(h0, x0, x1, n, q, s, w, **kwargs):
    """Returns depth h1 at location x1 upstream from x0 with depth h0
    based on gradually varying flow theory.

    Solved with scipy.integrate.solve_ivp, by default using RK45 numerical scheme.

    Parameters
    ----------
    h0: float
        Initial water depth at x0
    x0, x1: float
        Position along stream of downstream and next upstream point.
        Note: x1 > x0
    n: float
        manning rougness [s.m^(-1/3)]
    q: float
        bankfull discharge [m.s-1]
    s: float
        slope [m.m-1]
    w: float
        width [m]

    Returns
    -------
    h1: float
        depth at x1
    """
    res = solve_ivp(
        dhdx_gfv, [x0, x1], h0, args=(n, q, s, w), t_eval=[x0, x1], **kwargs
    )
    return res.y[-1][-1]


def h_hdg(q, a=0.27, b=0.3):
    """Returns depth based on hydraulic downstream geometry theory: h = a*Q^b.
    Default values for a and b are based on Andreadis et al. 2013.
    """
    return a * q ** b
