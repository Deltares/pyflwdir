import numpy as np
from numba import njit
from scipy.integrate import solve_ivp

import logging

logger = logging.Logger(__name__)


@njit(cache=True)
def classify_estuary(
    idxs_ds: np.ndarray,
    seq: np.ndarray,
    idxs_pit: np.ndarray,
    rivdst: np.ndarray,
    rivwth: np.ndarray,
    elevtn: np.ndarray,
    max_elevtn: float = 0,
    min_convergence: float = 1e-2,
) -> np.ndarray:
    """Classify estuaries based on river-width convergence.

    Parameters
    ----------
    idxs_ds : 1D array of int
        Linear index of the next downstream node.
    seq : 1D array of int
        Valid node indices ordered from downstream to upstream.
    idxs_pit : 1D array of int
        Linear indices of pit nodes.
    rivdst : 1D array of float
        Distance-to-outlet values [m].
    rivwth : 1D array of float
        River-width values [m].
    elevtn : 1D array of float
        Elevation values [m + reference elevation].
    max_elevtn : float, optional
        Maximum elevation for estuary outlets [m + reference elevation], by default 0.
    min_convergence : float, optional
        Minimum river-width convergence threshold [m/m], by default 1e-2.

    Returns
    -------
    np.ndarray of int8
        Estuary classification: 1 for estuary nodes, 2 at the upstream end of an
        estuary, and 0 elsewhere.
    """
    estuary = np.zeros(idxs_ds.size, np.int8)
    idxs0 = idxs_pit[elevtn[idxs_pit] <= max_elevtn]
    estuary[idxs0] = 1
    for idx in seq:  # down- to upstream
        idx_ds = idxs_ds[idx]
        if estuary[idx_ds] == 0 or idx == idx_ds:
            continue
        dx = rivdst[idx] - rivdst[idx_ds]
        dw = rivwth[idx_ds] - rivwth[idx]
        if (rivdst[idx_ds] == 0 and dw <= 0) or (dx > 0 and dw / dx > min_convergence):
            estuary[idx] = 1
        else:
            estuary[idx_ds] = 2  # most upstream estuary link
    return estuary


def rivdph_gvf(
    idxs_ds: np.ndarray,
    seq: np.ndarray,
    zs: np.ndarray,
    rivdph: np.ndarray,
    qbankfull: np.ndarray,
    rivdst: np.ndarray,
    rivwth: np.ndarray,
    manning: np.ndarray,
    min_rivslp: float = 1e-5,
    min_rivdph: float = 1,
    eps: float = 1e-1,
    n_iter: int = 2,
    logger: logging.Logger = logger,
) -> np.ndarray:
    """Estimate river depth with a gradually varied-flow solver.

    This experimental solver integrates the gradually varied-flow equation along the
    directed river network and iteratively updates water depths.

    Parameters
    ----------
    idxs_ds : 1D array of int
        Linear index of the next downstream node.
    seq : 1D array of int
        Valid node indices ordered from downstream to upstream.
    zs : 1D array of float
        Water-surface elevation values [m].
    rivdph : 1D array of float
        Initial river-depth values [m].
    qbankfull : 1D array of float
        Bankfull discharge values [m3/s].
    rivdst : 1D array of float
        Distance-to-outlet values [m].
    rivwth : 1D array of float
        River-width values [m].
    manning : 1D array of float
        Manning roughness values [s/m^(1/3)].
    min_rivslp : float, optional
        Minimum slope used by the solver, by default 1e-5.
    min_rivdph : float, optional
        Minimum output depth [m], by default 1.
    eps : float, optional
        Minimum depth used while evaluating the flow equation, by default 0.1.
    n_iter : int, optional
        Number of depth and bed-elevation update passes, by default 2.
    logger : logging.Logger, optional
        Logger used to report integration failures.

    Returns
    -------
    1D array of float
        Updated river-depth values [m].
    """

    # gradually varying flow solver for directed flw graph
    # NOTE: experimental!!
    def _gvf(
        x: float,
        h: float,
        n: float,
        q: float,
        s0: float,
        w: float,
        g: float = 9.81,
        eps: float = eps,
    ) -> float:
        h = max(h, eps)
        sf = lambda h: n**2 * (q / (w * h)) ** 2 * ((w * h) / (2 * h + w)) ** (-4 / 3)
        fr = lambda h: q / (w * np.sqrt(g * h))
        dhdx = (s0 - sf(h)) / (1 - fr(h) ** 2)
        return -dhdx

    rivdph_out = rivdph.copy()
    # initial bed levels
    zb = zs - rivdph
    for _ in range(n_iter):
        for idx in seq:  # from down- to upstream
            idx_ds = idxs_ds[idx]
            if qbankfull[idx] <= 0 or rivwth[idx] <= 0 or idx == idx_ds:  # pit
                continue
            dz = zb[idx] - zb[idx_ds]
            dx = rivdst[idx] - rivdst[idx_ds]
            # FIXME force a positive slp for stable solutions
            slp = max(min_rivslp, dz / dx)
            # print(np.round(dz/dx,8), np.round(slp,8))
            h0 = rivdph_out[idx_ds]
            args = (manning[idx], qbankfull[idx], slp, rivwth[idx])
            # solve riv depth for single node with RK45 numerical integration
            sol = solve_ivp(_gvf, [0, dx], [h0], method="RK45", args=args)
            h1 = sol.y[-1][-1]
            if abs((h1 - h0) / dx) > 1 or h1 < 0 or not sol.success:
                logger.warning(sol.message)
            else:
                rivdph_out[idx] = max(min_rivdph, h1)
        # update bed levels
        zb = zs - rivdph_out
    return rivdph_out
