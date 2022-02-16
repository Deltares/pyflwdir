import numpy as np
from numba import njit
from scipy.integrate import solve_ivp

import logging

logger = logging.Logger(__name__)


@njit
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
    """Classifies estuaries based on width convergence.

    Parameters
    ----------
    rivdst, rivwth, elevtn : np.ndarray
        Distance to river outlet [m], river width [m], elevation [m+REF]
    max_elevtn : float, optional
        Maximum elevation for estuary outlet, by default 0 m+REF
    min_convergence : float, optional
        River width convergence threshold, by default 1e-2 m/m

    Returns
    -------
    np.ndarray of int8
        Estuary classification: >= 1 where estuary; 2 at upstream end of estaury.
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
    idxs_ds,
    seq,
    zs,
    rivdph,
    qbankfull,
    rivdst,
    rivwth,
    manning,
    min_rivslp=1e-5,
    min_rivdph=1,
    eps=1e-1,
    n_iter=2,
    logger=logger,
):
    # gradually varying flow solver for directed flw graph
    # NOTE: experimental!!
    def _gvf(x, h, n, q, s0, w, g=9.81, eps=eps):
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
