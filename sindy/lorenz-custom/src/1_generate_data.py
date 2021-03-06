"""Data generation step."""

import argparse
import logging
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
from derivative import dxdt
from scipy.integrate import solve_ivp

from common import BETA, DATA_DIR, RHO, SIGMA


def lorenz(_: float, u: np.ndarray, sigma: float, rho: float,
           beta: float) -> np.ndarray:
    """Returns a list containing values of the three functions of the Lorenz
    system.

    The Lorenz equations have constant coefficients (that don't depend on t),
    but we still receive t as the first parameter because that's how the
    integrator works.
    """
    x = u[0]
    y = u[1]
    z = u[2]
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z

    return np.hstack((dx_dt, dy_dt, dz_dt))


def generate_u(t: np.ndarray) -> np.ndarray:
    """Simulates observed data u with the help of an integrator for the Lorenz
    equations.
    """
    u0 = np.array([-8, 8, 27])
    result = solve_ivp(fun=lorenz,
                       t_span=(t[0], t[-1]),
                       y0=u0,
                       t_eval=t,
                       args=(SIGMA, RHO, BETA))
    u = result.y.T
    return u


def calculate_exact_derivatives(u: np.ndarray) -> np.ndarray:
    """Calculates the exact derivatives by using the Lorenz equations.

    Since we have the equations in this scenario, we can calculate the
    exact derivatives. This is easily done by simply plugging u into the
    equations, and getting du/dt back. In a real-world scenario, we don't
    have the equations -- that's what we're trying to discover with SINDy!
    """
    logging.info("Using exact derivatives.")
    n = u.shape[0]
    uprime = np.zeros_like(u)
    for i in range(n):
        uprime[i, :] = lorenz(None, u[i, :], SIGMA, RHO, BETA)

    return uprime


def calculate_finite_difference_derivatives(u: np.ndarray,
                                            t: np.ndarray) -> np.ndarray:
    """Calculates the derivatives of u using finite differences.

    Finite difference derivatives are quick and simple to calculate. They
    may not do as well as total variation derivatives at denoising
    derivatives in general, but in our simple non-noisy scenario, they
    do just as well.
    """
    logging.info("Using finite difference derivatives.")
    uprime = dxdt(u.T, t, kind="finite_difference", k=1).T
    return uprime


def calculate_total_variation_derivatives(u: np.ndarray,
                                          t: np.ndarray) -> np.ndarray:
    """Calculates the derivatives of u using the "total variation
    derivatives with regularization" technique.

    The paper recommends using the "total variation
    derivatives with regularization" technique, because it's effective
    at denoising the derivatives. This technique has been implemented in the
    "derivative" package, so we'll use this API.

    This takes several minutes to run.
    """
    logging.info("Using total variation derivatives.")
    uprime = dxdt(u.T, t, kind="trend_filtered", order=0, alpha=1e-2)
    return uprime


def generate_data() -> Tuple[np.ndarray, np.ndarray]:
    """ Generates data u, and calculates its derivatives uprime.
    """
    t0 = 0.001
    dt = 0.001
    tmax = 100
    n = int(tmax / dt)
    t = np.linspace(start=t0, stop=tmax, num=n)

    # Step 1: Generate data u.
    u = generate_u(t)

    # Step 2: Calculate u' from u.
    # uprime = calculate_exact_derivatives(u)
    uprime = calculate_finite_difference_derivatives(u, t)
    # uprime = calculate_total_variation_derivatives(u, t)

    return (u, uprime)


def main() -> None:
    logging.info("Generating data.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", dest="data_dir", default=DATA_DIR)
    args = parser.parse_args()
    data_dir = args.data_dir

    (u, uprime) = generate_data()

    Path(data_dir).mkdir(exist_ok=True)
    data_file_path = Path(data_dir, "data.hdf5")
    with h5py.File(data_file_path, "w") as file:
        file.create_dataset(name="u", data=u)
        file.create_dataset(name="uprime", data=uprime)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
