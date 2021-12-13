"""Data generation step."""

import logging
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
from scipy.integrate import solve_ivp

from common import DATA_DIR, get_absolute_dir


def lorenz(_: float, u: np.ndarray, sigma: float, rho: float,
           beta: float) -> np.ndarray:
    """Returns a list containing the three functions of the Lorenz equation.

    The Lorenz equations have constant coefficients (that don't depend on t),
    but we still receive t as the first parameter because that's how the
    integrator works.
    """
    x = u[0]
    y = u[1]
    z = u[2]
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z

    return np.hstack((dxdt, dydt, dzdt))


def generate_data() -> Tuple[np.ndarray, np.ndarray]:
    """Simulates observed data u with the help of an integrator for the Lorenz
    equations. Then, calculates the derivatives of u by simply passing u as
    a parameter to the Lorenz ODE.
    """
    sigma = 10
    rho = 28
    beta = 8 / 3
    t0 = 0.001
    dt = 0.001
    tmax = 100
    n = int(tmax / dt + 1)

    # Step 1: Generate data u.
    u0 = np.array([-8, 8, 27])
    t_eval = np.linspace(start=t0, stop=tmax, num=n)
    result = solve_ivp(fun=lorenz,
                       t_span=(t0, tmax),
                       y0=u0,
                       t_eval=t_eval,
                       args=(sigma, rho, beta))
    u = result.y.T

    # Step 2: Calculate u' from u.
    uprime = np.zeros_like(u)
    for i in range(n):
        uprime[i, :] = lorenz(None, u[i, :], sigma, rho, beta)
    # Adding some noise to the derivatives.
    # uprime = uprime + np.random.normal(size=uprime.shape)

    return (u, uprime)


def main() -> None:
    logging.info("Processing data.")

    (u, uprime) = generate_data()
    data_dir_path = get_absolute_dir(DATA_DIR, True)

    data_file_path = Path(data_dir_path, "data.hdf5")
    with h5py.File(data_file_path, "w") as file:
        file.create_dataset(name="u", data=u)
        file.create_dataset(name="uprime", data=uprime)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
