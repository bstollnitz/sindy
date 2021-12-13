"""SINDy implementation for Lorenz equations."""

import logging

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from typing import Tuple


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


def create_library(u: np.ndarray, polynomial_order: int,
                   use_trig: bool) -> np.ndarray:
    """Creates a matrix containing a library of potential equation terms.

    For example, if our u depends on x, y, and z, and we specify
    polynomial_order=2 and use_trig=false, our terms would be:
    1, x, y, z, x^2, y^2, z^2, xy, yz, xz.
    """
    (m, n) = u.shape
    theta = np.ones((m, 1))

    # Polynomials of order 1.
    theta = np.hstack((theta, u))

    # Polynomials of order 2.
    if polynomial_order >= 2:
        for i in range(n):
            for j in range(i, n):
                theta = np.hstack((theta, u[:, i:i + 1] * u[:, j:j + 1]))

    # Polynomials of order 3.
    if polynomial_order >= 3:
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    theta = np.hstack(
                        (theta, u[:, i:i + 1] * u[:, j:j + 1] * u[:, k:k + 1]))

    # Polynomials of order 4.
    if polynomial_order >= 4:
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    for l in range(k, n):
                        theta = np.hstack(
                            (theta, u[:, i:i + 1] * u[:, j:j + 1] *
                             u[:, k:k + 1] * u[:, l:l + 1]))

    # Polynomials of order 5.
    if polynomial_order >= 5:
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    for l in range(k, n):
                        for m in range(l, n):
                            theta = np.hstack(
                                (theta, u[:, i:i + 1] * u[:, j:j + 1] *
                                 u[:, k:k + 1] * u[:, l:l + 1] * u[:, m:m + 1]))

    if use_trig:
        for i in range(1, 11):
            theta = np.hstack((theta, np.sin(i * u), np.cos(i * u)))

    return theta


def calculate_regression(theta: np.ndarray, uprime: np.ndarray,
                         lambda_value: float) -> np.ndarray:
    """Finds a xi matrix that fits theta * xi = uprime, using the sequential
    threshdolded least-squares algorithm, which is a regression algorithm that
    promotes sparsity.

    The authors of the SINDy paper designed this algorithm as an alternative
    to LASSO, because they found LASSO to be unstable algorithmically, and
    computationally expensive for very large data sets.
    """
    # Solve Ax = b, theta * xi = uprime.
    xi = np.linalg.lstsq(theta, uprime, rcond=None)[0]
    n = xi.shape[1]
    iteration_count = 10

    # Perform regression.
    for _ in range(iteration_count):
        small_indices = np.abs(xi) < lambda_value
        xi[small_indices] = 0
        for j in range(n):
            big_indices = np.logical_not(small_indices[:, j])
            xi[big_indices, j] = np.linalg.lstsq(theta[:, big_indices],
                                                 uprime[:, j],
                                                 rcond=None)[0]

    return xi


def lorenz_approximation(_: float, u: np.ndarray, xi: np.ndarray,
                         polynomial_order: int, use_trig: bool) -> np.ndarray:
    """For each 1 x 3 u vector, this function calculates the corresponding
    1 x n row of the theta matrix, and multiples that theta row by the n x 3 xi.
    The result is the corresponding 1 x 3 du/dt vector."""
    theta = create_library(u.reshape((1, 3)), polynomial_order, use_trig)
    return theta @ xi


def compute_trajectory(xi: np.ndarray, polynomial_order: int,
                       use_trig: bool) -> np.ndarray:
    """Calculates the trajectory of the model discovered by SINDy.

    Given an initial value for u, we call the lorenz_approximation function
    to calculate the corresponding du/dt. Then, using a
    numerical method, solve_ivp uses that derivative to calculate the next
    value of u, using the xi matrix discovered by SINDy.
    This process repeats, until we have all values of the u
    trajectory.
    """
    t0 = 0.001
    dt = 0.001
    tmax = 100
    n = int(tmax / dt + 1)

    u0 = np.array([-8, 8, 27])
    t_eval = np.linspace(start=t0, stop=tmax, num=n)
    result = solve_ivp(fun=lorenz_approximation,
                       t_span=(t0, tmax),
                       y0=u0,
                       t_eval=t_eval,
                       args=(xi, polynomial_order, use_trig))
    u = result.y.T

    return u


def style_axis(axis):
    """Styles a graph's x, y, or z axis."""
    # pylint: disable=protected-access
    axis._axinfo["grid"]["color"] = "#dddddd"
    axis._axinfo["grid"]["linewidth"] = 0.4
    axis._axinfo["tick"]["linewidth"][True] = 0.4
    axis._axinfo["tick"]["linewidth"][False] = 0.4
    axis.set_pane_color((0.98, 0.98, 0.98, 1.0))
    axis.line.set_color("#bbbbbb")
    axis.label.set_color("#333333")
    pass


def style_axis3d(axis3d):
    """Styles a 3D graph."""
    axis3d.set_xlabel("x")
    axis3d.set_ylabel("y")
    axis3d.set_zlabel("z")
    axis3d.tick_params(axis="x", colors="#666666")
    axis3d.tick_params(axis="y", colors="#666666")
    axis3d.tick_params(axis="z", colors="#666666")
    style_axis(axis3d.w_xaxis)
    style_axis(axis3d.w_yaxis)
    style_axis(axis3d.w_zaxis)
    axis3d.set_title(axis3d.get_title(), fontdict={"color": "#333333"})


def graph_results(u: np.ndarray, u_approximation: np.ndarray) -> None:
    """Graphs the original trajectory, and the one computed from the model
    discovered by SINDy."""
    sample_count = 20000
    orange = "#EF6C00"

    fig = plt.figure(figsize=plt.figaspect(0.5))

    # Graph trajectory from the true model.
    axis3d = fig.add_subplot(1, 2, 1, projection="3d")
    x = u[0:sample_count, 0]
    y = u[0:sample_count, 1]
    z = u[0:sample_count, 2]
    axis3d.plot3D(x, y, z, orange, linewidth=0.4)
    axis3d.set_title("Original Lorenz trajectory")
    style_axis3d(axis3d)

    # Graph trajectory computed from model discovered by SINDy.
    axis3d = fig.add_subplot(1, 2, 2, projection="3d")
    x = u_approximation[0:sample_count, 0]
    y = u_approximation[0:sample_count, 1]
    z = u_approximation[0:sample_count, 2]
    axis3d.plot3D(x, y, z, orange, linewidth=0.4)
    axis3d.set_title("SINDy approximation")
    style_axis3d(axis3d)

    plt.show()


def main() -> None:
    polynomial_order = 2
    use_trig = False
    (u, uprime) = generate_data()
    theta = create_library(u, polynomial_order, use_trig)
    xi = calculate_regression(theta, uprime, lambda_value=0.025)
    u_approximation = compute_trajectory(xi, polynomial_order, use_trig)
    graph_results(u, u_approximation)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
