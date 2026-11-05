# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Nonlinear Least Squares minimization with box bounds."""

import abc
import dataclasses
import enum
import time
from typing import Callable, List, Optional, Sequence, TextIO, Tuple, Union

import mujoco
import numpy as np


class Verbosity(enum.Enum):
  SILENT = 0
  FINAL = 1
  ITER = 2
  FULLITER = 3


class Status(enum.Enum):
  FACTORIZATION_FAILED = enum.auto()
  NO_IMPROVEMENT = enum.auto()
  MAX_ITER = enum.auto()
  DX_TOL = enum.auto()
  G_TOL = enum.auto()


_STATUS_MESSAGE = {
    Status.FACTORIZATION_FAILED: 'factorization failed.',
    Status.NO_IMPROVEMENT: 'insufficient reduction.',
    Status.MAX_ITER: 'maximum iterations reached.',
    Status.DX_TOL: 'norm(dx) < tol.',
    Status.G_TOL: 'norm(gradient) < tol.',
}


@dataclasses.dataclass(frozen=True)
class IterLog:
  """Log of a single iteration of the non-linear least-squares solver.

  Attributes:
    candidate: Value of the decision variable at the beginning this iteration.
    objective: Value of the objective at the candidate.
    reduction: Reduction of the objective during this iteration.
    regularizer: Value of the regularizer used for this iteration.
    residual: Optional value of the residual at the candidate.
    jacobian: Optional value of the Jacobian at the candidate.
    grad: Optional value of the gradient at the candidate.
    step: Optional change in decision variable during this iteration.
  """

  candidate: np.ndarray
  objective: np.float64
  reduction: np.float64
  regularizer: np.float64
  residual: Optional[np.ndarray] = None
  jacobian: Optional[np.ndarray] = None
  grad: Optional[np.ndarray] = None
  step: Optional[np.ndarray] = None


class Norm(abc.ABC):
  """Abstract interface for norm functions, measuring the magnitude of vectors.

  Key Concepts:

  * Norm Value: The value of the norm for a given input vector.
  * Gradient and Hessian: The gradient (first derivative) and Hessian (second
    derivative) of the norm function with respect to the input vector.

  Subclasses Must Implement:

  * `value(self, r: np.ndarray)`: Computes and returns the norm value for the
     input vector `r`.
  * `grad_hess(self, r: np.ndarray, proj: np.ndarray)`: Computes and returns
     both  the gradient and Hessian of the norm at `r`, projected onto `proj`.
     The reason we ask the user to perform the projection themselves is that
     norm Hessians are often large and sparse, and the "sandwich" projection
     operator `proj.T @ hess @ proj` can be computed efficiently by taking the
     specific norm structure into account.
  """

  @abc.abstractmethod
  def value(self, r: np.ndarray) -> np.float64:
    """Returns the value of the norm at the input vector `y = norm(r)`."""
    pass

  @abc.abstractmethod
  def grad_hess(self, r: np.ndarray, proj: np.ndarray):
    """Computes the projected gradient and Hessian of the norm at `r`.

    Args:
        r: A NumPy column vector (nr x 1).
        proj: A pre-computed projection matrix (nr x nx).

    Returns:
        A tuple containing:
            * Projected gradient: proj.T @ (d_norm/d_r).
            * Projected Hessian: proj.T @ (d^2_norm/d_r^2) @ proj.
    """
    pass


class Quadratic(Norm):
  """Implementation of the quadratic norm."""

  def value(self, r: np.ndarray):
    """Returns the quadratic norm of `r`."""
    return 0.5 * (r.T @ r).item()

  def grad_hess(self, r: np.ndarray, proj: np.ndarray):
    """Computes the projected gradient and Hessian of the quadratic norm at `r`.

    Args:
        r: A NumPy column vector (nr x 1).
        proj: A pre-computed projection matrix (nr x nx).

    Returns:
        A tuple containing:
            * Projected gradient: `proj.T @ r`.
            * Projected Hessian: `proj.T @ proj`.
    """
    grad = proj.T @ r
    hess = proj.T @ proj  # Notionally proj.T @ np.eye(r.size) @ proj
    return grad, hess


def least_squares(
    x0: np.ndarray,
    residual: Callable[[np.ndarray], np.ndarray],
    bounds: Optional[Sequence[np.ndarray]] = None,
    jacobian: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
    norm: Norm = Quadratic(),
    eps: float = np.finfo(np.float64).eps ** 0.5,
    mu_min: float = 1e-6,
    mu_max: float = 1e8,
    mu_factor: float = 10.0 ** 0.1,
    xtol: float = 1e-8,
    gtol: float = 1e-8,
    max_iter: int = 100,
    verbose: Union[Verbosity, int] = Verbosity.ITER,
    output: Optional[TextIO] = None,
    iter_callback: Optional[Callable[[List[IterLog]], None]] = None,
    check_derivatives: bool = False,
    x_scale: Optional[Union[float, np.ndarray, str]] = None,
) -> Tuple[np.ndarray, List[IterLog]]:
  """Nonlinear Least Squares minimization with box bounds.

  Args:
    x0: Initial guess
    residual: Vectorized function returning the residual for 1 or more points.
    bounds: Optional pair of lower and upper bounds on the solution.
    jacobian: Optional function that returns Jacobian of the residual at a given
      point and residual. If not given, `residual` will be finite-differenced.
    norm: Norm object returning norm scalar or its projected gradient and
      Hessian. See Norm class for detailed documentation.
    eps: Perurbation used for automatic finite-differencing.
    mu_min: Minimum value of the regularizer.
    mu_max: Maximum value of the regularizer.
    mu_factor: Factor for increasing or decreasing the regularizer.
    xtol: Termination tolerance on relative step size.
    gtol: Termination tolerance on gradient norm.
    max_iter: Maximum number of iterations.
    verbose: Verbosity level.
    output: Optional file or StringIO to which to print messages.
    iter_callback: Optional iteration callback, takes trace argument.
    check_derivatives: Compare user-defined Jacobian and norm against fin-diff.
    x_scale: Per-parameter scaling (matches scipy's x_scale). Setting
      x_scale=D solves the problem in the change of variables z = x / D
      and un-scales the result. None (default) or 1.0 is no scaling.
      'jac' sets D_i = 1 / ||J(:, i)|| at each iteration. An array of
      shape (n,) (or a positive scalar) is used as D directly. Note
      that mu, mu_min and mu_max then act on the scaled subproblem.

  Returns:
    x: best solution found
    trace: sequence of solution iterates.
  """
  t_start_total = time.time()

  # Convert verbosity to int.
  verbose = Verbosity(verbose).value

  # Constant for Armijo's sufficient-reduction rule
  armijo_c1 = 1e-2

  # Initialize locals.
  status = Status.MAX_ITER
  i = 0
  n = x0.size
  x = x0.astype(np.float64).reshape((n, 1))
  xnew = np.zeros((n, 1))
  dx = np.zeros((n, 1))
  scratch = np.zeros((n, n + 7))
  eps = np.float64(eps)
  mu = np.float64(0.0)  # Optimistically start with no regularization.
  n_reduc = 0  # Number of sequential mu reductions.

  # Resolve x_scale -> D of shape (n, 1). For 'jac', D is refreshed each iter;
  # None means no scaling (D = 1).
  adaptive_scale = isinstance(x_scale, str)
  if adaptive_scale and x_scale != 'jac':
    raise ValueError(
        f"x_scale must be None, 'jac', a positive scalar, or array, got "
        f'{x_scale!r}.'
    )
  if x_scale is None or adaptive_scale:
    D = np.ones((n, 1))
  else:
    D = np.asarray(x_scale, dtype=np.float64)
    if D.shape not in ((), (n,)):
      raise ValueError(f'x_scale array must have shape ({n},), got {D.shape}.')
    if not np.all(np.isfinite(D)) or np.any(D <= 0):
      raise ValueError('x_scale must be positive and finite.')
    D = (np.ones(n) * D).reshape(n, 1)

  # Initialize logging.
  trace = []
  n_res = 0
  n_jac = 0
  t_res = 0.0
  t_jac = 0.0

  if mu_factor <= 1:
    raise ValueError('mu_factor must be > 1.')

  # Decrease mu agressively: sequential decreases grow exponentially.
  def decrease_mu(mu, n_reduc):
    dmu = (1 / mu_factor) ** (2**n_reduc)
    mu = 0.0 if mu * dmu < mu_min else mu * dmu
    n_reduc += 1
    return mu, n_reduc

  # Increase mu carefully: always increase by mu_factor.
  def increase_mu(mu):
    mu = max(mu_min, mu_factor * mu)
    n_reduc = 0  # Reset n_reduc.
    return mu, n_reduc

  # Make local copy of bounds to avoid reshaping user input.
  bounds = None if bounds is None else list(bounds)
  if bounds is not None:
    # Checks bounds.
    if len(bounds) != 2:
      raise ValueError('bounds must have 2 elements.')
    if bounds[0].size != n or bounds[1].size != n:
      raise ValueError('bounds must have the same size as x0.')
    if not np.all(np.isfinite(bounds[0])) or not np.all(np.isfinite(bounds[1])):
      raise ValueError('bounds must be finite.')
    if not np.all(bounds[0] < bounds[1]):
      raise ValueError('bounds[0] must be smaller than bounds[1].')

    # Reshape and clip.
    bounds[0] = bounds[0].reshape(n, 1)
    bounds[1] = bounds[1].reshape(n, 1)
    np.clip(x, bounds[0], bounds[1], out=x)

  # Check for NaNs.
  if not np.all(np.isfinite(x)):
    raise ValueError('x0 must be finite.')

  # Get initial residual.
  t_start = time.time()
  r = residual(x)
  rnew = r
  t_res += time.time() - t_start
  n_res += 1

  if r.dtype != np.float64:
    raise ValueError('residual function must return float64 arrays.')

  # Minimize.
  for i in range(max_iter):
    if status != Status.MAX_ITER:
      break

    # Get objective y.
    y = norm.value(r)

    # Get Jacobian jac.
    t_start = time.time()
    if jacobian is None:
      jac, n_res = jacobian_fd(residual, x, r, eps, n_res, bounds)
      t_res += time.time() - t_start
    else:
      jac = jacobian(x, r)
      t_jac += time.time() - t_start
      n_jac += 1

      # Check user-provided Jacobian
      if i == 0 and check_derivatives:
        n_res = check_jacobian(residual, x, r, jac, eps, n_res, bounds, output)

    # Check user-provided norm
    if i == 0 and check_derivatives and not isinstance(norm, Quadratic):
      check_norm(r, norm, eps, output)

    # Refresh D for adaptive ('jac') scaling: column-norm preconditioner.
    if adaptive_scale:
      col_norms = np.linalg.norm(jac, axis=0)
      D = (1.0 / np.maximum(col_norms, eps)).reshape(n, 1)

    # Gradient/Hessian in scaled coords (jac * D.T scales columns).
    grad, hess = norm.grad_hess(r, jac * D.T)

    # Get free (unclamped) gradient.
    if bounds is None:
      grad_free = grad
    else:
      clamped_lower = (x == bounds[0]) & (grad > 0)
      clamped_upper = (x == bounds[1]) & (grad < 0)
      clamped = clamped_lower | clamped_upper
      grad_free = grad[~clamped]

    # Check termination condition on gradient norm.
    g_norm = np.linalg.norm(grad_free)
    if g_norm <= gtol:
      status = Status.G_TOL
      if g_norm == 0 and verbose > Verbosity.SILENT.value:
        print('Zero gradient norm: exact minimum found?', file=output)
      break

    # Bounds relative to x, expressed in scaled coords (dz = dx / D).
    dlower = None if bounds is None else (bounds[0] - x) / D
    dupper = None if bounds is None else (bounds[1] - x) / D

    # Find reduction satisfying Armijo's rule.
    armijo = -1
    reduction = 0.0
    while armijo < 0:
      # Increase mu until factorizable.
      factorizable = False
      while not factorizable:
        n_free = mujoco.mju_boxQP(
            dx, scratch, None, hess + mu * np.eye(n), grad, dlower, dupper
        )
        if n_free >= 0:
          factorizable = True
        elif mu >= mu_max:
          status = Status.FACTORIZATION_FAILED
          break
        else:
          mu, n_reduc = increase_mu(mu)

      if status != Status.MAX_ITER:
        break

      # New candidate (D * dx is the x-space step).
      xnew = x + D * dx
      t_start = time.time()
      rnew = residual(xnew)
      t_res += time.time() - t_start
      n_res += 1

      # New objective, evaluate reduction.
      ynew = norm.value(rnew)
      reduction = y - ynew

      # Reject nonfinite trial objectives and retry with more regularization.
      armijo = (
          reduction + armijo_c1 * (grad.T @ dx).item()
          if np.isfinite(ynew)
          else -np.inf
      )

      if armijo < 0:
        if mu >= mu_max:
          status = Status.NO_IMPROVEMENT
          break
        mu, n_reduc = increase_mu(mu)

    if status != Status.MAX_ITER:
      break

    # Compute reduction ratio.
    expected_reduction = -(grad.T @ dx + 0.5 * dx.T @ hess @ dx).item()
    reduction_ratio = 0.0
    if expected_reduction <= 0:
      if verbose > Verbosity.SILENT.value:
        if expected_reduction == 0:
          print('Zero expected reduction: exact minimum found?', file=output)
        elif expected_reduction < 0:
          print('Negative expected reduction: should not occur.', file=output)
    else:
      reduction_ratio = reduction / expected_reduction

    # Iteration message. Step printed in x-space (D * dx).
    step = D * dx
    dx_norm = np.linalg.norm(step)
    if verbose >= Verbosity.ITER.value:
      logmu = np.log10(mu) if mu > 0 else -np.inf
      message = (
          f'iter: {i:<3d}  y: {y:<9.4g}  log10mu: {logmu:>4.1f}  '
          f'ratio: {reduction_ratio:<7.2g}  '
          f'dx: {dx_norm:<7.2g}  reduction: {reduction:<7.2g}'
      )
      print(message, file=output)

    # Append log to trace, call iter_callback.
    log = IterLog(candidate=x, objective=y, reduction=reduction, regularizer=mu)  # pyrefly: ignore[bad-argument-type]
    if verbose >= Verbosity.FULLITER.value:
      log = dataclasses.replace(
          log, residual=r, jacobian=jac, grad=grad / D, step=step
      )
    trace.append(log)
    if iter_callback is not None:
      iter_callback(trace)

    # Check termination condition on step norm.
    if dx_norm < xtol * (xtol + np.linalg.norm(x)):
      x = xnew
      r = rnew
      status = Status.DX_TOL
      break

    # Modify regularizer like in (Fletcher, 1971)
    if reduction_ratio > 0.75:
      mu, n_reduc = decrease_mu(mu, n_reduc)
    elif reduction_ratio < 0.25:
      mu, n_reduc = increase_mu(mu)

    # Accept proposal.
    x = xnew
    r = rnew

  # Append final log to trace, call iter_callback.
  # Note: unlike other iter logs, values are computed at the end point.
  yfinal = norm.value(r)
  red = np.float64(0.0)  # No reduction since we didn't take a step.
  log = IterLog(candidate=x, objective=yfinal, reduction=red, regularizer=mu)

  # If full verbosity requested, compute values at the final point.
  if verbose >= Verbosity.FULLITER.value:
    # Get Jacobian jac.
    t_start = time.time()
    if jacobian is None:
      jac, n_res = jacobian_fd(residual, x, r, eps, n_res, bounds)
      t_res += time.time() - t_start
    else:
      jac = jacobian(x, r)
      t_jac += time.time() - t_start
      n_jac += 1

    # Get gradient, add to log.
    grad, _ = norm.grad_hess(r, jac)
    log = dataclasses.replace(log, residual=r, jacobian=jac, grad=grad)

  trace.append(log)
  if iter_callback is not None:
    iter_callback(trace)

  # Print final diagnostics.
  if verbose > Verbosity.SILENT.value:
    message = f'Terminated after {i} iterations: '
    message += _STATUS_MESSAGE[status]
    message += f' y: {yfinal:<.4g}, Residual evals: {n_res:d}'
    if n_jac > 0:
      message += f', Jacobian evals: {n_jac:d}'
    print(message, file=output)

    time_total = time.time() - t_start_total
    if time_total > 0:
      r_percent = 100 * t_res / time_total
      time_scale = 1 if time_total > 1 else 1000
      time_units = 's' if time_total > 1 else 'ms'
      message = f'total time {time_scale * time_total:<.1f}{time_units}'
      message += f' of which residual {r_percent:<.1f}%'
      if t_jac > 0:
        jac_percent = 100 * t_jac / time_total
        message += f' Jacobian {jac_percent:<.1f}%'
      print(message, file=output)

  return x.reshape(x0.shape), trace


def jacobian_fd(
    residual: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    r: np.ndarray,
    eps: np.float64,
    n_res: int,
    bounds: Optional[List[np.ndarray]] = None,
) -> Tuple[np.ndarray, int]:
  """Finite-difference Jacobian of a residual function.

  Args:
    residual: vectorized function that returns the residual of a vector array.
    x: point at which to evaluate the Jacobian.
    r: residual at x.
    eps: finite-difference step size.
    n_res: number or residual evaluations so far.
    bounds: optional pair of lower and upper bounds.

  Returns:
    jac: Jacobian of the residual at x.
    n_res: updated number of residual evaluations (add x.size).
  """
  n = x.size
  if bounds is None:
    eps_vec = eps * np.ones((n, 1))
  else:
    mid = 0.5 * bounds[0] + 0.5 * bounds[1]
    # The midpoint can round to the upper bound for adjacent floats.
    eps_vec = np.where((x > mid) | (x == bounds[1]), -eps, eps)
  eps_vec *= np.maximum(1.0, np.abs(x))
  xh = x + np.diag(eps_vec.flatten())
  if bounds is not None:
    np.clip(xh, bounds[0], bounds[1], out=xh)
  # Use the actual step after rounding and clipping the evaluation points.
  eps_vec = xh.diagonal()[:, None] - x
  rh = residual(xh)
  jac = (rh - r) / eps_vec.T
  return jac, n_res + n


def check_jacobian(
    residual: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    r: np.ndarray,
    jac: np.ndarray,
    eps: np.float64,
    n_res: int,
    bounds: Optional[List[np.ndarray]] = None,
    output: Optional[TextIO] = None,
    name: Optional[str] = 'Jacobian',
) -> int:
  """Check user-provided Jacobian against internal finite-differencing.

  Args:
    residual: vectorized function that returns the residual of a vector array.
    x: point at which the r and jac were evaluated.
    r: residual at x.
    jac: Jacobian at x.
    eps: finite-difference step size.
    n_res: number or residual evaluations so far.
    bounds: optional pair of lower and upper bounds.
    output: Optional file or StringIO to which to print messages.
    name: Optional name of the function being tested.

  Returns:
    n_res: updated number of residual evaluations.

  Raises:
    ValueError: If either Jacobian contains nonfinite values or they disagree.
  """
  # A NaN relative difference would otherwise pass the comparison below.
  if not np.all(np.isfinite(jac)):
    raise ValueError(f'User-provided {name} must contain only finite values.')
  jac_fd, n_res = jacobian_fd(residual, x, r, eps, n_res, bounds)
  if not np.all(np.isfinite(jac_fd)):
    raise ValueError(
        f'Finite-difference {name} must contain only finite values.'
    )
  denom = np.abs(jac).sum() + np.abs(jac_fd).sum() + 1e-8
  rel_diff = np.abs(jac - jac_fd) / denom
  if np.any(rel_diff > 1e-5):
    raise ValueError(
        f'User-provided {name} does not match finite-differences '
        'to a relative tolerance of 1e-5.'
    )
  print(f'User-provided {name} matches finite-differences.', file=output)
  return n_res


def check_norm(
    r: np.ndarray,
    norm: Norm,
    eps: np.float64,
    output: Optional[TextIO] = None,
):
  """Check user-provided norm against internal finite-differencing.

  Args:
    r: residual vector.
    norm: Norm function returning either the norm scalar or its gradient and
      Gauss-Newton Hessian.
    eps: finite-difference step size.
    output: Optional file or StringIO to which to print messages.
  """
  # Get norm(r) value and 1st, 2nd derivatives.
  n = np.atleast_2d(norm.value(r))  # norm value as 1x1 array.
  eye = np.eye(r.size)  # Identity projection.
  n_g, n_h = norm.grad_hess(r, eye)  # Gradient and Hessian.

  # A nonfinite Hessian would fail the checks below with misleading errors.
  if not np.all(np.isfinite(n_h)):
    raise ValueError(
        'User-provided norm Hessian must contain only finite values.'
    )

  # Check that Hessian is symmetric.
  if not np.allclose(n_h, n_h.T):
    raise ValueError('User-provided norm Hessian is not symmetric.')

  # Check that Hessian is positive-definite.
  if np.any(np.linalg.eigvals(n_h) < 0):
    h_min = np.min(np.linalg.eigvals(n_h))
    raise ValueError(
        'User-provided norm Hessian is not positive definite. '
        f'Minimum eigenvalue is {h_min:<.4g}'
    )

  # Local function returning norm values (vectorized).
  def norm_vec(v):
    norms = [
        np.atleast_2d(norm.value(v[:, i : i + 1])) for i in range(v.shape[1])
    ]
    return np.hstack(norms)

  # Check the norm gradient.
  check_jacobian(norm_vec, r, n, n_g.T, eps, 0, None, output, 'norm gradient')

  # Local function returning norm gradients (vectorized).
  def grad_vec(v):
    gradients = [
        norm.grad_hess(v[:, i : i + 1], eye)[0] for i in range(v.shape[1])
    ]
    return np.hstack(gradients)

  # Check the norm Hessian.
  check_jacobian(grad_vec, r, n_g, n_h, eps, 0, None, output, 'norm Hessian')
