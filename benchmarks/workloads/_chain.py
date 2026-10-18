"""Scaly's chain formulation and matching JAX dynamics, with synthetic sweep inputs."""

from functools import cache

import jax
import jax.numpy as jnp
import numpy as np
import scaly as sc

jax.config.update("jax_enable_x64", True)


NU = 3


N_PARAMS = 5


HORIZON = 40  # laopt instance: N=40 shooting intervals, tf=8.0 -> dt=0.2


Q_END, Q_VEL, R_U, Q_END_TERMINAL = 2.5, 25.0, 0.1, 10.0


END_REF = (0.75, 0.0, 0.0)


PARAMS = np.array([0.033, 1.0, 0.033, -9.81, 0.2])


def n_state(n_masses: int) -> int:
  if n_masses < 3:
    raise ValueError(f"n_masses must be at least 3, got {n_masses}")
  return 3 * (2 * (n_masses - 2) + 1)


def n_dec(n_masses: int, horizon: int) -> int:
  if horizon < 1:
    raise ValueError(f"horizon must be positive, got {horizon}")
  return horizon * (n_state(n_masses) + NU) + n_state(n_masses)


def n_param(n_masses: int) -> int:
  return n_state(n_masses) + N_PARAMS


@sc.function(sc.arg("dist", 3), sc.arg("mass", 1), sc.arg("spring_d", 1), sc.arg("rest_len", 1), outputs=sc.arg("accel", 3), name="chain_link_accel")
def chain_link_accel_fn(dist: sc.Expr, mass: sc.Expr, spring_d: sc.Expr, rest_len: sc.Expr) -> sc.Expr:
  return (spring_d[0] / mass[0]) * (1.0 - rest_len[0] / sc.norm_2(dist)) * dist


@sc.function(
  sc.arg("left", 3),
  sc.arg("pos", 3),
  sc.arg("right", 3),
  sc.arg("mass", 1),
  sc.arg("spring_d", 1),
  sc.arg("rest_len", 1),
  sc.arg("gravity", 1),
  outputs=sc.arg("accel", 3),
  name="chain_mass_accel",
)
def chain_mass_accel_fn(
  left: sc.Expr, pos: sc.Expr, right: sc.Expr, mass: sc.Expr, spring_d: sc.Expr, rest_len: sc.Expr, gravity: sc.Expr
) -> sc.Expr:
  left_accel = chain_link_accel_fn(pos - left, mass, spring_d, rest_len)
  right_accel = chain_link_accel_fn(right - pos, mass, spring_d, rest_len)
  return right_accel - left_accel + sc.stack([0.0, 0.0, gravity[0]])


def chain_ode_fn(n_masses: int) -> sc.Function:
  """Build the continuous-time chain dynamics for the free masses and actuated end mass."""
  nx = n_state(n_masses)
  constant = sc.TensorType((1,), diff=False)

  @sc.function(
    sc.arg("x", nx),
    sc.arg("u", NU),
    *(sc.arg(name, constant) for name in ("mass", "spring_d", "rest_len", "gravity")),
    outputs=sc.arg("xdot", nx),
    name=f"chain_ode_M{n_masses}",
  )
  def ode(x: sc.Expr, u: sc.Expr, mass: sc.Expr, spring_d: sc.Expr, rest_len: sc.Expr, gravity: sc.Expr) -> sc.Expr:
    positions = sc.concat([sc.const(np.zeros(3)), x[: 3 * (n_masses - 1)]])
    velocities = x[3 * (n_masses - 1) :]
    accel = sc.vmap(chain_mass_accel_fn, n_masses - 2)(
      sc.window(positions, 0, 3),
      sc.window(positions, 3, 3),
      sc.window(positions, 6, 3),
      sc.broadcast(mass),
      sc.broadcast(spring_d),
      sc.broadcast(rest_len),
      sc.broadcast(gravity),
    ).vec()
    return sc.concat([velocities, u, accel])

  return ode


def chain_step_fn(n_masses: int) -> sc.Function:
  """Build one fourth-order Runge-Kutta step with a runtime time step."""
  nx = n_state(n_masses)
  ode = chain_ode_fn(n_masses)
  constant = sc.TensorType((1,), diff=False)

  @sc.function(
    sc.arg("x", nx),
    sc.arg("u", NU),
    *(sc.arg(name, constant) for name in ("mass", "spring_d", "rest_len", "gravity", "dt")),
    outputs=sc.arg("next", nx),
    name=f"chain_step_M{n_masses}",
  )
  def step(x: sc.Expr, u: sc.Expr, mass: sc.Expr, spring_d: sc.Expr, rest_len: sc.Expr, gravity: sc.Expr, dt: sc.Expr) -> sc.Expr:

    def rhs(state: sc.Expr) -> sc.Expr:
      return ode(state, u, mass, spring_d, rest_len, gravity)

    h = dt[0]
    k1 = rhs(x)
    k2 = rhs(x + 0.5 * h * k1)
    k3 = rhs(x + 0.5 * h * k2)
    k4 = rhs(x + h * k3)
    return x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

  return step


def _eq_stage_fn(n_masses: int) -> sc.Function:
  nx, nz = n_state(n_masses), n_state(n_masses) + NU

  @sc.function(
    sc.arg("z", nz),
    sc.arg("xnext", nx),
    sc.arg("params", sc.TensorType((N_PARAMS,), diff=False)),
    outputs=sc.arg("eq", nx),
    name=f"chain_eq_stage_M{n_masses}",
  )
  def equality(z: sc.Expr, xnext: sc.Expr, params: sc.Expr) -> sc.Expr:
    predicted = chain_step_fn(n_masses)(z[:nx], z[nx:], *[params[i : i + 1] for i in range(N_PARAMS)])
    return (predicted - xnext).scalar()

  return equality


def _chain_eq_expr(z: sc.Expr, p: sc.Expr, n_masses: int, horizon: int) -> sc.Expr:
  nx, nz = n_state(n_masses), n_state(n_masses) + NU
  mapped = sc.vmap(_eq_stage_fn(n_masses), horizon)(sc.window(z, 0, nz), sc.window(z, nz, nz), sc.window(p, nx, 0)).vec()
  return sc.concat([z[:nx] - p[:nx], mapped])


def _objective(z: sc.Expr, n_masses: int, horizon: int) -> sc.Expr:
  nx, nz = n_state(n_masses), n_state(n_masses) + NU
  end = 3 * (n_masses - 2)
  vel = 3 * (n_masses - 1)
  h = 1.0 / horizon
  ref = sc.const(np.array(END_REF))
  cost = sc.const(0.0)
  for i in range(horizon):
    zi = z[i * nz : (i + 1) * nz]
    cost = cost + h * 0.5 * (Q_END * sc.sumsqr(zi[end : end + 3] - ref) + Q_VEL * sc.sumsqr(zi[vel:nx]) + R_U * sc.sumsqr(zi[nx:]))
  terminal = z[horizon * nz : horizon * nz + nx]
  return cost + 0.5 * Q_END_TERMINAL * sc.sumsqr(terminal[end : end + 3] - ref)


def chain_problem(n_masses: int, horizon: int):
  """Build the chain problem with bounded controls and runtime physics."""
  nx, nz = n_state(n_masses), n_state(n_masses) + NU
  n_variables = n_dec(n_masses, horizon)
  lb = np.full(n_variables, -np.inf)
  ub = np.full(n_variables, np.inf)
  for i in range(horizon):
    lb[i * nz + nx : (i + 1) * nz] = -1.0
    ub[i * nz + nx : (i + 1) * nz] = 1.0
  problem_name = f"chain_M{n_masses}_N{horizon}"

  @sc.problem(vars=sc.arg("z", n_variables), params=sc.arg("p", n_param(n_masses)), name=problem_name)
  def problem(z: sc.Expr, p: sc.Expr) -> sc.ProblemSpec[sc.Expr]:
    return sc.ProblemSpec(
      minimize=_objective(z, n_masses, horizon),
      eq=(_chain_eq_expr(z, p, n_masses, horizon),),
      lb=sc.const(lb),
      ub=sc.const(ub),
    )

  return problem


def initial_state(n_masses: int) -> np.ndarray:
  x = np.zeros(n_state(n_masses), dtype=np.float64)
  x[: 3 * (n_masses - 1) : 3] = 7.0 * np.arange(1, n_masses) / (n_masses - 1)
  return x


def sample_inputs(n_masses: int, horizon: int, *, seed: int = 7, params: np.ndarray = PARAMS) -> tuple[np.ndarray, np.ndarray]:
  rng = np.random.default_rng(seed)
  nx, nz = n_state(n_masses), n_state(n_masses) + NU
  z = np.empty(n_dec(n_masses, horizon), dtype=np.float64)
  base = initial_state(n_masses)
  for i in range(horizon):
    z[i * nz : i * nz + nx] = base + rng.normal(scale=0.03, size=nx)
    z[i * nz + nx : (i + 1) * nz] = rng.normal(scale=0.1, size=NU)
  z[horizon * nz :] = base + rng.normal(scale=0.03, size=nx)
  return z, np.concatenate([base, params])


def step_jax(zi, params, M: int):
  nx = n_state(M)
  x, u = zi[:nx], zi[nx:]
  mass, spring_d, rest_len, gravity, dt = params

  def rhs(state):
    positions = jnp.concatenate([jnp.zeros(3), state[: 3 * (M - 1)]]).reshape(M, 3)
    dist = positions[1:] - positions[:-1]
    forces = (spring_d / mass) * (1.0 - rest_len / jnp.linalg.norm(dist, axis=1, keepdims=True)) * dist
    accel = forces[1:] - forces[:-1] + jnp.array([0.0, 0.0, gravity])
    return jnp.concatenate([state[3 * (M - 1) :], u, accel.reshape(-1)])

  k1 = rhs(x)
  k2 = rhs(x + 0.5 * dt * k1)
  k3 = rhs(x + 0.5 * dt * k2)
  k4 = rhs(x + dt * k3)
  return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def stage_cost_jax(zi, M: int):
  nx, end, vel = n_state(M), 3 * (M - 2), 3 * (M - 1)
  return (
    0.5
    / HORIZON
    * (Q_END * jnp.sum((zi[end : end + 3] - jnp.asarray(END_REF)) ** 2) + Q_VEL * jnp.sum(zi[vel:nx] ** 2) + R_U * jnp.sum(zi[nx:] ** 2))
  )


def terminal_cost_jax(x, M: int):
  end = 3 * (M - 2)
  return 0.5 * Q_END_TERMINAL * jnp.sum((x[end : end + 3] - jnp.asarray(END_REF)) ** 2)


def constraints_jax(z, p, M: int):
  nx, nz = n_state(M), n_state(M) + NU
  stages = z[: HORIZON * nz].reshape(HORIZON, nz)
  following = jnp.concatenate([stages[1:, :nx], z[None, HORIZON * nz :]], axis=0)
  predicted = jax.vmap(lambda zi: step_jax(zi, p[nx:], M))(stages)
  return jnp.concatenate([z[:nx] - p[:nx], (predicted - following).reshape(-1)])


def lagrangian_jax(z, p, lam_f, lam_g, M: int):
  nz = n_state(M) + NU
  stages = z[: HORIZON * nz].reshape(HORIZON, nz)
  cost = jax.vmap(lambda zi: stage_cost_jax(zi, M))(stages).sum() + terminal_cost_jax(z[HORIZON * nz :], M)
  return lam_f * cost + lam_g @ constraints_jax(z, p, M)


def dense_hessian(M: int, z, lam_f, lam_g, p):
  """Scatter plain JAX stage Hessians, independent of sparse differentiation."""
  nx, nz = n_state(M), n_state(M) + NU

  def stage(zi, lam):
    return lam_f * stage_cost_jax(zi, M) + lam @ step_jax(zi, p[nx:], M)

  blocks = jax.jit(jax.vmap(jax.hessian(stage)))(z[: HORIZON * nz].reshape(HORIZON, nz), lam_g[nx:].reshape(HORIZON, nx))
  dense = np.zeros((z.size, z.size))
  for i, block in enumerate(np.asarray(blocks)):
    dense[i * nz : (i + 1) * nz, i * nz : (i + 1) * nz] = block
  dense[HORIZON * nz :, HORIZON * nz :] = lam_f * np.asarray(jax.hessian(lambda x: terminal_cost_jax(x, M))(z[HORIZON * nz :]))
  return dense


def dense_jacobian(M: int, z, p):
  """Scatter plain JAX step Jacobians and the linear equality blocks."""
  nx, nz = n_state(M), n_state(M) + NU
  blocks = jax.jit(jax.vmap(jax.jacfwd(lambda zi: step_jax(zi, p[nx:], M))))(z[: HORIZON * nz].reshape(HORIZON, nz))
  dense = np.zeros((nx * (HORIZON + 1), z.size))
  dense[:nx, :nx] = np.eye(nx)
  for i, block in enumerate(np.asarray(blocks)):
    row, col = nx * (i + 1), nz * i
    dense[row : row + nx, col : col + nz] = block
    dense[row : row + nx, col + nz : col + nz + nx] = -np.eye(nx)
  return dense


@cache
def nlp(M: int):
  """The NLP form used by Scaly's solver descriptor, without requiring an IPOPT plugin."""
  return chain_problem(M, HORIZON)._nlp
