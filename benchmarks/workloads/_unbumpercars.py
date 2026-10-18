"""Discrete hyperbolic control barrier filter, adapted from Scaly's unbumpercars benchmark.

Scaly copyright (c) 2026, EPFL. See _unbumpercars_LICENSE for attribution and license.
Synthetic samples match Scaly's sweep; optional harvested closed-loop inputs are not used.
"""

from functools import cache
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import scaly as sc
from scaly.utils import load_torch_state_dict

jax.config.update("jax_enable_x64", True)

NSTATE, NCTRL, N_PHYSICS = 7, 2, 8
DT_W0_SHAPE, DT_W1_SHAPE, DT_W2_SHAPE = (256, 6), (128, 256), (3, 128)
DT_SHAPES = ((4,), DT_W0_SHAPE, (256,), DT_W1_SHAPE, (128,), DT_W2_SHAPE, (3,))
DT_OFFSETS = np.cumsum([0, *(int(np.prod(shape)) for shape in DT_SHAPES)])
N_PW_DT = int(DT_OFFSETS[-1])
DT_RELU_EPS = 0.01


def n_g(C: int) -> int:
  return C * (C - 1) // 2 + 4 * C


def config(ncars: int):
  return SimpleNamespace(
    ncars=ncars,
    n_pairs=ncars * (ncars - 1) // 2,
    n_slack=n_g(ncars),
    arena_avoidance=True,
    safety_radius=2.28,
    wall_margin=1.0,
    wall_eps=1e-4,
    pair_gamma=0.35,
    wall_gamma=0.35,
    hcbf=SimpleNamespace(envelope_c=1.00994, envelope_q=0.8355, eps=0.05, single_envelope_c=1.00994 / 2.0 ** (1.0 - 0.8355)),
  )


def _world_vel_expr(state: sc.Expr, physics: sc.Expr) -> tuple[sc.Expr, sc.Expr, sc.Expr]:
  lf, lr = physics[0], physics[1]
  theta, vf, beta_f, beta_r = state[2], state[3], state[4], state[5]
  omega = vf * (beta_f - beta_r).sin() / ((lf + lr) * beta_r.cos())
  vx_b = vf * beta_f.cos()
  vy_b = vf * beta_f.sin() - lf * omega
  return vx_b * theta.cos() - vy_b * theta.sin(), vx_b * theta.sin() + vy_b * theta.cos(), omega


def _smooth_relu_expr(x: sc.Expr) -> sc.Expr:
  return 0.5 * (x + (x * x + DT_RELU_EPS**2).sqrt())


def _unpack_pw_dt_expr(pw: sc.Expr) -> tuple[sc.Expr, ...]:
  return (
    pw[DT_OFFSETS[0] : DT_OFFSETS[1]],
    pw[DT_OFFSETS[1] : DT_OFFSETS[2]].reshape(DT_W0_SHAPE),
    pw[DT_OFFSETS[2] : DT_OFFSETS[3]],
    pw[DT_OFFSETS[3] : DT_OFFSETS[4]].reshape(DT_W1_SHAPE),
    pw[DT_OFFSETS[4] : DT_OFFSETS[5]],
    pw[DT_OFFSETS[5] : DT_OFFSETS[6]].reshape(DT_W2_SHAPE),
    pw[DT_OFFSETS[6] : DT_OFFSETS[7]],
  )


@sc.function(sc.arg("state", NSTATE), sc.arg("physics", N_PHYSICS), outputs=sc.arg("posedot", NSTATE), name="ctdt_pose_dot")
def scaly_pose_dot_fn(state: sc.Expr, physics: sc.Expr) -> sc.Expr:
  x_dot, y_dot, omega = _world_vel_expr(state, physics)
  zero = 0.0 * state[3]
  return sc.stack([x_dot, y_dot, omega, zero, zero, zero, zero])


@sc.function(
  sc.arg("state", NSTATE),
  sc.arg("u", NCTRL),
  sc.arg("pw", N_PW_DT),
  sc.arg("physics", N_PHYSICS),
  sc.arg("dt", 1),
  outputs=sc.arg("next", NSTATE),
  name="ctdt_dt_mlp_step",
)
def scaly_dt_mlp_step_fn(state: sc.Expr, u: sc.Expr, pw: sc.Expr, physics: sc.Expr, dt: sc.Expr) -> sc.Expr:
  """The smoothed discrete network's one-step map."""
  h_dt = dt[0]
  max_delta, steering_time_constant = physics[2], physics[3]
  delta = state[6]
  k1 = scaly_pose_dot_fn(state, physics)
  k2 = scaly_pose_dot_fn(state + 0.5 * h_dt * k1, physics)
  k3 = scaly_pose_dot_fn(state + 0.5 * h_dt * k2, physics)
  k4 = scaly_pose_dot_fn(state + h_dt * k3, physics)
  pose = state + (h_dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
  x_scale, w0, b0, w1, b1, w2, b2 = _unpack_pw_dt_expr(pw)
  phi = sc.concat([sc.stack([state[3], state[4] - delta, state[5], delta]) / x_scale, sc.stack([u[1], u[0]])])
  h = _smooth_relu_expr(w0 @ phi + b0)
  h = _smooth_relu_expr(w1 @ h + b1)
  learned = w2 @ h + b2
  delta_next = delta + h_dt * (u[1] * max_delta - delta) / steering_time_constant
  return sc.stack([pose[0], pose[1], pose[2], learned[0], learned[1] + delta_next, learned[2], delta_next])


def _scaly_oracle_outputs(inputs: tuple[sc.Expr, sc.Expr, sc.Expr, sc.Expr, sc.Expr, sc.Expr], loop_cfg, filt_cfg) -> tuple[sc.Expr, sc.Expr]:
  """Build the tracking cost and barrier constraints."""
  z, bar_x, u_des, pw, physics, dt = inputs
  ncars = loop_cfg.ncars
  n_u = NCTRL * ncars
  n_s = loop_cfg.n_slack
  u = z[:n_u]
  slack = z[n_u:]
  step_fn = scaly_dt_mlp_step_fn
  states_next = sc.vmap(step_fn, ncars)(bar_x, u, sc.broadcast(pw), sc.broadcast(physics), sc.broadcast(dt)).vec()
  hcbf, R = loop_cfg.hcbf, loop_cfg.safety_radius

  def pair_b(xi: sc.Expr, xj: sc.Expr, physics: sc.Expr) -> sc.Expr:
    """Order-1 hyperbolic pair barrier."""
    px, py = xj[0] - xi[0], xj[1] - xi[1]
    vxi, vyi, _ = _world_vel_expr(xi, physics)
    vxj, vyj, _ = _world_vel_expr(xj, physics)
    vx, vy = vxj - vxi, vyj - vyi
    r = (px * px + py * py).sqrt()
    v_x = (px * vx + py * vy) / r
    v_y = (px * vy - py * vx) / r
    d_eps = ((r - R) * (r - R) + hcbf.eps**2).sqrt()
    s = (r - R) / d_eps
    a_env = hcbf.envelope_c * d_eps**hcbf.envelope_q
    q = (d_eps * (r + R)).sqrt() / R * v_y
    return v_x + s * (a_env**4 + q**4).sqrt().sqrt()

  m = loop_cfg.wall_margin

  def wall_b(xi: sc.Expr, physics: sc.Expr) -> sc.Expr:
    x_min, x_max, y_min, y_max = [physics[i] for i in range(4, 8)]
    vx, vy, _ = _world_vel_expr(xi, physics)
    clearances = [xi[0] - (x_min + m), (x_max - m) - xi[0], xi[1] - (y_min + m), (y_max - m) - xi[1]]
    out = []
    for clearance, v_closing in zip(clearances, (-vx, vx, -vy, vy), strict=True):
      d_eps = (clearance * clearance + loop_cfg.wall_eps**2).sqrt()
      s = clearance / d_eps
      v_max = hcbf.single_envelope_c * d_eps**hcbf.envelope_q
      out.append(s * v_max - v_closing)
    return sc.stack(out)

  @sc.function(
    sc.arg("xi", NSTATE),
    sc.arg("xj", NSTATE),
    sc.arg("xi_next", NSTATE),
    sc.arg("xj_next", NSTATE),
    sc.arg("physics", N_PHYSICS),
    outputs=sc.arg("g", 1),
    name="pair_hcbf",
  )
  def pair_hcbf(xi: sc.Expr, xj: sc.Expr, xi_next: sc.Expr, xj_next: sc.Expr, physics: sc.Expr) -> sc.Expr:
    return sc.stack([pair_b(xi_next, xj_next, physics) - (1.0 - loop_cfg.pair_gamma) * pair_b(xi, xj, physics)])

  @sc.function(
    sc.arg("state", NSTATE),
    sc.arg("state_next", NSTATE),
    sc.arg("physics", N_PHYSICS),
    outputs=sc.arg("g", 4),
    name="wall_hcbf",
  )
  def wall_hcbf(state: sc.Expr, state_next: sc.Expr, physics: sc.Expr) -> sc.Expr:
    return wall_b(state_next, physics) - (1.0 - loop_cfg.wall_gamma) * wall_b(state, physics)

  rows: list[sc.Expr] = []
  if loop_cfg.n_pairs:
    pairs = np.triu_indices(ncars, k=1)
    idx_i, idx_j = [np.concatenate([np.arange(NSTATE) + k * NSTATE for k in bodies]) for bodies in pairs]
    rows.append(
      sc.vmap(pair_hcbf, loop_cfg.n_pairs)(
        sc.gather(bar_x, idx_i), sc.gather(bar_x, idx_j), sc.gather(states_next, idx_i), sc.gather(states_next, idx_j), sc.broadcast(physics)
      ).vec()
    )
  if loop_cfg.arena_avoidance:
    rows.append(sc.vmap(wall_hcbf, ncars)(bar_x, states_next, sc.broadcast(physics)).vec())
  g = (sc.concat(rows) + slack) if rows else sc.const(np.zeros((0,)))
  assert g.shape == (n_s,)
  diff = u - u_des
  weights = sc.const(np.tile(np.asarray(filt_cfg.R, dtype=np.float64), ncars))
  cost = sc.dot(diff, weights * diff) + filt_cfg.slack_weight * slack.sum()
  return cost, g


@cache
def scaly_problem(C: int):
  """Build the safety-filter nonlinear program over controls and nonnegative slacks."""
  loop_cfg = config(C)
  filt_cfg = SimpleNamespace(R=(10.0, 1.0), slack_weight=1000.0)
  n_u, n_s = NCTRL * C, n_g(C)
  vars_tree = sc.group(sc.arg("u", (n_u,)), sc.arg("s", (n_s,)))
  params_tree = sc.group(
    sc.arg("bar_x", NSTATE * loop_cfg.ncars),
    sc.arg("u_des", n_u),
    sc.arg("pw", N_PW_DT),
    sc.arg("physics", N_PHYSICS),
    sc.arg("dt", 1),
  )
  problem_name = f"ctdt_scaly_problem_N{loop_cfg.ncars}_walls"

  @sc.problem(vars=vars_tree, params=params_tree, name=problem_name)
  def problem(
    variables: tuple[sc.Expr, sc.Expr], params: tuple[sc.Expr, sc.Expr, sc.Expr, sc.Expr, sc.Expr]
  ) -> sc.ProblemSpec[tuple[sc.Expr, sc.Expr]]:
    u, s = variables
    cost, constraints = _scaly_oracle_outputs((sc.concat([u, s]), *params), loop_cfg, filt_cfg)
    inequalities = (sc.bounded(constraints, lo=sc.const(np.zeros(n_s)), name="barrier"),) if n_s else ()
    return sc.ProblemSpec(
      minimize=cost,
      ineq=inequalities,
      lb=(sc.const(-np.ones(n_u)), sc.const(np.zeros(n_s))),
      ub=(sc.const(np.ones(n_u)), sc.const(np.full(n_s, np.inf))),
    )

  return problem


def sample_inputs(C: int) -> dict[str, np.ndarray]:
  rng = np.random.default_rng(42)
  states = np.zeros((C, NSTATE))
  for i in range(C):
    states[i, :2] = rng.uniform(2.28 + 0.4, 15.0 - (2.28 + 0.4), 2)
    states[i, 2] = rng.uniform(-np.pi, np.pi)
    states[i, 3] = 0.05
  state = load_torch_state_dict(Path(__file__).with_name("_unbumpercars_dt_kinematic_mlp.pt"))
  pw = np.concatenate(
    [
      np.array([1.9, 0.6, 0.1, 2.012]),
      *(np.asarray(state[f"model.{layer}.{field}"], dtype=np.float64).reshape(-1) for layer in (0, 2, 4) for field in ("weight", "bias")),
    ]
  )
  desired = np.tile([0.55, 0.0], C)
  return {
    "z": np.concatenate([desired, np.zeros(n_g(C))]),
    "bar_x": states.reshape(-1),
    "u_des": desired,
    "pw": pw,
    "physics": np.array([0.54, 0.33, 2.012, 0.155, 0.0, 15.0, 0.0, 15.0]),
    "dt": np.array([0.1]),
    "lam_f": np.array(1.0),
    "lam_g": np.linspace(0.1, 1.0, n_g(C)),
  }


def world_velocity_jax(x, physics):
  theta, vf, beta_f, beta_r = x[2], x[3], x[4], x[5]
  omega = vf * jnp.sin(beta_f - beta_r) / ((physics[0] + physics[1]) * jnp.cos(beta_r))
  vx, vy = vf * jnp.cos(beta_f), vf * jnp.sin(beta_f) - physics[0] * omega
  return jnp.stack([vx * jnp.cos(theta) - vy * jnp.sin(theta), vx * jnp.sin(theta) + vy * jnp.cos(theta), omega])


def step_jax(x, u, pw, physics, dt):
  def pose_dot(state):
    return jnp.concatenate([world_velocity_jax(state, physics), jnp.zeros(4)])

  h_dt, delta = dt[0], x[6]
  k1 = pose_dot(x)
  k2 = pose_dot(x + 0.5 * h_dt * k1)
  k3 = pose_dot(x + 0.5 * h_dt * k2)
  k4 = pose_dot(x + h_dt * k3)
  pose = x + h_dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
  scale, w0, b0, w1, b1, w2, b2 = [pw[a:b].reshape(shape) for a, b, shape in zip(DT_OFFSETS[:-1], DT_OFFSETS[1:], DT_SHAPES, strict=True)]
  h = jnp.concatenate([jnp.stack([x[3], x[4] - delta, x[5], delta]) / scale, u[::-1]])
  for w, b in ((w0, b0), (w1, b1)):
    t = w @ h + b
    h = 0.5 * (t + jnp.sqrt(t * t + DT_RELU_EPS**2))
  learned = w2 @ h + b2
  delta_next = delta + h_dt * (u[1] * physics[2] - delta) / physics[3]
  return jnp.concatenate([pose[:3], jnp.stack([learned[0], learned[1] + delta_next, learned[2], delta_next])])


def pair_jax(xi, xj, physics):
  p = xj[:2] - xi[:2]
  v = world_velocity_jax(xj, physics)[:2] - world_velocity_jax(xi, physics)[:2]
  r = jnp.sqrt(p @ p)
  vx, vy = (p @ v) / r, (p[0] * v[1] - p[1] * v[0]) / r
  d = jnp.sqrt((r - 2.28) ** 2 + 0.05**2)
  a = 1.00994 * d**0.8355
  q = jnp.sqrt(d * (r + 2.28)) / 2.28 * vy
  return vx + (r - 2.28) / d * jnp.sqrt(jnp.sqrt(a**4 + q**4))


def wall_jax(x, physics):
  vx, vy, _ = world_velocity_jax(x, physics)
  clearance = jnp.stack([x[0] - physics[4] - 1.0, physics[5] - 1.0 - x[0], x[1] - physics[6] - 1.0, physics[7] - 1.0 - x[1]])
  d = jnp.sqrt(clearance * clearance + 1e-8)
  return clearance / d * (1.00994 / 2.0 ** (1.0 - 0.8355)) * d**0.8355 - jnp.stack([-vx, vx, -vy, vy])


def constraints_jax(z, bar_x, u_des, pw, physics, dt, C: int):
  x = bar_x.reshape(C, NSTATE)
  nxt = jax.vmap(step_jax, in_axes=(0, 0, None, None, None))(x, z[: 2 * C].reshape(C, 2), pw, physics, dt)
  i, j = np.triu_indices(C, k=1)
  pairs = jax.vmap(pair_jax, in_axes=(0, 0, None))
  walls = jax.vmap(wall_jax, in_axes=(0, None))
  return (
    jnp.concatenate(
      [pairs(nxt[i], nxt[j], physics) - 0.65 * pairs(x[i], x[j], physics), (walls(nxt, physics) - 0.65 * walls(x, physics)).reshape(-1)]
    )
    + z[2 * C :]
  )


def lagrangian_jax(z, bar_x, u_des, pw, physics, dt, lam_f, lam_g, C: int):
  diff = z[: 2 * C] - u_des
  cost = jnp.dot(diff, jnp.tile(jnp.array([10.0, 1.0]), C) * diff) + 1000.0 * z[2 * C :].sum()
  return lam_f * cost + lam_g @ constraints_jax(z, bar_x, u_des, pw, physics, dt, C)
