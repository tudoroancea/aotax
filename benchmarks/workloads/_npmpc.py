"""The npmpc problem: its layout, trained decoder and sample inputs, and its Lagrangian written once in JAX and once in Scaly."""

from functools import cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import scaly as sc
from scaly.utils import load_torch_state_dict
from scipy.linalg import solve_discrete_are

jax.config.update("jax_enable_x64", True)

# State (theta, phi, theta_dot, phi_dot) with theta = 0 upright, input (torque,). The decision vector
# blocks the states first, then the controls, then one slack that softens the arm-angle limits:
#   z = [x_0 ... x_N | u_0 ... u_{N-1} | s]
NX, NU = 4, 1
DT = 0.02

# The conditional neural process's mean decoder, 9 -> 32 -> 32 -> 2 with sigmoids on the hidden layers.
# Its features are (sin theta, cos theta, theta_dot, phi_dot, torque), scaled, followed by the latent
# code. The parameter tail carries every number of it, in this order.
W0, W1, W2 = (32, 9), (32, 32), (2, 32)
PW_BLOCKS = {"x_scale_w": 5, "x_scale_b": 5, "y_inv_w": 2, "y_inv_b": 2, "w0": 288, "w1": 1024, "w2": 64, "bias": 2, "latent": 4}
_ends = np.cumsum(list(PW_BLOCKS.values()))
PW = {name: slice(int(end - size), int(end)) for (name, size), end in zip(PW_BLOCKS.items(), _ends, strict=True)}
N_PW = int(_ends[-1])

# The latent code of the paper's system 3, the pendulum whose closed loop the reference episode records.
LATENT = np.array([-2.34141442, 0.16122490, -10.00758909, -1.65464341])

# Stage weights on the lifted state (2 sin(theta/2), phi, theta_dot, phi_dot), inter-stage weights on
# state differences, the torque weight and the slack penalty; then the weights of the terminal
# Riccati solve.
COST = np.array([2.5, 1.5, 0.005, 0.01, 0.0, 0.0, 0.05, 0.125, 1.0, 1000.0])
Q_END, R_END = np.diag([1.0, 10.0, 0.1, 0.1]), np.diag([1.0])

# p = [xstart (4) | decoder tail (1396) | dt | cost weights (10) | terminal weight P (16, row-major)]
P_PW = slice(NX, NX + N_PW)
P_DT = NX + N_PW
P_COST = slice(P_DT + 1, P_DT + 1 + COST.size)
P_TERMINAL = slice(P_COST.stop, P_COST.stop + NX * NX)
N_P = P_TERMINAL.stop

PHI_LIMIT, TORQUE_LIMIT, SLACK_LIMIT, X0_BAND = 2.0, 0.05, 5.0, 1e-3
CHECKPOINT = Path(__file__).parents[1] / "data" / "cnp_model.pth"


def n_z(N: int) -> int:
  return NX * (N + 1) + NU * N + 1


def n_g(N: int) -> int:
  """Constraint rows: the 4N dynamics equalities, the initial-state band, then two arm-angle rows per node."""
  return NX * N + NX + 2 * (N + 1)


def decoder_tail() -> np.ndarray:
  """The trained decoder and its scalers from the checkpoint, plus the latent code, in `PW_BLOCKS` order."""
  state = load_torch_state_dict(CHECKPOINT)
  mu, x_scale, y_inv = state["decoder"], state["x_direct_scaler"], state["y_inverse_scaler"]
  blocks = [
    x_scale["weight"],
    x_scale["bias"],
    y_inv["weight"],
    y_inv["bias"],
    *(mu[f"mu_decoder.{i}.weight"] for i in (0, 2, 4)),
    mu["mu_decoder.4.bias"],
  ]
  return np.concatenate([*(np.asarray(b, dtype=np.float64).reshape(-1) for b in blocks), LATENT])


# The model in JAX.


def decoder_jax(pw, x, u):
  feat = jnp.stack([jnp.sin(x[0]), jnp.cos(x[0]), x[2], x[3], u[0]]) * pw[PW["x_scale_w"]] + pw[PW["x_scale_b"]]
  h = jnp.concatenate([feat, pw[PW["latent"]]])
  h = 1.0 / (1.0 + jnp.exp(-(pw[PW["w0"]].reshape(W0) @ h)))
  h = 1.0 / (1.0 + jnp.exp(-(pw[PW["w1"]].reshape(W1) @ h)))
  return (pw[PW["w2"]].reshape(W2) @ h + pw[PW["bias"]]) * pw[PW["y_inv_w"]] + pw[PW["y_inv_b"]]


def stage_jax(x, xnext, u, pw, dt):
  """Dynamics residual of one stage: the decoder gives the velocity change, trapezoidal integration the positions."""
  y = decoder_jax(pw, x, u)
  return x + jnp.concatenate([dt * (x[2:4] + y / 2.0), y]) - xnext


def stage_cost_jax(x, xnext, u, w):
  dx = xnext - x
  return w[0] * 2.0 * (1.0 - jnp.cos(x[0])) + w[1:4] @ (x[1:] * x[1:]) + w[4:8] @ (dx * dx) + w[8] * u[0] * u[0]


def lagrangian_jax(z, p, lam_g, N: int):
  """cost(z) + lam_g . [dynamics; initial-state band; arm-angle rows], with both stage terms as a vmap over the horizon."""
  X = z[: NX * (N + 1)].reshape(N + 1, NX)
  U = z[NX * (N + 1) : NX * (N + 1) + NU * N].reshape(N, NU)
  slack = z[-1]
  pw, dt, w, P = p[P_PW], p[P_DT], p[P_COST], p[P_TERMINAL].reshape(NX, NX)
  eq = jax.vmap(stage_jax, in_axes=(0, 0, 0, None, None))(X[:-1], X[1:], U, pw, dt).reshape(-1)
  stages = jax.vmap(stage_cost_jax, in_axes=(0, 0, 0, None))(X[:-1], X[1:], U, w).sum()
  e = jnp.stack([2.0 * jnp.sin(X[-1, 0] / 2.0), X[-1, 1], X[-1, 2], X[-1, 3]])
  cost = stages + e @ P @ e + w[9] * 0.5 * (slack * slack + slack)
  phi = X[:, 1]
  ineq = jnp.concatenate([X[0] - p[:NX], phi + slack, phi - slack])
  return cost + lam_g @ jnp.concatenate([eq, ineq])


# The same model in Scaly.


@cache
def _stage_scaly() -> sc.Function:
  @sc.function(
    sc.arg("x", NX), sc.arg("xnext", NX), sc.arg("u", NU), sc.arg("pw", N_PW), sc.arg("dt", ()), outputs=sc.arg("eq", NX), name="npmpc_stage"
  )
  def stage(x: sc.Expr, xnext: sc.Expr, u: sc.Expr, pw: sc.Expr, dt: sc.Expr) -> sc.Expr:
    feat = sc.stack([x[0].sin(), x[0].cos(), x[2], x[3], u[0]]) * pw[PW["x_scale_w"]] + pw[PW["x_scale_b"]]
    h = sc.concat([feat, pw[PW["latent"]]])
    h = 1.0 / (1.0 + (-(pw[PW["w0"]].reshape(W0) @ h)).exp())
    h = 1.0 / (1.0 + (-(pw[PW["w1"]].reshape(W1) @ h)).exp())
    y = (pw[PW["w2"]].reshape(W2) @ h + pw[PW["bias"]]) * pw[PW["y_inv_w"]] + pw[PW["y_inv_b"]]
    return x + sc.concat([dt * (x[2:4] + y / 2.0), y]) - xnext

  return stage


@cache
def _stage_cost_scaly() -> sc.Function:
  @sc.function(sc.arg("x", NX), sc.arg("xnext", NX), sc.arg("u", NU), sc.arg("w", COST.size), outputs=sc.arg("cost", ()), name="npmpc_stage_cost")
  def stage_cost(x: sc.Expr, xnext: sc.Expr, u: sc.Expr, w: sc.Expr) -> sc.Expr:
    dx = xnext - x
    total = w[0] * 2.0 * (1.0 - x[0].cos()) + w[8] * u[0] * u[0]
    for k in range(1, NX):
      total = total + w[k] * x[k] * x[k]
    for k in range(NX):
      total = total + w[NX + k] * dx[k] * dx[k]
    return total

  return stage_cost


def lagrangian_scaly(N: int) -> sc.Function:
  """`lagrangian_jax` as a Scaly Function of (z, p, lam_g), with both stage terms as an `sc.vmap` over the horizon."""
  offset = NX * (N + 1)

  @sc.function(sc.arg("z", n_z(N)), sc.arg("p", N_P), sc.arg("lam_g", n_g(N)), outputs=sc.arg("lag", ()), name=f"npmpc_lagrangian_N{N}")
  def lagrangian(z: sc.Expr, p: sc.Expr, lam_g: sc.Expr) -> sc.Expr:
    x, xnext, u = sc.window(z, 0, NX), sc.window(z, NX, NX), sc.window(z, offset, NU)
    eq = sc.vmap(_stage_scaly(), N)(x, xnext, u, sc.broadcast(p[P_PW]), sc.broadcast(p[P_DT : P_DT + 1].reshape(()))).vec()
    stages = sc.vmap(_stage_cost_scaly(), N)(x, xnext, u, sc.broadcast(p[P_COST])).vec()
    xN, w, slack = z[NX * N : offset], p[P_COST], z[offset + NU * N]
    e = sc.stack([2.0 * (xN[0] / 2.0).sin(), xN[1], xN[2], xN[3]])
    cost = stages.sum() + sc.dot(e, p[P_TERMINAL].reshape((NX, NX)) @ e) + w[9] * 0.5 * (slack * slack + slack)
    phi = z[1:offset:NX]
    ineq = sc.concat([z[:NX] - p[:NX], phi + slack, phi - slack])
    return cost + sc.dot(lam_g, sc.concat([eq, ineq]))

  return lagrangian


# Inputs every backend is evaluated at, and the dense reference they are checked against.


def terminal_weight(pw: np.ndarray) -> np.ndarray:
  """The Riccati terminal weight for the learned dynamics linearized at the upright equilibrium."""
  zero = jnp.zeros(NX)
  A, B = jax.jacfwd(stage_jax, argnums=(0, 2))(zero, zero, jnp.zeros(NU), jnp.asarray(pw), DT)
  return solve_discrete_are(np.asarray(A), np.asarray(B), Q_END, R_END)


def sample_inputs(N: int, seed: int = 11) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """A decision vector spread over a swing-up's range of angles and speeds, the parameters, and mixed-sign multipliers."""
  rng = np.random.default_rng(seed)
  states = np.stack(
    [rng.uniform(-np.pi, np.pi, N + 1), rng.uniform(-2.0, 2.0, N + 1), rng.normal(scale=4.0, size=N + 1), rng.normal(scale=4.0, size=N + 1)], axis=1
  )
  z = np.concatenate([states.reshape(-1), rng.uniform(-TORQUE_LIMIT, TORQUE_LIMIT, NU * N), np.zeros(1)])
  pw = decoder_tail()
  p = np.concatenate([z[:NX], pw, [DT], COST, terminal_weight(pw).reshape(-1)])
  return z, p, np.linspace(-0.75, 0.75, n_g(N))


def dense_hessian(N: int, z: np.ndarray, p: np.ndarray, lam_g: np.ndarray) -> np.ndarray:
  """The dense Lagrangian Hessian by plain `jax.hessian`, independent of any sparsity detection."""
  return np.asarray(jax.jit(jax.hessian(lagrangian_jax), static_argnums=3)(z, p, lam_g, N))


def constraints_jax(z, p, N: int):
  """Dynamics, initial-state band and soft arm-angle bounds, in solver row order."""
  X = z[: NX * (N + 1)].reshape(N + 1, NX)
  U = z[NX * (N + 1) : -1].reshape(N, NU)
  eq = jax.vmap(stage_jax, in_axes=(0, 0, 0, None, None))(X[:-1], X[1:], U, p[P_PW], p[P_DT]).reshape(-1)
  return jnp.concatenate([eq, X[0] - p[:NX], X[:, 1] + z[-1], X[:, 1] - z[-1]])


def constraints_scaly(N: int) -> sc.Function:
  """The full solver constraint vector as a Scaly Function."""
  offset = NX * (N + 1)

  @sc.function(sc.arg("z", n_z(N)), sc.arg("p", N_P), outputs=sc.arg("g", n_g(N)), name=f"npmpc_constraints_N{N}")
  def constraints(z: sc.Expr, p: sc.Expr) -> sc.Expr:
    eq = sc.vmap(_stage_scaly(), N)(
      sc.window(z, 0, NX), sc.window(z, NX, NX), sc.window(z, offset, NU), sc.broadcast(p[P_PW]), sc.broadcast(p[P_DT : P_DT + 1].reshape(()))
    ).vec()
    phi, slack = z[1:offset:NX], z[-1]
    return sc.concat([eq, z[:NX] - p[:NX], phi + slack, phi - slack])

  return constraints
