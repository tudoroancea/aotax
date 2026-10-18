"""Width-parameterized npmpc at Scaly's fixed horizon, with untrained weights."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import scaly as sc
from workloads import _npmpc as base

N = 12
OFFSET = base.NX * (N + 1)


@dataclass(frozen=True)
class Decoder:
  width: int

  @property
  def shapes(self):
    return ((self.width, 9), (self.width, self.width), (2, self.width))

  @property
  def blocks(self):
    return (
      ("x_scale_w", 5),
      ("x_scale_b", 5),
      ("y_inv_w", 2),
      ("y_inv_b", 2),
      *((f"w{i}", a * b) for i, (a, b) in enumerate(self.shapes)),
      ("bias", 2),
      ("latent", 4),
    )

  def slice(self, name):
    offset = 0
    for key, size in self.blocks:
      if key == name:
        return slice(offset, offset + size)
      offset += size
    raise KeyError(name)

  @property
  def n_pw(self):
    return sum(size for _, size in self.blocks)

  @property
  def n_p(self):
    return base.NX + self.n_pw + 27

  @property
  def params(self):
    end = base.NX + self.n_pw
    return slice(base.NX, end), end, slice(end + 1, end + 11), slice(end + 11, end + 27)

  def sample(self):
    rng = np.random.default_rng(11)
    states = np.stack(
      [rng.uniform(-np.pi, np.pi, N + 1), rng.uniform(-2.0, 2.0, N + 1), rng.normal(scale=4.0, size=N + 1), rng.normal(scale=4.0, size=N + 1)], axis=1
    )
    z = np.concatenate([states.reshape(-1), rng.uniform(-0.05, 0.05, N), np.zeros(1)])
    rng = np.random.default_rng(0)
    matrices = [rng.normal(scale=np.sqrt(2.0 / shape[1]), size=shape).reshape(-1) for shape in self.shapes]
    pw = np.concatenate([np.ones(5), np.zeros(5), np.ones(2), np.zeros(2), *matrices, np.zeros(2), base.LATENT])
    p = np.concatenate([z[:4], pw, [base.DT], base.COST, base.Q_END.reshape(-1)])
    return {"z": z, "p": p, "lam_f": np.array(1.0), "lam_g": np.linspace(-0.75, 0.75, base.n_g(N))}

  def stage(self, x, xnext, u, pw, dt):
    feat = jnp.stack([jnp.sin(x[0]), jnp.cos(x[0]), x[2], x[3], u[0]]) * pw[self.slice("x_scale_w")] + pw[self.slice("x_scale_b")]
    h = jnp.concatenate([feat, pw[self.slice("latent")]])
    for i, shape in enumerate(self.shapes[:-1]):
      h = 1.0 / (1.0 + jnp.exp(-(pw[self.slice(f"w{i}")].reshape(shape) @ h)))
    y = (pw[self.slice("w2")].reshape(self.shapes[-1]) @ h + pw[self.slice("bias")]) * pw[self.slice("y_inv_w")] + pw[self.slice("y_inv_b")]
    return x + jnp.concatenate([dt * (x[2:4] + y / 2.0), y]) - xnext

  def constraints(self, z, p):
    X, U = z[:OFFSET].reshape(N + 1, 4), z[OFFSET:-1].reshape(N, 1)
    pw, dt, _, _ = self.params
    eq = jax.vmap(self.stage, in_axes=(0, 0, 0, None, None))(X[:-1], X[1:], U, p[pw], p[dt]).reshape(-1)
    return jnp.concatenate([eq, X[0] - p[:4], X[:, 1] + z[-1], X[:, 1] - z[-1]])

  def lagrangian(self, z, p, lam_f, lam_g):
    X, U = z[:OFFSET].reshape(N + 1, 4), z[OFFSET:-1].reshape(N, 1)
    _, _, cost, terminal = self.params
    w, P = p[cost], p[terminal].reshape(4, 4)
    stages = jax.vmap(base.stage_cost_jax, in_axes=(0, 0, 0, None))(X[:-1], X[1:], U, w).sum()
    e = jnp.concatenate([jnp.array([2.0 * jnp.sin(X[-1, 0] / 2.0)]), X[-1, 1:]])
    objective = stages + e @ P @ e + w[9] * 0.5 * (z[-1] * z[-1] + z[-1])
    return lam_f * objective + lam_g @ self.constraints(z, p)

  def stage_scaly(self):
    @sc.function(
      sc.arg("x", 4),
      sc.arg("xnext", 4),
      sc.arg("u", 1),
      sc.arg("pw", self.n_pw),
      sc.arg("dt", ()),
      outputs=sc.arg("eq", 4),
      name=f"npmpc_stage_h{self.width}x{self.width}",
    )
    def stage(x, xnext, u, pw, dt):
      feat = sc.stack([x[0].sin(), x[0].cos(), x[2], x[3], u[0]]) * pw[self.slice("x_scale_w")] + pw[self.slice("x_scale_b")]
      h = sc.concat([feat, pw[self.slice("latent")]])
      for i, shape in enumerate(self.shapes[:-1]):
        h = 1.0 / (1.0 + (-(pw[self.slice(f"w{i}")].reshape(shape) @ h)).exp())
      y = (pw[self.slice("w2")].reshape(self.shapes[-1]) @ h + pw[self.slice("bias")]) * pw[self.slice("y_inv_w")] + pw[self.slice("y_inv_b")]
      return x + sc.concat([dt * (x[2:4] + y / 2.0), y]) - xnext

    return stage

  def nlp(self):
    pw, dt, cost, terminal = self.params
    lower, upper = np.full(base.n_z(N), -np.inf), np.full(base.n_z(N), np.inf)
    lower[OFFSET:-1], upper[OFFSET:-1] = -base.TORQUE_LIMIT, base.TORQUE_LIMIT
    lower[-1], upper[-1] = 0.0, base.SLACK_LIMIT
    lo = np.r_[np.full(4, -base.X0_BAND), np.full(N + 1, -base.PHI_LIMIT), np.full(N + 1, -np.inf)]
    hi = np.r_[np.full(4, base.X0_BAND), np.full(N + 1, np.inf), np.full(N + 1, base.PHI_LIMIT)]

    @sc.problem(vars=sc.arg("z", base.n_z(N)), params=sc.arg("p", self.n_p), name=f"npmpc_N{N}")
    def problem(z, p):
      eq = sc.vmap(self.stage_scaly(), N)(
        sc.window(z, 0, 4), sc.window(z, 4, 4), sc.window(z, OFFSET, 1), sc.broadcast(p[pw]), sc.broadcast(p[dt : dt + 1].reshape(()))
      ).vec()
      stages = sc.vmap(base._stage_cost_scaly(), N)(sc.window(z, 0, 4), sc.window(z, 4, 4), sc.window(z, OFFSET, 1), sc.broadcast(p[cost])).vec()
      xN, w, slack = z[4 * N : OFFSET], p[cost], z[-1]
      e = sc.stack([2.0 * (xN[0] / 2.0).sin(), xN[1], xN[2], xN[3]])
      objective = stages.sum() + sc.dot(e, p[terminal].reshape((4, 4)) @ e) + w[9] * 0.5 * (slack * slack + slack)
      phi = z[1:OFFSET:4]
      rows = sc.concat([z[:4] - p[:4], phi + slack, phi - slack])
      return sc.ProblemSpec(
        minimize=objective,
        eq=(eq,),
        ineq=(sc.bounded(rows, lo=sc.const(lo), hi=sc.const(hi), name="soft_bounds"),),
        lb=sc.const(lower),
        ub=sc.const(upper),
      )

    return problem._nlp
