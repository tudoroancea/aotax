"""Complete CPU programs from the H0 spike."""

import jax
import jax.numpy as jnp
import numpy as np
from workloads import _chain as chain


def bratu(z, lam):
  h = 1 / (z.shape[-1] + 1)
  padded = jnp.pad(z, (1, 1))
  return (2 * z - padded[:-2] - padded[2:]) / h**2 - lam * jnp.exp(z)


def newton(z0, lam):
  def cond(state):
    i, z = state
    return (i < 20) & (jnp.max(jnp.abs(bratu(z, lam))) > 1e-10)

  def body(state):
    i, z = state
    return i + 1, z - jnp.linalg.solve(jax.jacfwd(bratu)(z, lam), bratu(z, lam))

  i, z = jax.lax.while_loop(cond, body, (jnp.int32(0), z0))
  return z, i, bratu(z, lam)


def chain_scan(x0, us, p):
  M = (x0.size // 3 + 3) // 2

  def step(x, u):
    y = chain.step_jax(jnp.concatenate([x, u]), p, M)
    return y, y

  return jax.lax.scan(step, x0, us)


def chain_inputs(M, N):
  return chain.initial_state(M), np.zeros((N, 3)), chain.PARAMS.copy()


def least_squares(z, t, y, iterations):
  terms = (z.size - 1) // 2

  def residual(q):
    return jnp.exp(-t[:, None] * jnp.exp(q[terms : 2 * terms])) @ q[:terms] + q[-1] - y

  def body(_, q):
    r = residual(q)
    j = jax.jacfwd(residual)(q)
    l = jnp.linalg.cholesky(j.T @ j + 1e-5 * jnp.eye(q.size))
    v = jax.scipy.linalg.solve_triangular(l, j.T @ r, lower=True)
    return q - jax.scipy.linalg.solve_triangular(l.T, v, lower=False)

  z = jax.lax.fori_loop(0, iterations, body, z)
  return z, residual(z), jnp.int32(iterations)


def box_ipm(q, c, g, h, iterations):
  n, m = c.size, h.size

  def body(_, state):
    x, s, z = state
    rd, rp = q @ x + c + g.T @ z, g @ x + s - h
    rc = s * z - 0.1 * jnp.dot(s, z) / m
    a = q + g.T @ ((z / s)[:, None] * g)
    dx = jnp.linalg.solve(a, -rd + g.T @ ((rc - z * rp) / s))
    ds = -rp - g @ dx
    dz = (-rc - z * ds) / s
    alpha = jnp.minimum(
      1.0,
      jnp.minimum(jnp.min(jnp.where(ds < 0, -0.99 * s / ds, jnp.inf)), jnp.min(jnp.where(dz < 0, -0.99 * z / dz, jnp.inf))),
    )
    return x + alpha * dx, s + alpha * ds, z + alpha * dz

  x, s, z = jax.lax.fori_loop(0, iterations, body, (jnp.zeros(n), jnp.ones(m), jnp.ones(m)))
  residuals = jnp.stack([jnp.max(jnp.abs(q @ x + c + g.T @ z)), jnp.max(jnp.abs(g @ x + s - h)), jnp.dot(s, z) / m])
  return x, s, z, residuals, jnp.int32(iterations)


def chain_qp_inputs(M, N):
  x, _, p = chain_inputs(M, N)
  end, vel = 3 * (M - 2), 3 * (M - 1)

  def residual(uflat):
    u = uflat.reshape(N, 3)
    _, ys = chain_scan(jnp.asarray(x), u, jnp.asarray(p))
    xs = jnp.concatenate([jnp.asarray(x)[None, :], ys])
    return jnp.concatenate(
      [
        (jnp.sqrt(chain.Q_END / N) * (xs[:-1, end : end + 3] - jnp.asarray(chain.END_REF))).ravel(),
        (jnp.sqrt(chain.Q_VEL / N) * xs[:-1, vel:]).ravel(),
        (jnp.sqrt(chain.R_U / N) * u).ravel(),
        jnp.sqrt(chain.Q_END_TERMINAL) * (xs[-1, end : end + 3] - jnp.asarray(chain.END_REF)),
      ]
    )

  u0 = jnp.zeros(3 * N)
  r = np.asarray(jax.jit(residual)(u0))
  j = np.asarray(jax.jit(jax.jacfwd(residual))(u0))
  return j.T @ j, j.T @ r, np.concatenate([np.eye(3 * N), -np.eye(3 * N)]), np.ones(6 * N)
