"""Race-car NMPC from PREDICT-EPFL/scaly's benchmarks/problems/race_cars (main).

The reference trajectory is an input parameter; spline planning is outside the timed kernel.
Synthetic inputs match Scaly's sweep without its optional harvested closed-loop data.
"""

from functools import cache

import jax
import jax.numpy as jnp
import numpy as np
import scaly as sc

jax.config.update("jax_enable_x64", True)

NX, NU, NZ, N_PARAMS = 4, 2, 6, 7
CAR_LENGTH, CAR_WIDTH = 2.8, 1.5
PARAMS = np.array([1.5706, 0.05, 230.0, 4.950, 297.030, 16.665, 0.6784])


def n_z(N: int) -> int:
  return NZ * (N + 1)


def n_p(N: int) -> int:
  return NX * (N + 1) + N_PARAMS


def n_g(N: int) -> int:
  return NX * (N + 1) + 2 * N


def weights(N: int) -> np.ndarray:
  w = np.tile([1e-3, 2.0, 10.0, 20.0, 50.0, 20.0], (N + 1, 1))
  w[0, 2:] = 0.0
  w[-1, 2:] = [1000.0, 1000.0, 500.0, 1000.0]
  return w


def sample_inputs(N: int) -> dict[str, np.ndarray]:
  rng = np.random.default_rng(7)
  z = rng.normal(scale=0.4, size=n_z(N))
  p = np.concatenate([rng.normal(scale=0.4, size=NX * (N + 1)), PARAMS])
  return {"z": z, "p": p, "lam_f": np.array(1.0), "lam_g": np.linspace(-0.75, 0.75, n_g(N))}


def _continuous_dynamics(x, u, params):
  wheelbase, _, mass, c_m0, c_r0, c_r1, c_r2 = [params[i] for i in range(N_PARAMS)]
  phi, v = x[2], x[3]
  throttle, delta = u[0], u[1]
  beta = 0.5 * delta
  vx = v * beta.cos()
  lr = 0.5 * wheelbase
  return sc.stack(
    [
      v * (phi + beta).cos(),
      v * (phi + beta).sin(),
      v * beta.sin() / lr,
      (c_m0 * throttle - (c_r0 + c_r1 * vx + c_r2 * vx * vx) * (10 * vx).tanh()) / mass,
    ]
  )


def _rk4(x, u, params):
  dt = params[1]
  k1 = _continuous_dynamics(x, u, params)
  k2 = _continuous_dynamics(x + dt / 2 * k1, u, params)
  k3 = _continuous_dynamics(x + dt / 2 * k2, u, params)
  k4 = _continuous_dynamics(x + dt * k3, u, params)
  return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


@sc.function(sc.arg("z", NZ), sc.arg("p", NX), outputs=sc.arg("eq", NX), name="race_car_eq_initial")
def eq_initial(z: sc.Expr, p: sc.Expr) -> sc.Expr:
  return z[:NX] - p[:NX]


@sc.function(sc.arg("z", NZ), sc.arg("znext", NZ), sc.arg("params", N_PARAMS), outputs=sc.arg("eq", NX), name="race_car_eq_interstage")
def eq_interstage(z: sc.Expr, znext: sc.Expr, params: sc.Expr) -> sc.Expr:
  return _rk4(z[:NX], z[NX : NX + NU], params) - znext[:NX]


@sc.function(sc.arg("z", NZ), sc.arg("ref", NX), outputs=sc.arg("corridor", 2), name="race_car_corridor_stage")
def _corridor_stage(z: sc.Expr, ref: sc.Expr) -> sc.Expr:
  cos_ref, sin_ref = ref[2].cos(), ref[2].sin()
  dx, dy = z[0] - ref[0], z[1] - ref[1]
  e_lat = -sin_ref * dx + cos_ref * dy
  d_phi = z[2] - ref[2]
  reach = e_lat + 0.5 * CAR_LENGTH * d_phi.sin()
  half_width = 0.5 * CAR_WIDTH * d_phi.cos()
  return sc.stack([reach + half_width, reach - half_width])


@sc.function(sc.arg("z", NZ), sc.arg("ref", NX), sc.arg("params", N_PARAMS), outputs=sc.arg("residuals", NZ), name="race_car_cost_stage")
def _cost_stage(z: sc.Expr, ref: sc.Expr, params: sc.Expr) -> sc.Expr:
  c_m0, c_r0, c_r1, c_r2 = params[3], params[4], params[5], params[6]
  v_ref = ref[3]
  throttle_ref = (10.0 * v_ref).tanh() * (c_r0 + c_r1 * v_ref + c_r2 * v_ref * v_ref) / c_m0
  cos_ref, sin_ref = ref[2].cos(), ref[2].sin()
  dx, dy = z[0] - ref[0], z[1] - ref[1]
  return sc.stack([z[NX] - throttle_ref, z[NX + 1], cos_ref * dx + sin_ref * dy, -sin_ref * dx + cos_ref * dy, z[2] - ref[2], z[3] - v_ref])


@cache
def problem_scaly(N: int):
  lb, ub = np.full((N + 1, NZ), -np.inf), np.full((N + 1, NZ), np.inf)
  lb[:, 3], ub[:, 3] = 0.0, 5.0
  lb[:, NX:], ub[:, NX:] = [-500.0, -0.5], [500.0, 0.5]

  @sc.problem(vars=sc.arg("z", n_z(N)), params=sc.arg("p", n_p(N)), name=f"race_car_closed_loop_N{N}")
  def problem(z: sc.Expr, p: sc.Expr) -> sc.ProblemSpec[sc.Expr]:
    residuals = sc.vmap(_cost_stage, N + 1)(z, sc.window(p, 0, NX), sc.window(p, NX * (N + 1), 0)).vec()
    corridor = sc.vmap(_corridor_stage, N)(sc.window(z, NZ, NZ), sc.window(p, NX, NX)).vec()
    initial = eq_initial(z[:NZ], p[:NX])
    mapped = sc.vmap(eq_interstage, N)(sc.window(z, 0, NZ), sc.window(z, NZ, NZ), sc.window(p, NX * (N + 1), 0)).vec()
    return sc.ProblemSpec(
      minimize=sc.dot(sc.const(weights(N).reshape(-1)), residuals**2),
      eq=(sc.concat([initial, mapped]),),
      ineq=(sc.bounded(corridor, lo=sc.const(np.full(2 * N, -1.5)), hi=sc.const(np.full(2 * N, 1.5)), name="corridor"),),
      lb=sc.const(lb.reshape(-1)),
      ub=sc.const(ub.reshape(-1)),
    )

  return problem


@jax.custom_jvp
def tanh_jax(x):
  """Keep tanh's derivative in arithmetic understood by asdex 0.5.2."""
  return jnp.tanh(x)


@tanh_jax.defjvp
def _tanh_jvp(primals, tangents):
  (x,), (dx,) = primals, tangents
  y = tanh_jax(x)
  return y, dx * (1.0 - y * y)


def dynamics_jax(x, u, params):
  wheelbase, _, mass, c_m0, c_r0, c_r1, c_r2 = params
  beta = 0.5 * u[1]
  vx = x[3] * jnp.cos(beta)
  return jnp.stack(
    [
      x[3] * jnp.cos(x[2] + beta),
      x[3] * jnp.sin(x[2] + beta),
      x[3] * jnp.sin(beta) / (0.5 * wheelbase),
      (c_m0 * u[0] - (c_r0 + c_r1 * vx + c_r2 * vx * vx) * tanh_jax(10 * vx)) / mass,
    ]
  )


def step_jax(z, params):
  x, u, dt = z[:NX], z[NX:], params[1]
  k1 = dynamics_jax(x, u, params)
  k2 = dynamics_jax(x + dt / 2 * k1, u, params)
  k3 = dynamics_jax(x + dt / 2 * k2, u, params)
  k4 = dynamics_jax(x + dt * k3, u, params)
  return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def residuals_jax(z, ref, params):
  v = ref[3]
  throttle = tanh_jax(10 * v) * (params[4] + params[5] * v + params[6] * v * v) / params[3]
  dx, dy = z[0] - ref[0], z[1] - ref[1]
  c, s = jnp.cos(ref[2]), jnp.sin(ref[2])
  return jnp.stack([z[4] - throttle, z[5], c * dx + s * dy, -s * dx + c * dy, z[2] - ref[2], z[3] - v])


def corridor_jax(z, ref):
  dx, dy, phi = z[0] - ref[0], z[1] - ref[1], z[2] - ref[2]
  reach = -jnp.sin(ref[2]) * dx + jnp.cos(ref[2]) * dy + 0.5 * CAR_LENGTH * jnp.sin(phi)
  half_width = 0.5 * CAR_WIDTH * jnp.cos(phi)
  return jnp.stack([reach + half_width, reach - half_width])


def constraints_jax(z, p, N: int):
  stages, refs = z.reshape(N + 1, NZ), p[:-N_PARAMS].reshape(N + 1, NX)
  eq = jax.vmap(step_jax, in_axes=(0, None))(stages[:-1], p[-N_PARAMS:]) - stages[1:, :NX]
  corridor = jax.vmap(corridor_jax)(stages[1:], refs[1:])
  return jnp.concatenate([stages[0, :NX] - refs[0], eq.reshape(-1), corridor.reshape(-1)])


def lagrangian_jax(z, p, lam_f, lam_g, N: int):
  stages, refs = z.reshape(N + 1, NZ), p[:-N_PARAMS].reshape(N + 1, NX)
  residuals = jax.vmap(residuals_jax, in_axes=(0, 0, None))(stages, refs, p[-N_PARAMS:])
  return lam_f * jnp.sum(weights(N) * residuals**2) + lam_g @ constraints_jax(z, p, N)
