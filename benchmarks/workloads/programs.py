"""Complete programs with explicit size axes."""

from functools import partial

import jax
import numpy as np
from workload import ProgramWorkload
from workloads import _programs as p


def axes(size, names):
  values = tuple(int(n) for n in size.split("x"))
  if len(values) != len(names) or any(n < 1 for n in values):
    raise ValueError(f"expected positive {'x'.join(names)}, got {size!r}")
  return dict(zip(names, values, strict=True))


def rollout_inputs(size):
  return p.chain_inputs(**axes(size, ("M", "N")))


def newton_inputs(size):
  n = axes(size, ("n",))["n"]
  return np.zeros(n), np.array(1.0)


def fit_inputs(size):
  a = axes(size, ("observations", "parameters", "iterations"))
  terms = (a["parameters"] - 1) // 2
  if terms < 1 or a["parameters"] != 2 * terms + 1:
    raise ValueError("fit parameter count must be odd and at least 3")
  t = np.linspace(0, 2, a["observations"])
  rates = np.linspace(0.7, 1.7, terms)
  amplitudes = np.linspace(1.2, 0.6, terms)
  z = np.r_[0.9 * amplitudes, np.log(rates) + 0.15, 0.1]
  y = np.exp(-t[:, None] * rates) @ amplitudes + 0.2
  return ({"z": z, "data": (t, y)},)


def fit_function(size):
  iterations = axes(size, ("observations", "parameters", "iterations"))["iterations"]
  return lambda values: p.least_squares(values["z"], *values["data"], iterations)


def qp_inputs(size):
  a = axes(size, ("M", "N", "iterations"))
  return p.chain_qp_inputs(a["M"], a["N"])


def qp_function(size):
  return partial(p.box_ipm, iterations=axes(size, ("M", "N", "iterations"))["iterations"])


def batched_inputs(size):
  a = axes(size, ("batch", "n"))
  return np.zeros((a["batch"], a["n"])), np.linspace(0.5, 2.0, a["batch"])


WORKLOADS = (
  ProgramWorkload(
    "chain_rollout",
    ("3x12", "3x50", "3x200", "5x50", "10x50"),
    partial(axes, names=("M", "N")),
    rollout_inputs,
    lambda _: p.chain_scan,
    lambda _: ((1e-8, 1e-8),) * 2,
  ),
  ProgramWorkload(
    "newton",
    ("4", "8", "16", "32", "64"),
    partial(axes, names=("n",)),
    newton_inputs,
    lambda _: p.newton,
    lambda _: ((1e-8, 1e-10), (0, 0), (1e-8, 1e-10)),
  ),
  ProgramWorkload(
    "gauss_newton",
    ("16x3x8", "64x3x8", "64x5x12"),
    partial(axes, names=("observations", "parameters", "iterations")),
    fit_inputs,
    fit_function,
    lambda _: ((1e-7, 1e-9), (1e-7, 1e-9), (0, 0)),
  ),
  ProgramWorkload(
    "chain_qp",
    ("3x4x15", "3x12x15", "5x12x15"),
    partial(axes, names=("M", "N", "iterations")),
    qp_inputs,
    qp_function,
    lambda _: ((1e-7, 1e-8),) * 4 + ((0, 0),),
  ),
  ProgramWorkload(
    "batched_newton",
    ("1x8", "4x8", "16x8", "64x8"),
    partial(axes, names=("batch", "n")),
    batched_inputs,
    lambda _: jax.vmap(p.newton),
    lambda _: ((1e-8, 1e-10), (0, 0), (1e-8, 1e-10)),
  ),
)
