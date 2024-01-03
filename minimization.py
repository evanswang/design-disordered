#@title running energy minimization
###################################
# @name : minimization.py
# @author : mzu
# @created date : 07/01/22
# @function : subfile including modified FIRE and program of minimization
# @ref: https://git.ist.ac.at/goodrichgroup/common_utils/-/blob/team/common_utils/minimization.py
###################################
"""
Defines a modified FIRE minimization.

The code implements the "optimization of the fast intertial relaxation engine" from [1].

[1] Julien Guénolé, Wolfram G.Nöhring, Aviral Vaid, Frédéric Houllé, Zhuocheng Xie, Aruna Prakash, and Erik Bitzek.
"Assessment and optimization of the fast inertial relaxation engine (fire) for energy minimization in atomistic
simulations and its implementation in lammps". Computational Materials Science Volume 175, 109584 (2020).
"""

import jax.numpy as jnp
from jax.config import config
config.update('jax_enable_x64', True)  # use double-precision numbers

from jax import jit, lax

from jax_md import minimize, util, quantity
from jax_md import dataclasses

f32 = jnp.float32
f64 = jnp.float64
Array = util.Array


def run_minimization_while_neighbor_list(energy_fn, neighbor_fn, R_init, shift,
                                         max_grad_thresh=1e-12, max_num_steps=1000000,
                                         step_inc=1000, verbose=False, **kwargs):
    nbrs = neighbor_fn.allocate(R_init)

    init, apply = minimize.fire_descent(jit(energy_fn), shift, **kwargs)
    apply = jit(apply)

    @jit
    def get_maxgrad(state):
        return jnp.amax(jnp.abs(state.force))

    @jit
    def body_fn(state_nbrs, t):
        state, nbrs = state_nbrs
        nbrs = neighbor_fn.update(state.position, nbrs)
        state = apply(state, neighbor=nbrs)
        return (state, nbrs), 0

    state = init(R_init, neighbor=nbrs)

    step = 0
    while step < max_num_steps:
        if verbose:
            print('minimization step {}'.format(step))
        rtn_state, _ = lax.scan(body_fn, (state, nbrs), step + jnp.arange(step_inc))
        new_state, nbrs = rtn_state
        # If the neighbor list overflowed, rebuild it and repeat part of
        # the simulation.
        if nbrs.did_buffer_overflow:
            print('Buffer overflow.')
            nbrs = neighbor_fn.allocate(state.position)
        else:
            state = new_state
            step += step_inc
            if get_maxgrad(state) <= max_grad_thresh:
                break

    if verbose:
        print('successfully finished {} steps.'.format(step * step_inc))

    return state.position, get_maxgrad(state), nbrs, step

def run_minimization_while(energy_fn, R_init, shift, min_style,
                           max_grad_thresh=1e-12, max_num_steps=10000000,
                           **kwargs):
  if min_style == 1:
      init, apply = minimize.fire_descent(jit(energy_fn), shift, **kwargs)
  else:
      init, apply = modif_fire(jit(energy_fn), shift, **kwargs)

  apply = jit(apply)

  @jit
  def get_maxgrad(state):
     return jnp.amax(jnp.abs(state.force))

  @jit
  def cond_fn(val):
     state, i = val
     return jnp.logical_and(get_maxgrad(state) > max_grad_thresh, i < max_num_steps)

  @jit
  def body_fn(val):
     state, i = val
     return apply(state), i + 1

  state = init(R_init)
  state, num_iterations = lax.while_loop(cond_fn, body_fn, (state, 0))

  return state.position, get_maxgrad(state), num_iterations

# This version is fully differentiable and internally jitted
def run_minimization_scan(energy_fn, R_init, shift, min_style, num_steps=5000, **kwargs):
    if min_style == 1:
        init, apply = minimize.fire_descent(jit(energy_fn), shift, **kwargs)
    else:
        init, apply = modif_fire(jit(energy_fn), shift, **kwargs)

    apply = jit(apply)

    @jit
    def scan_fn(state, i):
        return apply(state), 0.0

    state = init(R_init)
    state, _ = lax.scan(scan_fn, state, jnp.arange(num_steps))

    return state.position, jnp.amax(jnp.abs(energy_fn(state.position)))

def run_minimization_scan_nl_fn(neigh_fn, force_fn, R_init, shift, min_style,
                              forced_rebuilding=True,
                              max_grad_thresh=1e-12, max_num_steps=10000000,
                              scan_steps=100,
                              **kwargs):
  if min_style == 1:
      init, apply = minimize.fire_descent(jit(force_fn), shift, **kwargs)
  else:
      init, apply = modif_fire(jit(force_fn), shift, **kwargs)

  apply = jit(apply)

  nbrs = neigh_fn.allocate(R_init)
  state = init(R_init, neighbor=nbrs)

  @jit
  def get_maxgrad(state):
     return jnp.amax(jnp.abs(state.force))

  @jit
  def cond_fn(state, i):
     return jnp.logical_and(get_maxgrad(state) > max_grad_thresh, i < max_num_steps)

  @jit
  def update_nbrs(R, nbrs):
    return neigh_fn.update(R, nbrs)

  @jit
  def scan_fn(val, i):
    state, nbrs = val
    nbrs = update_nbrs(state.position, nbrs)
    return (apply(state, neighbor=nbrs), nbrs), 0.0

  @jit
  def update_fn(state, nbrs):
    scan_val, _ = lax.scan(scan_fn, (state,nbrs), jnp.arange(scan_steps))
    state, nbrs = scan_val
    return state, nbrs

  steps = 0
  while cond_fn(state, steps):
    new_state, nbrs = update_fn(state, nbrs)
    if nbrs.did_buffer_overflow:
      print("Rebuilding neighbor_list.")
      nbrs = neigh_fn.allocate(state.position)
    else:
      state = new_state
      steps += scan_steps
  
  num_iterations = steps
  return state.position, jnp.amax(jnp.abs(state.force)), nbrs, num_iterations

def run_minimization_while_nl_fn(neigh_fn, force_fn, R_init, shift, min_style,
                              forced_rebuilding=True,
                              max_grad_thresh=1e-12, max_num_steps=10000000,
                              **kwargs):
  if min_style == 1:
      init, apply = minimize.fire_descent(jit(force_fn), shift, **kwargs)
  else:
      init, apply = modif_fire(jit(force_fn), shift, **kwargs)

  apply = jit(apply)

  nbrs = neigh_fn.allocate(R_init)
  state = init(R_init, neighbor=nbrs)

  @jit
  def get_maxgrad(state):
     return jnp.amax(jnp.abs(state.force))

  @jit
  def cond_fn(state, i):
     return jnp.logical_and(get_maxgrad(state) > max_grad_thresh, i < max_num_steps)

  @jit
  def update_nbrs(R, nbrs):
    return neigh_fn.update(R, nbrs)

  steps = 0
  nrebuild = 0
  while cond_fn(state, steps):
    nbrs = update_nbrs(state.position, nbrs)
    new_state = apply(state, neighbor=nbrs)
    if nbrs.did_buffer_overflow:
      print("Rebuilding neighbor_list.")
      nbrs = neigh_fn.allocate(state.position)
      nrebuild += 1
    else:
      state = new_state
      steps += 1
#      print(steps, jnp.amax(jnp.abs(state.force)))
  
# steps = 0
# while cond_fn(state, steps):
#   print(steps)
#   nbrs = update_nbrs(state.position, nbrs)
#   overflow = nbrs.did_buffer_overflow
#   overflow_nbrs = neigh_fn.allocate(state.position)
#   new_state = apply(state, neighbor=nbrs)
#   def infinite_loop():
#       while True: pass
#   nbrs = lax.cond(overflow, (), lambda _: overflow_nbrs, (), lambda _: overflow_nbrs)
#   state = lax.cond(overflow, (), lambda _: state, (), lambda _: new_state)
#   steps = lax.cond(overflow, steps, lambda x: x, steps, lambda x: x+1)
#   print(steps, jnp.amax(jnp.abs(state.force)))

  num_iterations = steps
  return state.position, jnp.amax(jnp.abs(state.force)), nbrs, num_iterations

def run_minimization_while_nl_overflow_fn(force_fn, nbrs, R_init, shift, min_style,
                              forced_rebuilding=True,
                              max_grad_thresh=1e-12, max_num_steps=1000000,
                              **kwargs):
  if min_style == 1:
      init, apply = minimize.fire_descent(jit(force_fn), shift, **kwargs)
  else:
      init, apply = modif_fire(jit(force_fn), shift, **kwargs)

  apply = jit(apply)

  state = init(R_init, neighbor=nbrs)

  @jit
  def get_maxgrad(state):
     return jnp.amax(jnp.abs(state.force))
 
  @jit
  def cond_fn(val):
     state, nbrs, i = val
     cond_min = jnp.logical_and(get_maxgrad(state) > max_grad_thresh, i < max_num_steps)
     overflow = jnp.logical_not(nbrs.did_buffer_overflow)
     return jnp.logical_and(overflow, cond_min)

  @jit
  def body_fn(val):
     state, nbrs, i = val
     nbrs = nbrs.update(state.position)
     state = apply(state, neighbor=nbrs)
     return state, nbrs, i + 1

  state, nbrs_final, num_iterations = lax.while_loop(cond_fn, body_fn, (state, nbrs, 0))
  return state.position, jnp.amax(jnp.abs(state.force)), nbrs_final, num_iterations

def run_minimization_while_nl_unsafe_fn(force_fn, nbrs, R_init, shift, min_style,
                              forced_rebuilding=True,
                              max_grad_thresh=1e-12, max_num_steps=1000000,
                              **kwargs):
  if min_style == 1:
      init, apply = minimize.fire_descent(jit(force_fn), shift, **kwargs)
  else:
      init, apply = modif_fire(jit(force_fn), shift, **kwargs)

  apply = jit(apply)

  state = init(R_init, neighbor=nbrs)

  @jit
  def get_maxgrad(state):
     return jnp.amax(jnp.abs(state.force))
 
  @jit
  def cond_fn(val):
     state, nbrs, i = val
     cond_min = jnp.logical_and(get_maxgrad(state) > max_grad_thresh, i < max_num_steps)
     return cond_min

  @jit
  def body_fn(val):
     state, nbrs, i = val
     nbrs = nbrs.update(state.position)
     state = apply(state, neighbor=nbrs)
     return state, nbrs, i + 1

  state, nbrs_final, num_iterations = lax.while_loop(cond_fn, body_fn, (state, nbrs, 0))
  return state.position, jnp.amax(jnp.abs(state.force)), nbrs_final, num_iterations


@dataclasses.dataclass
class ModifFireState:
    position: Array
    velocity: Array
    force: Array
    dt: float
    alpha: float
    n_pos: int


def modif_fire(energy_or_force_fn, shift_fn,
               dt_start: float = 0.1,
               dt_max: float = 0.4,
               n_min: float = 5,
               f_inc: float = 1.1,
               f_dec: float = 0.5,
               alpha_start: float = 0.1,
               f_alpha: float = 0.99) -> ModifFireState:
    force = quantity.canonicalize_force(energy_or_force_fn)

    def init_fn(R: Array, **kwargs) -> ModifFireState:
        V = jnp.zeros_like(R)
        n_pos = jnp.zeros((), jnp.int32)
        F = force(R, **kwargs)
        return ModifFireState(R, V, F, dt_start, alpha_start, n_pos)

    def apply_fn(state: ModifFireState, **kwargs) -> ModifFireState:
        R, V, F_old, dt, alpha, n_pos = dataclasses.astuple(state)

        P = jnp.array(jnp.dot(jnp.reshape(F_old, (-1)), jnp.reshape(V, (-1))))

        n_pos = jnp.where(P > 0, n_pos + 1, 0)
        dt_choice = jnp.array([dt * f_inc, dt_max])
        dt = jnp.where(P > 0,
                       jnp.where(n_pos > n_min,
                                 jnp.min(dt_choice),
                                 dt),
                       dt)
        dt = jnp.where(P < 0, dt * f_dec, dt)
        alpha = jnp.where(P > 0,
                          jnp.where(n_pos > n_min,
                                    alpha * f_alpha,
                                    alpha),
                          alpha)
        alpha = jnp.where(P < 0, alpha_start, alpha)

        R = jnp.where(P < 0, shift_fn(R, -dt * V * 0.5, **kwargs), R)
        V = jnp.where(P < 0, 0, V)

        V = V + dt * F_old
        F_norm = jnp.sqrt(jnp.sum(F_old ** f32(2)) + f32(1e-6))
        V_norm = jnp.sqrt(jnp.sum(V ** f32(2)))

        V = (1.0 - alpha) * V + alpha * F_old * V_norm / F_norm
        R = shift_fn(R, dt * V, **kwargs)

        F = force(R, **kwargs)

        return ModifFireState(R, V, F, dt, alpha, n_pos)

    return init_fn, apply_fn
