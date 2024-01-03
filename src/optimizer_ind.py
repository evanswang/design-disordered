###################################
# @name : optimizer_individual.py
# @author : mzu
# @created date : 28/09/22
# @function : optimization functions
# @ref: 
###################################
import numpy as onp

import jax
import jax.numpy as jnp
from jax.config import config

config.update('jax_enable_x64', True)  # use double-precision numbers

from jax import random
from jax import jit, lax, vmap, value_and_grad, grad

from jax_md import util

from jaxopt.implicit_diff import custom_root

import os
#os.environ["XLA_FLAGS"]='--xla_force_host_platform_device_count=16'

from functools import partial
import optax

from utils import check_c

import gc

f32 = jnp.float32
f64 = jnp.float64

Array = util.Array

def write_file(filename, vals):
  step, lr, loss, predicts, params = vals
  p_flat = []
  for values in params.values():
      p_flat.extend(values)

  with open(filename, 'a') as f:
      print('{:6d}\t{:.16f}\t{:.16e}\t{:.16f}\t'.format(step, lr, loss, predicts),
              '\t'.join("%.16f" % d for d in p_flat), file=f) 

def write_file2_Ba(filename, vals):
  step, lr, loss, predicts, params = vals
  p_B = params["B_seed"]
  p_a = params["alpha_seed"]
  with open(filename, 'a') as f:
      print('{:6d}\t{:.16f}\t{:.16e}\t{:.16f}\t'.format(step, lr, loss, predicts),
              '\t'.join("%.16f" % b for b in p_B), '\t',
              '\t'.join("%.16f" % a for a in p_a), file=f)

def write_file2_D(filename, vals):
  step, lr, loss, predicts, params = vals
  for key in params:
      if key == "diameters_seed":
        p_D = params["diameters_seed"]
      else:
        p_B = params[key]
  with open(filename, 'a') as f:
      print('{:6d}\t{:.16f}\t{:.16e}\t{:.16f}\t'.format(step, lr, loss, predicts),
              '\t'.join("%.16f" % d for d in p_D), '\t',
              '\t'.join("%.16f" % b for b in p_B), file=f)

def write_file2C_D(filename, vals):
  step, lr, loss, predicts, params = vals
  for key in params:
      if key == "diameters_seed":
        p_D = params["diameters_seed"]
      else:
        p_B = params[key]
  with open(filename, 'a') as f:
      print('{:6d}\t{:.16f}\t{:.16e}\t'.format(step, lr, loss),
              '\t'.join("%.16f" % c for c in predicts), '\t',
              '\t'.join("%.16f" % d for d in p_D), '\t',
              '\t'.join("%.16f" % b for b in p_B), file=f)

def write_confile(filename, params, R):
    dimension = 2
    with open(filename, 'w') as f:
        print('\t'.join("%.16f" % d for d in params["diameters_seed"]), file=f)
        print('\t'.join("%.16f" % b for b in params["B_seed"]), file=f)
#    print('\t'.join("%.16f" % a for a in params["alpha_seed"]), file=f)
        if dimension == 2:
            for i, r in enumerate(R):
                print('{:.16f}\t{:.16f}'.format(r[0], r[1]), file=f)
        elif dimension == 3:
            for i, r in enumerate(R):
                print('{:.16f}\t{:.16f}\t{:.16f}'.format(r[0], r[1], r[2]), file=f)

def optimize_Rf_meta_fn(run_fn, Rinit, 
        param_init,
        resfile, logfile,
        start_learning_rate=1e-6,
        start_meta_learning_rate=3e-4,
        target=-0.1,
        ltol=1.e-6,
        opt_steps=2000,
        step_start=0,
        nothreshold=False,
        conout=False,
        path=None):

    opt = optax.inject_hyperparams(optax.rmsprop)(learning_rate=start_learning_rate)

    meta_opt = optax.adam(learning_rate=start_meta_learning_rate)

    def loss_fn(p, R):
      predict, Rf = run_fn(p, R)
      loss2 = (predict-target)**2
#      loss2 = jnp.where(predict==0.0, 0.0, loss2)
      return loss2, (predict, Rf)

    vg_loss = value_and_grad(loss_fn, has_aux=True)

# normalize gradients if needed
    def norm_grad(g):
        g_D = g["diameters_seed"]
        g_D_max = jnp.amax(jnp.abs(g_D))
        g_B = g["B_seed"]
        g_B_max = jnp.amax(jnp.abs(g_B))
        g_D = g_D / g_D_max
        g_B = g_B / g_B_max
        g_D = jnp.nan_to_num(g_D)
        g_B = jnp.nan_to_num(g_B)
        return {"diameters_seed":g_D, "B_seed":g_B}

    def update_step(R_init, params, state):
        (loss, aux), grad = vg_loss(params, R_init)
        # norm_g = norm_grad(grad)
        updates, state = opt.update(grad, state)
        params = optax.apply_updates(params, updates)
        return params, grad, state, loss, aux

    def outer_loss(eta, params, state, R):
      state.hyperparams['learning_rate'] = jax.nn.sigmoid(eta)
      params, g, state, loss, aux_inner = update_step(R, params, state)
      predict = aux_inner[0]
      Rf = aux_inner[1]
      return loss, (params, g, state, Rf, predict)

    def outer_step(eta, params, meta_state, state, R):
      (loss, aux), meta_grad = value_and_grad(outer_loss, has_aux=True)(eta, params, state, R)
      meta_grad = jnp.nan_to_num(meta_grad)
      params, g, state, Rf, predict = aux

      meta_updates, meta_state = meta_opt.update(meta_grad, meta_state)
      eta = optax.apply_updates(eta, meta_updates)
      return eta, params, meta_state, state, loss, predict, g, Rf

    num_keys = len(param_init)
    if num_keys == 1:
        write_fn = write_file1
    elif num_keys == 2:
        if "diameters_seed" in param_init:
            write_fn = write_file2_D
        else:
            write_fn = write_file2_Ba

    state = opt.init(param_init)
    params = param_init
    eta = -jnp.log(1.0 / start_learning_rate - 1)
    meta_state = meta_opt.init(eta)

    R0 = Rinit
    loss = 100.0
    k = -1
    ncount = 0
    keys = list(param_init.keys())
    while (k < opt_steps and loss > ltol):
        eta, update_params, meta_state, state, loss, nu, grad, Rf = jit(outer_step)(eta, params, meta_state, state, R0)
        if "B_seed" in update_params:
            B = update_params["B_seed"]
            B_list = jnp.where(B<0.0, 0.0, B)
            update_params["B_seed"] = jnp.array(B_list)
        if loss == 0.0:
            update_params = {key: params[key] + 1.e-6*grad_tmp[key] for key in keys}
            loss = 100.0
            ncount += 1
        else:
          R0 = Rf
          ncount = 0

        lr_tmp = jax.nn.sigmoid(eta)

        k += 1
        steps = step_start + k

        write_fn(resfile, (steps, lr_tmp, loss, nu, params))
        write_fn(logfile, (steps, lr_tmp, loss, nu, grad))

        if ncount > 100:
            loss = ltol * 0.1

        if nothreshold:
            if nu > 0.0:
                loss = ltol * 0.1

        if conout:
            if jnp.mod(k, 10)==0:
                confile = str(path) + "_lrs_" + str(k) + ".con"
                write_confile(confile, params, Rf)

        params = update_params
        
    return params, Rf

def optimize_C_Rf_meta_fn(run_fn, Rinit,
                          param_init,
                          resfile, logfile,
                          start_learning_rate=1e-6,
                          start_meta_learning_rate=3e-4,
                          target=None,
                          ltol=1.e-6,
                          opt_steps=2000,
                          step_start=0,
                          nothreshold=False,
                          conout=False,
                          path=None):
    opt = optax.inject_hyperparams(optax.rmsprop)(learning_rate=start_learning_rate)
    meta_opt = optax.adam(learning_rate=start_meta_learning_rate)

    def loss_fn(p, R):
        predict, Rf = run_fn(p, R)
        loss2 = sum((predict - target) ** 2)
        #   loss2 = jnp.where(predict==0.0, 0.0, loss2)
        return loss2, (predict, Rf)

    vg_loss = value_and_grad(loss_fn, has_aux=True)

    def norm_grad(g):
        g_D = g["diameters_seed"]
        g_D_max = jnp.amax(jnp.abs(g_D))
        g_B = g["B_seed"]
        g_B_max = jnp.amax(jnp.abs(g_B))
        g_D = g_D / g_D_max
        g_B = g_B / g_B_max
        return {"diameters_seed": g_D, "B_seed": g_B}

    def update_step(R_init, params, state):
        (loss, aux), grad = vg_loss(params, R_init)
        # norm_g = norm_grad(grad)
        updates, state = opt.update(grad, state)
        params = optax.apply_updates(params, updates)
        return params, grad, state, loss, aux

    def outer_loss(eta, params, state, R):
        state.hyperparams['learning_rate'] = jax.nn.sigmoid(eta)
        params, g, state, loss, aux_inner = update_step(R, params, state)
        Rf = aux_inner[1]
        predict = aux_inner[0]
        return loss, (params, g, state, Rf, predict)

    def outer_step(eta, params, meta_state, state, R):
        (loss, aux), meta_grad = value_and_grad(outer_loss, has_aux=True)(eta, params, state, R)
        params, g, state, Rf, predict = aux

        meta_updates, meta_state = meta_opt.update(meta_grad, meta_state)
        eta = optax.apply_updates(eta, meta_updates)
        return eta, params, meta_state, state, loss, predict, g, Rf

    num_keys = len(param_init)
    if num_keys == 1:
        write_fn = write_file1
    elif num_keys == 2:
        if "diameters_seed" in param_init:
            write_fn = write_file2C_D
        else:
            write_fn = write_file2_Ba

    state = opt.init(param_init)
    params = param_init
    eta = -jnp.log(1.0 / start_learning_rate - 1)
    meta_state = meta_opt.init(eta)

    R0 = Rinit
    loss = 100.0
    k = -1
    ncount = 0
    keys = list(param_init.keys())
    while (k < opt_steps and loss > ltol):
        eta, update_params, meta_state, state, loss, C_ijkl, grad, Rf = jit(outer_step)(eta, params, meta_state, state,
                                                                                        R0)
        if "B_seed" in update_params:
            B = update_params["B_seed"]
            B_list = jnp.where(B < 0.0, 0.0, B)
            update_params["B_seed"] = jnp.array(B_list)
        if loss == 0.0:
            update_params = {key: params[key] + 1.e-6 * grad_tmp[key] for key in keys}
            loss = 100.0
            ncount += 1
        else:
            R0 = Rf
            ncount = 0

        lr_tmp = jax.nn.sigmoid(eta)

        #       (loss, aux), grad = vg_loss(params, Rf)
        #       nu, Rf0 = aux

        k += 1
        steps = step_start + k

        write_fn(resfile, (steps, lr_tmp, loss, C_ijkl, params))
        write_fn(logfile, (steps, lr_tmp, loss, C_ijkl, grad))

        if ncount > 100:
            loss = ltol * 0.1

        if nothreshold:
            if nu > 0.0:
                loss = ltol * 0.1

        if conout:
            if jnp.mod(k, 100) == 0:
                confile = str(path) + "_lrs_" + str(k) + ".con"
                write_confile(confile, params, Rf)

        params = update_params

    return params, Rf
