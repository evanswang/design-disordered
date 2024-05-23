import jax
import jax.numpy as jnp
from jax.config import config

config.update('jax_enable_x64', True)

from jax import jit, value_and_grad, random

from jax_md import util

import optax


f32 = jnp.float32
f64 = jnp.float64

Array = util.Array

def write_file(filename, vals):
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

def write_confile(filename, params, R):
    dimension = 2
    with open(filename, 'w') as f:
        print('\t'.join("%.16f" % d for d in params["diameters_seed"]), file=f)
        print('\t'.join("%.16f" % b for b in params["B_seed"]), file=f)
        if dimension == 2:
            for i, r in enumerate(R):
                print('{:.16f}\t{:.16f}'.format(r[0], r[1]), file=f)
        elif dimension == 3:
            for i, r in enumerate(R):
                print('{:.16f}\t{:.16f}\t{:.16f}'.format(r[0], r[1], r[2]), file=f)


def optimize_Rf_meta_fn(run_fn, Rinit, key,
        param_init,
        resfile,
        start_learning_rate=1e-6,
        start_meta_learning_rate=3e-4,
        target=-0.1,
        ltol=1.e-6,
        opt_steps=2000,
        step_start=0):

    opt = optax.inject_hyperparams(optax.rmsprop)(learning_rate=start_learning_rate)
    meta_opt = optax.adam(learning_rate=start_meta_learning_rate)

    def loss_fn(p, R):
      predict, Rf = run_fn(p, R)
      loss2 = (predict-target)**2
      loss2 = jnp.where(predict==0.0, 0.0, loss2)
      return loss2, (predict, Rf)

    vg_loss = value_and_grad(loss_fn, has_aux=True)

    def update_step(R_init, params, state):
        (loss, aux), grad = vg_loss(params, R_init)
        updates, state = opt.update(grad, state)
        params = optax.apply_updates(params, updates)
        return params, grad, state, aux

    def outer_loss(eta, params, state, R):
      state.hyperparams['learning_rate'] = jax.nn.sigmoid(eta)
      params, g, state, aux_inner = update_step(R, params, state)
      Rf = aux_inner[1]
      loss, aux = loss_fn(params, Rf)
      predict = aux[0]
      return loss, (params, g, state, Rf, predict)

    def outer_step(eta, params, meta_state, state, R):
      (loss, aux), grad = value_and_grad(outer_loss, has_aux=True)(eta, params, state, R)
      params, g, state, Rf, predict = aux

      meta_updates, meta_state = meta_opt.update(grad, meta_state)
      eta = optax.apply_updates(eta, meta_updates)
      return eta, params, meta_state, state, loss, predict, g, Rf

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
        eta, update_params, meta_state, state, loss, pred, grad, Rf = jit(outer_step)(eta, params, meta_state, state, R0)
        if "B_seed" in update_params:
            B = update_params["B_seed"]
            B_list = jnp.where(B<0.0, 0.0, B)
            update_params["B_seed"] = jnp.array(B_list)
        if loss == 0.0:
            # key, split = random.split(key, num=2)
            # subplits = random.split(split, num=len(params))
            # randoms  = random.normal(subplits, )
            update_params = {key: params[key] + 1.e-6*grad[key] for key in keys}
            loss = 100.0
            ncount += 1
        else:
          R0 = Rf
          ncount = 0

        lr_tmp = jax.nn.sigmoid(eta)

        k += 1
        steps = step_start + k

        write_file(resfile, (steps, lr_tmp, loss, pred, params))

        params = update_params

    return params, Rf

