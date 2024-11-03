import numpy as onp

import jax
import jax.numpy as jnp
# from jax.config import config

# config.update('jax_enable_x64', True)  # use double-precision numbers

from jax import random
from jax import jit, lax, vmap, value_and_grad

from jax_md import util

import os
os.environ["XLA_FLAGS"]='--xla_force_host_platform_device_count=8' # require 8 cpus to run the code

import optax

import gc


f32 = jnp.float32
f64 = jnp.float64

Array = util.Array

def write_file(filename, vals):
  step, lr, loss, predicts,num_rigids, params = vals
  for key in params:
      if key == "diameters_seed":
        p_D = params["diameters_seed"]
      else:
        p_B = params[key]
  with open(filename, 'a') as f:
      print('{:6d}\t{:.16f}\t{:.16f}\t{:.16f}\t{:.16f}\t'.format(step, lr, loss, predicts, num_rigids),
              '\t'.join("%.16f" % d for d in p_D), '\t',
              '\t'.join("%.16f" % b for b in p_B), file=f)


def dict_mean(dict_list):
    mean_dict = {}
    for key in dict_list[0].keys():
        mean_dict[key] = onp.mean([d[key] for d in dict_list], axis=0)
    return mean_dict

def dict_sum(dict_list):
    sum_dict = {}
    for key in dict_list.keys():
        sum_dict[key] = onp.sum(dict_list[key], axis=0)
    return sum_dict


def optimize_batch_avegrad(
        run_fn,
        keyseed,
        boxsize,
        N,
        dimension,
        param_init,
        resfile,
        learning_rate_const=True,
        start_learning_rate=1e-3,
        afactor=0.1,
        target=10.0,
        mini_batch=16,
        opt_steps=1000,
        step_start=0,
        ndev=8):

    if learning_rate_const:
        learning_rate_scheduler = optax.constant_schedule(start_learning_rate)
    else:
        learning_rate_scheduler = optax.cosine_decay_schedule(start_learning_rate, opt_steps, afactor)

    optimizer = optax.rmsprop(learning_rate_scheduler)

    vg_loss = value_and_grad(run_fn, has_aux=True)
    keys = list(param_init.keys())
    zero_g = {key:jnp.zeros_like(param_init[key]) for key in keys}

    def loss_fn(p, R):
        (pred, check), g = vg_loss(p, R)
        # flat_g = jnp.hstack(([g[key] for key in keys]))
        # count_ireg_g = sum(abs(i) > 100.0 for i in flat_g)
        # rigid_reg_g = jnp.logical_and(count_ireg_g==0, check)
        dnu = pred - target
        new_g = {key: g[key]*dnu*2.0 for key in keys}
        grad = lax.cond(check, (), lambda _: new_g, (), lambda _: zero_g)
        loss = dnu**2
        loss = jnp.where(check, loss, 0.0)
        pred = jnp.where(check, pred, 0.0)

        grads = jax.lax.psum(grad, axis_name='obj')
        preds = jax.lax.psum(pred, axis_name='obj')
        losses = jax.lax.psum(loss, axis_name='obj')
        rigids = jax.lax.psum(check, axis_name='obj')
        return grads, preds, losses, rigids

    gen_conf = lambda x: random.uniform(x, (N, dimension), minval=0.0, maxval=boxsize, dtype=f64)
    gen_conf = jit(vmap(gen_conf))

    def update_step(params, opt_state, mini_batch, subkey):
        ns = 0
        predicts = 0.0
        losses = 0.0
        grads = zero_g
        subsamples = ndev
        replicated_params = jax.tree_map(lambda x: jnp.array([x] * subsamples), params)

        while (ns < mini_batch):
            subkey, splits = random.split(subkey)
            splits = random.split(splits, num=subsamples)
            Rs = gen_conf(splits)
            subgrad, subpred, subloss, subrigid = jax.pmap(loss_fn, axis_name='obj')(replicated_params, Rs)
            predicts += subpred[0]
            losses += subloss[0]
            ns += subrigid[0]
            gdic = subgrad
            tmp_d =  {k: grads.get(k, 0) + gdic.get(k, 0)[0] for k in set(grads)}
            grads = tmp_d

        @jit
        def ave_batch(real_nsamples, preds, Losses, gs):
            mean_pred = preds / real_nsamples
            mean_loss = Losses / real_nsamples
            factor = 2.0 / real_nsamples
            new_gs = {key: gs[key]*factor for key in keys}
            return new_gs, mean_loss, mean_pred

        mean_gs, mean_losses, mean_preds = ave_batch(ns, predicts, losses, grads)
        updates, opt_state = optimizer.update(mean_gs, opt_state)
        params = optax.apply_updates(params, updates)

        return params, opt_state, (mean_losses, mean_preds, ns, mean_gs, subkey)

    opt_state = optimizer.init(param_init)
    params = param_init
    lrs = [learning_rate_scheduler(i) for i in range(opt_steps)]

    optkey = random.PRNGKey(keyseed[0])
    for k in range(opt_steps):
      gc.collect()
      params_update, opt_state, auxes = update_step(params, opt_state, mini_batch, optkey)
      steps = step_start + k
      lr_tmp = lrs[k]
      loss_tmp, predicts_tmp, nrigid, grads, optkey = auxes
      
      params = params_update
      B = params["B_seed"]
      B_list = jnp.where(B<1e-6, 1.e-6, B)
      params["B_seed"] = jnp.array(B_list)

      write_file(resfile, (steps, lr_tmp, loss_tmp, predicts_tmp, nrigid, params))

    return params
