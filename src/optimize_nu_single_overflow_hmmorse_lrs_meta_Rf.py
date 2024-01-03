###################################
# @name : optimization using previous minimized configuration via meta-learning
# @author : mzu
# @created date : 14/11/22
###################################
import time
import numpy as onp
import argparse

import jax
import jax.numpy as jnp
from jax.config import config

config.update('jax_enable_x64', True)  # use double-precision numbers

from jax import random
from jax import jit, lax, vmap, value_and_grad, grad

from jax_md import space, energy, util, quantity, elasticity, partition, smap

from jaxopt.implicit_diff import custom_root

from minimization import run_minimization_while_nl_overflow_fn

from optimizer_ind import optimize_Rf_meta_fn

import os

import functools
import optax

f32 = jnp.float32
f64 = jnp.float64
Array = util.Array
maybe_downcast = util.maybe_downcast
DisplacementOrMetricFn = space.DisplacementOrMetricFn

def box_at_packing_fraction(sigmas, Ns, phi, dimension):
  sphere_volume_2d = lambda s, n: (jnp.pi / f32(4)) * n * s**2
  sphere_volume_3d = lambda s, n: (jnp.pi / f32(6)) * n * s**3
  if dimension == 2:
    sphere_volume = sphere_volume_2d
  elif dimension == 3:
    sphere_volume = sphere_volume_3d
  
  sphere_volume_total = jnp.sum(vmap(sphere_volume, in_axes=(0,0))(jnp.array(sigmas), jnp.array(Ns)))
  return (sphere_volume_total / phi) ** (1/dimension)

@jit
def _vector2dsymmat(v, zeros):
  n = zeros.shape[0]
  assert v.shape == (
    n * (n + 1) / 2,
  ), f"The input must have shape jnp.int16(((1 + 8 * v.shape[0]) ** 0.5 - 1) / 2) = {(n * (n + 1) / 2,)}, got {v.shape} instead."
  ind = jnp.triu_indices(n)
  return zeros.at[ind].set(v).at[(ind[1], ind[0])].set(v)

@jit
def vector2dsymmat(v):
  """ Convert a vector into a symmetric matrix.
  Args:
    v: vector of length (n*(n+1)/2,)
  Return:
    symmetric matrix m of shape (n,n) that satisfies
    m[jnp.triu_indices_from(m)] == v
  Example:
    v = jnp.array([0,1,2,3,4,5])
    returns: [[ 0, 1, 2],
              1, 3, 4],
              2, 4, 5]]
  """
  n = int(((1 + 8 * v.shape[0]) ** 0.5 - 1) / 2)
  return _vector2dsymmat(v, jnp.zeros((n, n), dtype=v.dtype))

def diameters_to_sigma_matrix(diameters):
  return vmap(vmap(lambda d1,d2: (d1 + d2) * 0.5, in_axes=(None, 0)), in_axes=(0, None))(diameters, diameters)

def check_c(C):
  C_dict = elasticity.extract_elements(C)
  cxxxx = C_dict["cxxxx"]
  cyyyy = C_dict["cyyyy"]
  cxxyy = C_dict["cxxyy"]
  cxyxy = C_dict["cxyxy"]
  cyyxy = C_dict["cyyxy"]
  cxxxy = C_dict["cxxxy"]
  C_matrix = jnp.array([[cxxxx, cxxyy, 2.0*cxxxy],[cxxyy, cyyyy, 2.0*cyyxy], [2.0*cxxxy, 2.0*cyyxy, 4.0*cxyxy]])
  eigens = jnp.linalg.eigvalsh(C_matrix)
  positive_C = jnp.all(jnp.linalg.eigvals(C_matrix)>0)
  return positive_C

def set_cutoff(alpha):
    def func_to_solve(x, Etol=1e-6, **kwargs):
        return jnp.exp(-2. * alpha * x) - 2. * jnp.exp(-alpha * x) + Etol

    func = jit(func_to_solve)
    root = fsolve(func, 1.0)[0]
    return root

def harmonic_morse(dr: Array,
        epsilon: Array=5.0,
        alpha: Array=5.0,
        sigma: Array=1.0,
        k: Array=50.0, **kwargs) -> Array:
  U = jnp.where(dr < sigma,
               0.5 * k * (dr - sigma)**2 - epsilon,
               epsilon * (jnp.exp(-2. * alpha * (dr - sigma)) - 2. * jnp.exp(-alpha * (dr - sigma)))
               )
  return jnp.array(U, dtype=dr.dtype)

@jit
def harmonic_morse_cutoff(dr,
        epsilon=0.1,
        alpha=20.0,
        sigma=1.0,
        delta=2.0,
        k=5.0, **kwargs):
  r_onset = 5.0 / alpha + sigma
  r_cutoff = 9.9 / alpha + sigma
  r_o = r_onset ** f64(2)
  r_c = r_cutoff ** f64(2)
  def smooth_fn(dx):
      x = dx ** f64(2)
      inner = jnp.where(dx < r_cutoff,
                     (r_c - x)**2 * (r_c + 2 * x - 3 * r_o) / (r_c - r_o)**3,
                     0)
      return jnp.where(dx < r_onset, 1, inner)

  k_delta = k / delta
  dr_sigma = f64(dr / sigma)
  a_dr = alpha * (dr - sigma)
  U = jnp.where(dr_sigma < 1,
        k_delta * (sigma - dr)**f64(delta) - epsilon,
        (epsilon * (jnp.exp(-2. * a_dr) - 2. * jnp.exp(-a_dr)))*smooth_fn(dr))

  return jnp.nan_to_num(U)




def setup(params,
          N=128,
          dimension=2,
          nspecies=2,
          alpha=5.0,
          B=0.1,
          sigma=1.0,
          k=5.0,
          density=1.0,
          box_size=None,
          dt_start=0.001,
          dt_max=0.1,
          Ftol=1.e-12):
  """ Set up the system and return a function that calculates P((D,B), R_init)
      Differentiating over this function will use implicit diff for the minimization.
  """
  
  N_s = int(N // nspecies)
  Ns = N_s * jnp.ones(nspecies, dtype=int)
  if box_size == None:
    box_size = quantity.box_size_at_number_density(N, density, dimension)
  displacement, shift = space.periodic(box_size)

  species_seed = jnp.arange(nspecies)
  species_vec = jnp.repeat(species_seed, N_s)

  def energy_hm_fn_nl(p):
    sigma = diameters_to_sigma_matrix(p["diameters_seed"])
    B = vector2dsymmat(p["B_seed"])
 #   alpha = vector2dsymmat(p["alpha_seed"])
    energy_fn = smap.pair_neighbor_list(
            harmonic_morse_cutoff,
            space.canonicalize_displacement_or_metric(displacement),
            species=species_vec,
            epsilon=B,
            alpha=alpha,
            sigma=sigma,
            k=k)
    return energy_fn

###### initialize nbrs ######
###### small alpha ##########
  if alpha < 50.0:
    if "diameters_seed" in params:
        rcutoff_neigh = (9.9 / alpha + jnp.amax(params["diameters_seed"])) * 1.0
    else:
        rcutoff_neigh = (9.9 / alpha + jnp.amax(sigma)) * 1.0
  else:
    if "diameters_seed" in params:
        rcutoff_neigh = (9.9 / alpha + jnp.amax(params["diameters_seed"])) * 2.0
    else:
        rcutoff_neigh = (9.9 / alpha + jnp.amax(sigma)) * 2.0
  if rcutoff_neigh > jnp.min(box_size) * 0.5:
      raise ValueError("box size is too small!")
  else:
      if alpha <= 10.0:
          capacity = 1.15
      else:
          capacity = 1.25
  neighformat = partition.OrderedSparse
  neighbor_list_fn = partition.neighbor_list(displacement, box_size,
                           r_cutoff=rcutoff_neigh, dr_threshold=0.5, capacity_multiplier=capacity,
                           format=neighformat)
  R_tmp = random.uniform(random.PRNGKey(0), (N, dimension), minval=0.0, maxval=box_size, dtype=f64)
  nbrs = neighbor_list_fn.allocate(R_tmp)

  def solver_nl(R, params):
   new_energy_fn_nl = energy_hm_fn_nl(params)
   R_final, maxgrad, nbrs_final, niter = run_minimization_while_nl_overflow_fn(
                                                             new_energy_fn_nl,
                                                             nbrs,
                                                             R, shift,
                                                             min_style=2,
                                                             dt_start=dt_start,
                                                             dt_max=dt_max,
                                                             max_grad_thresh=Ftol)
   return R_final, (nbrs_final, maxgrad)

  def optimality_fn_nl(R, p):
    o_energy_fn_nl = energy_hm_fn_nl(p)
    o_nbrs = nbrs.update(R)
    return quantity.force(o_energy_fn_nl)(R, neighbor=o_nbrs)
  decorated_solver_nl = custom_root(optimality_fn_nl, has_aux=True)(solver_nl)

  def run_nl_imp(params_dict, R_init):
   R_final, minimize_info = decorated_solver_nl(R_init, params_dict)
   con_min = jnp.where(minimize_info[1] <= Ftol, True, False)

   f_energy_fn_nl = energy_hm_fn_nl(params_dict)
#   f_nbrs = nbrs.update(R_final)
   nbrs = minimize_info[0]
   emt_fn = elasticity.athermal_moduli(f_energy_fn_nl, check_convergence=True)

   def true_fn():
       C, converg = emt_fn(R_final, box_size, neighbor=nbrs)
       nu = elasticity.extract_isotropic_moduli(C)['nu']
       return nu

   def false_fn():
       return 0.0

   nu = lax.cond(con_min, (), lambda _: true_fn(), (), lambda _: false_fn())
   return nu, R_final

  return run_nl_imp, box_size

parser = argparse.ArgumentParser()
parser.add_argument('-k', '--keyseed', action='store', type=int, help='key seed')
parser.add_argument('-np', '--ninitial', action='store', type=int, help='number of initials')
parser.add_argument('-D', '--dimension', action='store', type=int, help='dimension')
parser.add_argument('-N', '--nparticles', action='store', type=int, help='number of particles')
parser.add_argument('-a', '--alpha', action='store', type=float, help='alpha')
parser.add_argument('-r', '--density', action='store', type=float, help='number of density')
parser.add_argument('-dmin', '--diammin', action='store', type=float, help='initial minimum diameter ratio')
parser.add_argument('-B', '--Binit', action='store', type=float, help='initial binding energy')
parser.add_argument('-t0', '--dtstart', action='store', type=float, help='start timestep in FIRE')
parser.add_argument('-tmax', '--dtmax', action='store', type=float, help='maximum timestep in FIRE')
parser.add_argument('-nu', '--targetnu', action='store', type=float, help='target poisson ratio')
#parser.add_argument('-p', '--targetpress', action='store', type=float, help='target pressure')
parser.add_argument('-ns', '--nspecies', action='store', type=int, help='number of species')
parser.add_argument('-l', '--learnrate', action='store', type=float, help='learning rate')
parser.add_argument('-ml', '--metalearnrate', action='store', type=float, help='meta learning rate')
parser.add_argument('-ltol', '--losstol', action='store', type=float, help='tolerant of loss function')
parser.add_argument('-s', '--optsteps', action='store', type=int, help='optimization steps')

args = parser.parse_args()

key_seed = args.keyseed
np = args.ninitial
dimension = args.dimension
N = args.nparticles
alpha_init = args.alpha
density = args.density
dmin = args.diammin
B_init = args.Binit
dt_start = args.dtstart
dt_max = args.dtmax
nu_target = args.targetnu
#press_target = args.targetpress
n_species = args.nspecies
start_learning_rate = args.learnrate
start_meta_learning_rate = args.metalearnrate
ltol = args.losstol
opt_steps = args.optsteps

key = random.PRNGKey(key_seed)

dimension = dimension
num_B = int((n_species + 1) * n_species * 0.5)

# path=("/Users/mzu/Documents/Work/self-assembly of disorder/jax_md_mp/data/Data_bea81/optimization/individual/"
#       "alpha5.0_density1.6/loss_hm_Rf_meta_individual1"
#       +"_a"+str(alpha_init)+"_density"+str(density)+"_dmin"+str(dmin)+"_N"+str(N)+"_nsp"+str(n_species)+
#       "_nu"+str(nu_target)+"_ltol"+str(ltol)+"_np"+str(np)
#       )
path=("/Users/mzu/PycharmProjects/jaxmetal/data/disordered_solids/hmmorse_a5.0/"
      "loss_hm_Rf_meta_individual1_a"+str(alpha_init)
      +"_density"+str(density)+"_dmin"+str(dmin)+"_N"+str(N)+"_nsp"+str(n_species)+
      "_nu"+str(nu_target)+"_ltol"+str(ltol)+"_np"+str(np)
      )

inputfile=str(path)+"_lrs.dat"
if os.path.isfile(inputfile):
    with open(inputfile, 'r') as f:
        for line in f:
            pass
        last_line = line
        sp = last_line.split("\t")
        step_start = sp[0]
        start_learning_rate=f64(sp[1])
        loss_tmp = f64(sp[2])
#       if loss_tmp < ltol:
#           exit()
### read params in need            
        diameters_seed = jnp.array([sp[s] for s in range(4, int(4+n_species))], dtype=f64)
        B_seed = jnp.array([sp[s] for s in range(int(4+n_species), int(4+n_species+num_B))], dtype=f64)
#        alpha_seed = jnp.array([sp[s] for s in range(4, int(4+num_B))], dtype=f64)
else:
    step_start = 0
    D_seed = jnp.array([dmin, 1.0], dtype=f64)
    diameters_seed = jnp.repeat(D_seed, int(n_species/2))
    B_seed = jnp.ones(num_B, dtype=f64) * B_init
#    alpha_seed = jnp.ones(num_B, dtype=f64) * alpha_init
param_dict = {"diameters_seed":diameters_seed, "B_seed":B_seed}
#param_dict = {"alpha_seed":alpha_seed}

boxfile=str(path)+"_lrs.box"
if os.path.isfile(boxfile):
    with open(boxfile, 'r') as f:
        box_size = f.readline()
        box_size = f64(box_size)
else:
    box_size = None

#### alpha in param_dict
#   D_seed = jnp.array([dmin, 1.0], dtype=f64)
#   diameters_seed = jnp.repeat(D_seed, int(n_species/2))
#   sigma = diameters_to_sigma_matrix(diameters_seed)
#   run, box_size = setup(param_dict, 
#           N, dimension, 
#           box_size=box_size, density=density, 
#           B=B_init,
#           sigma=sigma,
#           nspecies=n_species) 

run, box_size = setup(param_dict, 
        N, dimension, 
        box_size=box_size, density=density, 
        alpha=alpha_init,
        nspecies=n_species) 

if os.path.isfile(boxfile):
    pass
else:
    with open(boxfile, 'w') as f:
        print('{:.60g}'.format(box_size), file=f)

confile=str(path)+"_lrs.con"
if os.path.isfile(confile):
    Rinit = []
    with open(confile, 'r') as f:
        for i, line in enumerate(f):
            sp = line.split("\t")
            if i > 2:
                # Rinit += [[sp[i] for i in range(dimension, int(2*dimension))]]
                Rinit += [[sp[i] for i in range(0, dimension)]]
            else:
                pass
    Rinit = jnp.array(Rinit, dtype=f64)
    print('read conf0')
else:
    nu_tmp = 0.0
    while (nu_tmp == 0.0):
        key, split = random.split(key)
        Rinit = random.uniform(split, (N, dimension), minval=0.0, maxval=box_size, dtype=f64)
        nu_tmp = run(param_dict, Rinit)[0]

nothreshold = False
if nothreshold:
    ltol = 1e-100
    confile=str(path)+"_lrs_break.con"

logfile=str(path)+"_lrs.log"
params, Rfinal = optimize_Rf_meta_fn(run, Rinit, param_dict, inputfile, logfile, 
        start_learning_rate=start_learning_rate,
        start_meta_learning_rate=start_meta_learning_rate,
        target=nu_target,
        ltol=ltol,
        opt_steps=opt_steps, step_start=int(step_start),
        conout=False, path=path,
        nothreshold=nothreshold)

with open(confile, 'w') as f:
    print('{:6d}\t{:.60g}'.format(N, box_size), file=f)
    print('\t'.join("%.60g" % d for d in params["diameters_seed"]), file=f)
    print('\t'.join("%.60g" % b for b in params["B_seed"]), file=f)
#    print('\t'.join("%.16f" % a for a in params["alpha_seed"]), file=f)
    if dimension == 2:
        for i, r in enumerate(Rfinal):
            print('{:.16f}\t{:.16f}'.format(r[0], r[1]), file=f)
    elif dimension == 3:
        for i, r in enumerate(Rinit):
            print('{:.16f}\t{:.16f}\t{:.16f}\t{:.16f}\t{:.16f}\t{:.16f}'.format(r[0], r[1], r[2], 
                Rfinal[i,0], Rfinal[i,1], Rfinal[i,2]), file=f)
