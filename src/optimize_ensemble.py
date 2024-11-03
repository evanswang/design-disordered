
import jax.numpy as jnp
# from jax.config import config
#
# config.update('jax_enable_x64', True)  # use double-precision numbers

from jax import random
from jax import jit, lax

from jax_md import space, util, quantity, elasticity, partition, smap

from jaxopt.implicit_diff import custom_root

from minimization import run_minimization_while_nl_overflow_fn

from optimizer_minibatch import optimize_batch_avegrad

from utils import vector2dsymmat, diameters_to_sigma_matrix, get_psi_k_function
from utils import load_yaml, merge_dicts

import energy

import os
os.environ["XLA_FLAGS"]='--xla_force_host_platform_device_count=8'

f32 = jnp.float32
f64 = jnp.float64

Array = util.Array
maybe_downcast = util.maybe_downcast
DisplacementOrMetricFn = space.DisplacementOrMetricFn

@jit
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

original_yaml = 'original_ens.yaml'
custom_yaml = 'custom_ens.yaml'
original_params = load_yaml(original_yaml)
custom_params = load_yaml(custom_yaml)

params = merge_dicts(original_params, custom_params)

input_path = params['path']['input_path']
output_path = params['path']['output_path']
config_path = params['path']['config_path']

prop = params['property']['name']
qorder = params['property']['q']
target = params['property']['target']

key_seed = params['model']['key']
N = params['model']['N']
nspecies = params['model']['nspecies']
dimension = params['model']['dimension']
density = params['model']['density']
box_size = params['model']['box_size']
alpha = params['model']['alpha']
k = params['model']['k']
D_seed = params['model']['sigma']
B_seed = params['model']['B']

dr_threshold = params['neighbor']['dr_threshold']
capacity = params['neighbor']['capacity']

dt_start = params['minimize']['dt_start']
dt_max = params['minimize']['dt_max']
Ftol = params['minimize']['Ftol']
num_steps = params['minimize']['num_steps']
min_style = params['minimize']['min_style']

optimizer = params['optimize']['optimizer']
opt_steps = params['optimize']['opt_steps']
lr_type = params['optimize']['lr_type']
start_learning_rate = params['optimize']['lr']
afactor = params['optimize']['afactor']
ltol = params['optimize']['ltol']
ndev = params['optimize']['ndev']
mini_batch = params['optimize']['mini_batch']

design_params = params['design']
for key, value in design_params.items():
    design_params[key] = jnp.array(value, dtype=f64)

hmmorse_params = {
    'B_seed': B_seed,
    'alpha': alpha,
    'k': k,
    'diameters_seed': D_seed
     }

N_s = int(N // nspecies)
Ns = N_s * jnp.ones(nspecies, dtype=int)
if box_size == None:
    box_size = quantity.box_size_at_number_density(N, density, dimension)
displacement, shift = space.periodic(box_size)

species_seed = jnp.arange(nspecies)
species_vec = jnp.repeat(species_seed, N_s)

def energy_hm_fn_nl(p):
    energy_params = merge_dicts(hmmorse_params, p)

    sigma = diameters_to_sigma_matrix(energy_params['diameters_seed'])
    B = vector2dsymmat(energy_params['B_seed'])
    alpha = energy_params['alpha']
    k = energy_params['k']

    energy_fn = smap.pair_neighbor_list(
            energy.harmonic_morse_cutoff,
            space.canonicalize_displacement_or_metric(displacement),
            species=species_vec,
            epsilon=B,
            alpha=alpha,
            sigma=sigma,
            k=k)
    return energy_fn

###### initialize neighbor list######
energy_params = merge_dicts(hmmorse_params, design_params)
if alpha < 30.0:
    rcutoff_neigh = (9.9 / alpha + max(energy_params["diameters_seed"])) * 1.0
    capacity = 1.15
else:
    rcutoff_neigh = (9.9 / alpha + max(energy_params["diameters_seed"])) * 2.0
    capacity = 1.25

if rcutoff_neigh > jnp.min(box_size) * 0.5:
      raise ValueError("box size is too small!")

neighformat = partition.OrderedSparse
neighbor_list_fn = partition.neighbor_list(displacement, box_size,
                           r_cutoff=rcutoff_neigh, dr_threshold=dr_threshold, capacity_multiplier=capacity,
                           format=neighformat)
R_tmp = random.uniform(random.PRNGKey(0), (N, dimension), minval=0.0, maxval=box_size, dtype=f64)
nbrs = neighbor_list_fn.allocate(R_tmp)

def solver_nl(R, p):
    new_energy_fn_nl = energy_hm_fn_nl(p)
    R_final, maxgrad, nbrs_final, niter = run_minimization_while_nl_overflow_fn(
                                                             new_energy_fn_nl,
                                                             nbrs,
                                                             R, shift,
                                                             min_style=min_style,
                                                             dt_start=dt_start,
                                                             dt_max=dt_max,
                                                             max_grad_thresh=Ftol)
    return R_final, (nbrs_final, maxgrad)

def optimality_fn_nl(R, p):
    o_energy_fn_nl = energy_hm_fn_nl(p)
    o_nbrs = nbrs.update(R)
    return quantity.force(o_energy_fn_nl)(R, neighbor=o_nbrs)
decorated_solver_nl = custom_root(optimality_fn_nl, has_aux=True)(solver_nl)

if prop == 'psi':
    displacement_all = space.map_product(displacement)
    psi_fn = get_psi_k_function(displacement_all, qorder)

def measure_nu_nl_fn(energy_fn_nl, nbrs, R):
    emt_fn = elasticity.athermal_moduli(energy_fn_nl, check_convergence=True)
    C, converg = emt_fn(R, box_size, neighbor=nbrs)
    nu = elasticity.extract_isotropic_moduli(C)['nu']
    return nu

def measure_pressure_nl_fn(energy_fn_nl, nbrs, R):
    press = quantity.pressure(energy_fn_nl, R, box_size, neighbor=nbrs)
    return press

def measure_psi_nl_fn(p, R):
    dr_cut = jnp.amax(p["diameters_seed"])
    psi = psi_fn(R, dr_cut)
    return psi

if prop == 'nu':
    measure_fn = measure_nu_nl_fn
elif prop == 'press':
    measure_fn = measure_pressure_nl_fn
elif prop == 'psi':
    measure_fn = measure_psi_nl_fn

# define implicit differentiation function
def run_nl_imp(design_params, R_init):
    R_final, minimize_info = decorated_solver_nl(R_init, design_params)
    con_min = jnp.where(minimize_info[1] <= Ftol, True, False)

    f_energy_fn_nl = energy_hm_fn_nl(design_params)
    f_nbrs = nbrs.update(R_final)

    measure = measure_fn(f_energy_fn_nl, f_nbrs, R_final)

    measure = lax.cond(con_min, (), lambda _: measure, (), lambda _: 0.0)

    return measure, con_min

key = random.PRNGKey(key_seed)
key, simkey = random.split(key, num=2)

num_B = int((nspecies + 1) * nspecies * 0.5)

inputfile = str(input_path)
if os.path.isfile(inputfile):
    with open(inputfile, 'r') as f:
        for line in f:
            pass
        last_line = line
        sp = last_line.split("\t")
        step_start = sp[0]
        start_learning_rate=f64(sp[1])
        loss_tmp = f64(sp[2])
        if loss_tmp < ltol:
            exit()
        diameters_seed = jnp.array([sp[s] for s in range(5, int(5+nspecies))], dtype=f64)
        B_seed = jnp.array([sp[s] for s in range(int(5+nspecies), int(5+nspecies+num_B))], dtype=f64)
        param_dict = {"diameters_seed": diameters_seed, "B_seed": B_seed}
else:
    step_start = 0
    param_dict = design_params

params = optimize_batch_avegrad(
    run_nl_imp,
    simkey,
    box_size,
    N,
    dimension,
    param_dict,
    inputfile,
    learning_rate_const=True,
    start_learning_rate=start_learning_rate,
    afactor=afactor,
    target=target,
    mini_batch=mini_batch,
    opt_steps=opt_steps,
    step_start=int(step_start),
    ndev=ndev)
