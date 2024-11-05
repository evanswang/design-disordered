###################################
# @name : optimization using previous minimized configuration via meta-learning
###################################
import os
import typing as t
from pathlib import Path

import jax.numpy as jnp
from jax import random
from jax import lax
from jax_md import space, quantity, elasticity, partition, smap
from jaxopt.implicit_diff import custom_root

from minimization import run_minimization_while_nl_overflow_fn
from optimizer_ind import optimize_Rf_meta_fn
from utils import vector2dsymmat, diameters_to_sigma_matrix, get_psi_k_function
from utils import merge_dicts
import energy
from config.settings import settings


def energy_hm_fn_nl(
    p: t.Dict,
) -> t.Callable:
    """
    this func is to

    """
    energy_hm_fn_nl_energy_params = merge_dicts(settings.hmmorse_params, p)

    diameters_seed = jnp.array(
        energy_hm_fn_nl_energy_params["diameters_seed"], dtype=settings.f64
    )
    b_seed = jnp.array(energy_hm_fn_nl_energy_params["B_seed"], dtype=settings.f64)

    sigma = diameters_to_sigma_matrix(diameters_seed)
    b_matrix = vector2dsymmat(b_seed)
    alpha = energy_hm_fn_nl_energy_params["alpha"]
    k = energy_hm_fn_nl_energy_params["k"]

    energy_fn = smap.pair_neighbor_list(
        energy.harmonic_morse_cutoff,
        space.canonicalize_displacement_or_metric(displacement),
        species=species_vec,
        epsilon=b_matrix,
        alpha=alpha,
        sigma=sigma,
        k=k,
    )
    return energy_fn


def solver_nl(R, p):
    new_energy_fn_nl = energy_hm_fn_nl(p)
    R_final, maxgrad, nbrs_final, niter = run_minimization_while_nl_overflow_fn(
        new_energy_fn_nl,
        nbrs,
        R,
        shift,
        min_style=settings.min_style,
        dt_start=settings.start_time_step,
        dt_max=settings.dt_max,
        max_grad_thresh=settings.f_tol,
    )
    return R_final, (nbrs_final, maxgrad)


def optimality_fn_nl(R, p):
    o_energy_fn_nl = energy_hm_fn_nl(p)
    o_nbrs = nbrs.update(R)
    return quantity.force(o_energy_fn_nl)(R, neighbor=o_nbrs)


def measure_nu_nl_fn(energy_fn_nl, nbrs, R):
    emt_fn = elasticity.athermal_moduli(energy_fn_nl, check_convergence=True)
    C, converg = emt_fn(R, box_size, neighbor=nbrs)
    nu = elasticity.extract_isotropic_moduli(C)["nu"]
    return nu


def measure_pressure_nl_fn(energy_fn_nl, nbrs, R):
    press = quantity.pressure(energy_fn_nl, R, box_size, neighbor=nbrs)
    return press


def measure_psi_nl_fn(p, R):
    dr_cut = jnp.amax(p["diameters_seed"])
    psi = psi_fn(R, dr_cut)
    return psi


# define implicit differentiation function
def run_nl_imp(design_params, R_init):
    R_final, minimize_info = decorated_solver_nl(R_init, design_params)
    con_min = jnp.where(minimize_info[1] <= settings.f_tol, True, False)
    f_energy_fn_nl = energy_hm_fn_nl(design_params)
    f_nbrs = nbrs.update(R_final)
    measure = measure_fn(f_energy_fn_nl, f_nbrs, R_final)
    measure = lax.cond(con_min, (), lambda _: measure, (), lambda _: 0.0)
    return measure, R_final


def get_parameters(inputfile: str) -> (dict, int, float):
    start_learning_rate = -1.0
    if os.path.isfile(inputfile):
        with open(inputfile, "r") as f:
            for line in f:
                pass
            last_line = line
            sp = last_line.split("\t")
            step_start = sp[0]
            start_learning_rate = settings.f64(sp[1])
            loss_tmp = settings.f64(sp[2])
            if loss_tmp < settings.ltol:
                exit()
            diameters_seed = jnp.array(
                [sp[s] for s in range(4, int(4 + settings.nspecies))],
                dtype=settings.f64,
            )
            B_seed = jnp.array(
                [
                    sp[s]
                    for s in range(
                        int(4 + settings.nspecies), int(4 + settings.nspecies + num_B)
                    )
                ],
                dtype=settings.f64,
            )
            param_dict = {"diameters_seed": diameters_seed, "B_seed": B_seed}
    else:
        step_start = 0
        param_dict = settings.design_params
        # diameters_seed = jnp.repeat(D_seed, int(settings.nspecies/2))
        # Bs_seed = jnp.ones(num_B, dtype=settings.f64) * B_seed
    # param_dict = {"diameters_seed":diameters_seed, "B_seed":Bs_seed}
    return param_dict, step_start, start_learning_rate


def get_right_cutoff_neigh_capacity(alpha, diameters_seed):
    ###### initialize neighbor list######
    if settings.alpha < 30.0:
        right_cutoff_neigh = (9.9 / alpha + max(diameters_seed)) * 1.0
        capacity = 1.15
    else:
        right_cutoff_neigh = (9.9 / alpha + max(diameters_seed)) * 2.0
        capacity = 1.25

    if right_cutoff_neigh > jnp.min(box_size) * 0.5:
        raise ValueError("box size is too small!")

    return right_cutoff_neigh, capacity


if __name__ == "__main__":
    box_size = settings.box_size
    if box_size is None:
        box_size = quantity.box_size_at_number_density(
            settings.particle_count, settings.density, settings.dimension
        )
    displacement, shift = space.periodic(box_size)

    species_seed = jnp.arange(settings.nspecies)
    species_vec = jnp.repeat(
        species_seed, int(settings.particle_count // settings.nspecies)
    )

    right_cutoff_neigh, capacity = get_right_cutoff_neigh_capacity(
        settings.alpha, settings.energy_params["diameters_seed"]
    )

    neighbor_list_fn = partition.neighbor_list(
        displacement,
        box_size,
        r_cutoff=right_cutoff_neigh,
        dr_threshold=settings.dr_threshold,
        capacity_multiplier=capacity,
        format=partition.OrderedSparse,
    )
    R_tmp = random.uniform(
        random.PRNGKey(0),
        (settings.particle_count, settings.dimension),
        minval=0.0,
        maxval=box_size,
        dtype=settings.f64,
    )
    nbrs = neighbor_list_fn.allocate(R_tmp)

    decorated_solver_nl = custom_root(optimality_fn_nl, has_aux=True)(solver_nl)

    prop = settings.prop
    if prop == "nu":
        measure_fn = measure_nu_nl_fn
    elif prop == "press":
        measure_fn = measure_pressure_nl_fn
    elif prop == "psi":
        measure_fn = measure_psi_nl_fn
        displacement_all = space.map_product(displacement)
        psi_fn = get_psi_k_function(displacement_all, settings.params["property"]["q"])

    key = random.PRNGKey(settings.params["model"]["key"])

    num_B = int((settings.nspecies + 1) * settings.nspecies * 0.5)

    input_file = str(settings.params["path"]["input_path"])
    param_dict, step_start, start_learning_rate = get_parameters(input_file)

    confile = settings.params["path"]["config_path"]
    if os.path.isfile(confile):
        Rinit = []
        with open(confile, "r") as f:
            for i, line in enumerate(f):
                sp = line.split("\t")
                if i > 2:
                    Rinit += [
                        [
                            sp[i]
                            for i in range(
                                settings.dimension, int(2 * settings.dimension)
                            )
                        ]
                    ]
                else:
                    pass
        Rinit = jnp.array(Rinit, dtype=settings.f64)
    else:
        nu_tmp = 0.0
        while nu_tmp == 0.0:
            key, split = random.split(key)
            Rinit = random.uniform(
                split,
                (settings.particle_count, settings.dimension),
                minval=0.0,
                maxval=box_size,
                dtype=settings.f64,
            )
            nu_tmp = run_nl_imp(param_dict, Rinit)[0]

    # call optimization
    params, Rfinal = optimize_Rf_meta_fn(
        run_nl_imp,
        Rinit,
        key,
        param_dict,
        input_file,
        start_learning_rate=settings.start_learning_rate,
        start_meta_learning_rate=settings.meta_learning_rate,
        target=settings.params["property"]["target"],
        ltol=settings.ltol,
        opt_steps=settings.opt_steps,
        step_start=int(step_start),
    )

    repo_dir = Path.cwd().parent
    confile_path = repo_dir / confile

    with open(confile_path, "w") as f:
        print("{:6d}\t{:.16f}".format(settings.particle_count, box_size), file=f)
        print("\t".join("%.16f" % d for d in params["diameters_seed"]), file=f)
        print("\t".join("%.16f" % b for b in params["B_seed"]), file=f)
        if settings.dimension == 2:
            for i, r in enumerate(Rinit):
                print(
                    "{:.16f}\t{:.16f}\t{:.16f}\t{:.16f}".format(
                        r[0], r[1], Rfinal[i, 0], Rfinal[i, 1]
                    ),
                    file=f,
                )
        elif settings.dimension == 3:
            for i, r in enumerate(Rinit):
                print(
                    "{:.16f}\t{:.16f}\t{:.16f}\t{:.16f}\t{:.16f}\t{:.16f}".format(
                        r[0], r[1], r[2], Rfinal[i, 0], Rfinal[i, 1], Rfinal[i, 2]
                    ),
                    file=f,
                )
#
# if __name__ == "__main__":
#     main()
