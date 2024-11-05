from dataclasses import dataclass

import jax.numpy as jnp

from utils import load_yaml, merge_dicts


@dataclass
class Settings:
    f32 = jnp.float32
    f64 = jnp.float64
    Array = jnp.ndarray

    # Load original and custom YAML files
    original_yaml = "config/original.yaml"
    custom_yaml = "config/custom.yaml"
    original_params = load_yaml(original_yaml)
    custom_params = load_yaml(custom_yaml)

    params = merge_dicts(original_params, custom_params)
    particle_count = params["model"]["N"]
    nspecies = params["model"]["nspecies"]
    dimension = params["model"]["dimension"]
    box_size = params["model"]["box_size"]
    alpha = params["model"]["alpha"]
    k = params["model"]["k"]
    D_seed = params["model"]["sigma"]
    B_seed = params["model"]["B"]
    dr_threshold = params["neighbor"]["dr_threshold"]
    capacity = params["neighbor"]["capacity"]
    start_time_step = params["minimize"]["dt_start"]
    dt_max = params["minimize"]["dt_max"]
    f_tol = params["minimize"]["Ftol"]
    num_steps = params["minimize"]["num_steps"]
    min_style = params["minimize"]["min_style"]
    optimizer = params["optimize"]["optimizer"]
    opt_steps = params["optimize"]["opt_steps"]
    lr_function = params["optimize"]["lr_type"]
    density = params["model"]["density"]
    start_learning_rate = params["optimize"]["lr"]
    meta_learning_rate = params["optimize"]["metalr"]
    prop = params["property"]["name"]
    ltol = params["optimize"]["ltol"]
    design_params = params["design"]
    for key, value in design_params.items():
        design_params[key] = jnp.array(value, dtype=f64)
    hmmorse_params = {"B_seed": B_seed, "alpha": alpha, "k": k, "diameters_seed": D_seed}
    energy_params = merge_dicts(hmmorse_params, design_params)


settings = Settings()
