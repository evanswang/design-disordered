import jax.numpy as jnp

from jax import jit

# from jax.config import config
# config.update('jax_enable_x64', True)

f32 = jnp.float32
f64 = jnp.float64

@jit
def harmonic_morse_cutoff(dr,
        epsilon=0.1,
        alpha=50.0,
        sigma=1.0,
        delta=2.0,
        k=50.0, **kwargs):
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

  return jnp.nan_to_num(jnp.array(U, dtype=dr.dtype))