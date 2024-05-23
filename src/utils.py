#@title Imports and utility code
import yaml

import jax.numpy as jnp
from jax.config import config
config.update('jax_enable_x64', True)

from jax import jit, vmap

from jax_md import space

import chex
Array = chex.Array
from typing import Union

Array = chex.Array
Numeric = Union[Array, int, float]

f32 = jnp.float32
f64 = jnp.float64

def load_yaml(file_path):
  with open(file_path, 'r') as file:
    return yaml.safe_load(file)


def merge_dicts(original, new):
  for key, value in new.items():
    if isinstance(value, dict) and key in original:
      original[key] = merge_dicts(original[key], value)
    else:
      original[key] = value
  return original


def box_at_packing_fraction(sigmas, Ns, phi, dimension):
  '''
  :param sigmas: Array, (nsp,), diameters for nsp types of particles
  :param Ns: Int array, (nsp,), number of particles for each type of species
  :param phi: float, volume fraction
  :param dimension
  :return: float, box size
  '''
  sphere_volume_2d = lambda s, n: (jnp.pi / f32(4)) * n * s ** 2
  sphere_volume_3d = lambda s, n: (jnp.pi / f32(6)) * n * s ** 3
  if dimension == 2:
    sphere_volume = sphere_volume_2d
  elif dimension == 3:
    sphere_volume = sphere_volume_3d

  sphere_volume_total = jnp.sum(vmap(sphere_volume, in_axes=(0, 0))(jnp.array(sigmas), jnp.array(Ns)))
  return (sphere_volume_total / phi) ** (1 / dimension)


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

def alphas_to_alpha_matrix(alphas):
  return vmap(vmap(lambda a1,a2: jnp.sqrt(a1 * a2), in_axes=(None, 0)), in_axes=(0, None))(alphas, alphas)

def B_full(B_vec):
  return vmap(vmap(lambda B1,B2: jnp.sqrt(B1 * B2), in_axes=(None, 0)), in_axes=(0, None))(B_vec, B_vec)


'''
Ref: https://arxiv.org/pdf/1202.5281.pdf
'''
def _make_nearest_neigh_SANN(drlen):
  drlen_sorted = jnp.sort(drlen)
  sort_index = jnp.argsort(drlen)
  dr_include = jnp.zeros(len(drlen))
  m = 3
  Rm = sum(drlen_sorted[1:4]) / (m-2)
  dr_include = dr_include.at[sort_index[1]].set(1)
  dr_include = dr_include.at[sort_index[2]].set(1)
  dr_include = dr_include.at[sort_index[3]].set(1)
  drlen_length = len(drlen)

  while m<len(drlen) and Rm>drlen_sorted[m+1]:
      m += 1
      m1 = m + 1
      Rm = jnp.sum(drlen_sorted[1:m1]) / (m-2)
      dr_include = dr_include.at[sort_index[m]].set(1)
  return dr_include

def make_nearest_neigh_SANN(dr):
  dr_include = jnp.array([_make_nearest_neigh_SANN(dr[i]) for i in range(dr.shape[0])])
  return dr_include

def get_psi_k_function(displacement_all, k):

  i_imaginary = complex(0, 1)
  def order(dR_ij):
    epsilon = 1e-4
    theta_ij = jnp.arctan2(dR_ij[1] + epsilon, dR_ij[0] + epsilon)
    return jnp.exp(i_imaginary*k*theta_ij)

  order_all = vmap(vmap(order))

  def psi_k_i(R, dr_cut):
    dr = displacement_all(R, R)
    dRlen = space.distance(dr)
#    dr_include = make_nearest_neigh_SANN(dRlen)
    dr_include = jnp.array(jnp.where(dRlen<dr_cut, 1, 0)) - jnp.eye(R.shape[0], dtype=jnp.int32)
    o = order_all(dr)
    wo = jnp.multiply(dr_include, o)
    return jnp.nan_to_num(jnp.sum(wo, axis=0)/jnp.sum(dr_include, axis=0))

  def psi_k(R, rcut):
    return jnp.mean(jnp.abs(psi_k_i(R, rcut)))

  return psi_k

