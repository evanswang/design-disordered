#@title Imports and utility code
#!pip install https://github.com/cgoodri/jax-md/archive/elasticity.zip

import os
import numpy as onp

import jax.numpy as jnp
from jax.config import config
config.update('jax_enable_x64', True)  # use double-precision numbers

from jax import random
from jax import jit, lax, grad, vmap

import jax.scipy as jsp

from jax_md import space, energy, smap, util, elasticity, quantity

import chex
Array = chex.Array
from typing import Callable, Optional, Sequence, Union

Array = chex.Array
Numeric = Union[Array, int, float]

f32 = jnp.float32
f64 = jnp.float64

import matplotlib
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
#import seaborn as sns
#sns.set_style(style='white')

def format_plot(x, y):
  plt.grid(True)
  plt.xlabel(x, fontsize=20)
  plt.ylabel(y, fontsize=20)

def finalize_plot(shape=(1.0, 0.7)):
  plt.gcf().set_size_inches(
     shape[0] * 1.5 * plt.gcf().get_size_inches()[1],
     shape[1] * 1.5 * plt.gcf().get_size_inches()[1])
  plt.tight_layout()
  plt.show()

def draw_2dsystem(R, box_size, marker_size, color=None):
  if color == None:
     color = [64 / 256] * 3
  ms = marker_size / box_size
  R = onp.array(R)

  marker_style = dict(
    linestyle='none',
    markeredgewidth=3,
    marker='o',
    markersize=ms,
    color=color,
    fillstyle='none')
  
  plt.plot(R[:, 0], R[:, 1], **marker_style)

  plt.xlim([0, box_size])
  plt.ylim([0, box_size])
  plt.axis('off')

def draw_3dsystem(R, box_size, marker_size, color=None):
  if color == None:
     color = [64 / 256] * 3
  ms = marker_size / box_size
  R = onp.array(R)

  fig = plt.figure(figsize=(12,12))
  ax = fig.add_subplot(projection='3d')

  ax.scatter(R[:,0], R[:,1], R[:,2], )
  plt.show()

def read_xyd(fname):
  R = []
  diameters = []
  if not os.path.isfile(fname):
    raise IOError("This file '{}' does not exist.".format(fname))
  f = open(fname, "r")
  while True:
    xyd = f.readline()
    if not xyd:
      break
    x, y, d = xyd.split()
    R.append([float(x), float(y)])
    diameters.extend([float(d)])
  return jnp.array(R, dtype=f64), jnp.array(diameters, dtype=f64)


def cal_packing_fraction(sigmas, N, nspecies, box_size, dimension):
  N_s = int(N // nspecies)
  Ns = N_s * jnp.ones(nspecies, dtype=int)
  sphere_volume_2d = lambda s, n: (jnp.pi / f32(4)) * n * s ** 2
  sphere_volume_3d = lambda s, n: (jnp.pi / f32(6)) * n * s ** 3
  if dimension == 2:
    sphere_volume = sphere_volume_2d
  elif dimension == 3:
    sphere_volume = sphere_volume_3d

  sphere_volume_total = jnp.sum(vmap(sphere_volume, in_axes=(0, 0))(jnp.array(sigmas), jnp.array(Ns)))
  return sphere_volume_total / (box_size ** dimension)
  
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

def alphas_to_alpha_matrix(alphas):
  return vmap(vmap(lambda a1,a2: jnp.sqrt(a1 * a2), in_axes=(None, 0)), in_axes=(0, None))(alphas, alphas)

def B_full(B_vec):
  return vmap(vmap(lambda B1,B2: jnp.sqrt(B1 * B2), in_axes=(None, 0)), in_axes=(0, None))(B_vec, B_vec)

def stress_tensor(R, displacement, potential, box, **kwargs):
  volume = box.diagonal().prod()
  dR = space.map_product(displacement)(R, R)
  dr = space.distance(dR)

#  kwargs = smap._kwargs_to_parameters(None, **kwargs)
  dUdr = vmap(vmap(grad(potential)))(dr, **kwargs)
  temp = vmap(vmap(lambda s, m: s*m))(smap._diagonal_mask(dUdr/dr), 
              jnp.einsum('abi,abj->abij', dR, dR))
  return util.high_precision_sum(temp, axis=(0, 1)) * f32(0.5) / volume

## calculate the excess coordinate number \delt Z = Z - Z_iso ##
## where Z_iso = 2d - 2d/N is the isotropic coordinate number ##
def get_coordinate_number(displacement_or_metric, R, sigma, species=None):
  r_cutoff = 1.0
  if (species is not None):
   # convert jax.array into numpy.array to speed up
   sigma_onp = onp.array(sigma) * r_cutoff
   tmp_onp_array = [[sigma_onp[i, j] for i in species] for j in species]
   dr_cutoff = jnp.array(tmp_onp_array)
  else:
   dr_cutoff = sigma * r_cutoff

  metric = space.map_product(space.canonicalize_displacement_or_metric(displacement_or_metric))
  dr = metric(R, R)

  coordinate_metric = jnp.where(dr<=dr_cutoff, 1, 0) - jnp.eye(R.shape[0], dtype=jnp.int32)
  coordinate_number = jnp.sum(coordinate_metric, axis=1).tolist()
  coordinate_list = []
  for i in range(R.shape[0]):
    coordinate_list_i = []
    for j in range(R.shape[0]):
      if coordinate_metric[i,j] > 0:
        coordinate_list_i.append(j)
    coordinate_list.append(coordinate_list_i)

  ave_coordinates = 0.0
  non_rattler = 0
  for i in range(len(coordinate_number)):
    if coordinate_number[i] >= 3:
      ave_coordinates += coordinate_number[i]
      non_rattler += 1
  if non_rattler > 0:
    ave_coordinates = ave_coordinates / float(non_rattler)
  else:
    ave_coordinates = 0.0

  return coordinate_list, ave_coordinates, non_rattler

# Turn regular packings into spring networks
def calculate_bond_data(displacement_or_metric, R, dr_cutoff, species=None):
  if (species is not None):
#    dr_cutoff1 = onp.array([[dr_cutoff[i,j] for i in species] for j in species])
    N_s = int(R.shape[0] / len(onp.unique(species).tolist()))
    dr_cutoff = jnp.repeat(jnp.repeat(dr_cutoff, N_s, axis=0), N_s, axis=1)

  metric = space.map_product(space.canonicalize_displacement_or_metric(displacement_or_metric))
  dr = metric(R, R)

  N = R.shape[0]
  dr_include = jnp.triu(jnp.where(dr<dr_cutoff, 1, 0)) - jnp.eye(R.shape[0],dtype=jnp.int32)
  index_list = jnp.dstack(onp.meshgrid(jnp.arange(N), jnp.arange(N), indexing='ij'))

  i_s = jnp.where(dr_include==1, index_list[:,:,0], -1).flatten()
  j_s = jnp.where(dr_include==1, index_list[:,:,1], -1).flatten()
  ij_s = jnp.transpose(jnp.array([i_s,j_s]))

  bonds = ij_s[(ij_s!=jnp.array([-1,-1]))[:,1]]
  lengths = dr.flatten()[(ij_s!=jnp.array([-1,-1]))[:,1]]

  return bonds, lengths
  
def _remove_rattlers_oneshot(R, bonds, node_arrays, bond_arrays):
  N, dimension = R.shape
  #Z_alpha = get_Z_alpha(bonds, R.shape[0])
  Z_alpha = jnp.bincount(bonds.reshape(bonds.size), length=R.shape[0])
  rattler_yesno = jnp.where(Z_alpha > dimension, False, True)
  
  if jnp.any(rattler_yesno):
    nodes_to_keep = jnp.where(rattler_yesno == False)
    node_map = jnp.full((N,), -1).at[nodes_to_keep].set(jnp.arange(nodes_to_keep[0].shape[0]))

    rattlers, = jnp.where(rattler_yesno == True)
    bonds_with_rattlers = vmap(jnp.any)(jnp.isin(bonds, rattlers)) #boolean vector of length N_bonds, True if bond contains a rattler node
    bonds_to_keep = jnp.where(bonds_with_rattlers==False)

    R_new = R[nodes_to_keep]
    if node_arrays is None:
      node_arrays_new = None
    else:
      node_arrays_new = [a[nodes_to_keep] for a in node_arrays]

    bonds_new = node_map[bonds[bonds_to_keep]]
    if bond_arrays is None:
      bond_arrays_new = None
    else:
      bond_arrays_new = [a[bonds_to_keep] for a in bond_arrays]

    return R_new, bonds_new, node_arrays_new, bond_arrays_new, True
  return R, bonds, node_arrays, bond_arrays, False

def remove_rattlers(R, bonds, node_arrays = None, bond_arrays = None):
  """ Remove rattlers from a network

  Recursively removes all nodes (and connected bonds) which do not have at least
  dimension+1 bonds. Both R and bonds are updated.

  Args:
    R:            Array of length (N, dimension) of node positions
    bonds:        Array of length (Nbonds, 2) of bond indices
    node_arrays:  List of node-based arrays. If node i is identified as a
                  rattler and removed, element i of each array is also removed.
    bond_arrays:  List of bond-based arrays. If bond i is connected to a rattler
                  and removed, element i of each array is also removed.

  Return: new versions of R, bonds, node_arrays, bond_arrays

  Note: the contents of bonds is updated to reflect the new indices of nodes
    in R. However, the contents of node_arrays and bond_arrays are not updated
    other than removing the appropriate elements. If you need to map old indices
    to new indices, this can be obtained by passing jnp.arange(N) to the
    node_arrays list. e.g.
      R_new, bonds, [index_map], _ = remove_rattlers(R, bonds, [jnp.arange(N)])
      R[(index_map,)] == R_new # All True
  """
  _R = R
  _bonds = bonds
  _node_arrays = node_arrays
  _bond_arrays = bond_arrays
  keep_trying = True

  ii = 0
  while keep_trying:
    ii += 1
    print('removing rattlers, iteration {}'.format(ii))
    _R, _bonds, _node_arrays, _bond_arrays, keep_trying = _remove_rattlers_oneshot(_R, _bonds, _node_arrays, _bond_arrays)
    if R.shape[0] < 1:
      keep_trying = False

  return _R, _bonds, _node_arrays, _bond_arrays


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

def get_lr_schedule(base_lr: float,
                    lr_decay_steps: Sequence[int],
                    lr_decay_factor: float):
  """Returns a callable that defines the learning rate for a given step."""
  if not lr_decay_steps:
    return lambda _: base_lr

  lr_decay_steps = jnp.array(lr_decay_steps)
  if not jnp.all(lr_decay_steps[1:] > lr_decay_steps[:-1]):
    raise ValueError('Expected learning rate decay steps to be increasing, got '
                     f'{lr_decay_steps}.')

  def lr_schedule(update_step: Numeric) -> Array:
    i = jnp.sum(lr_decay_steps <= update_step)
    return base_lr * lr_decay_factor**i

  return lr_schedule
