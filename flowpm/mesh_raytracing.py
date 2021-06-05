
import mesh_tensorflow as mtf
import tensorflow.compat.v1 as tf

import flowpm
import flowpm.mesh_utils as mesh_utils
import flowpm.mtfpm as mpm

def density_plane(state,
                  nc,
                  plane_resolution,
                  halo_size,
                  lp_shape,
                  rotation=None,
                  shift=None,
                  name='density_plane'):
  """ Extract a slice from the input state vector and
  project it as a density plane.

  Args:
    state: `Tensor`, input state tensor.
    nc: int or list of int, size of simulation volume
    center: float, center of the slice along the z axis in voxel coordinates
    width: float, width of the slice in voxel coordinates
    plane_size: int, number of pixels of the density plane. 
    rotation: 3x3 float tensor, 3D rotation matrix to apply to the cube before cutting the plane
    shift: float tensor of shape [3], 3D shift to apply to the cube before cutting the plane
    name: `string`, name of the operation.

  Returns:
    `Tensor` of shape [batch_size, plane_size, plane_size], of projected density plane.
  """
  pos = state[0]
  mesh = pos.mesh
  # Slicing and rescaling to target grid
  xy = mtf.slice(pos / nc * plane_resolution, 0, 2, "ndim")

  # TODO: enable slicing of sub slice of the cube
  # Selecting only particles that fall inside the volume of interest   
  #   d = mtf.slice(pos, 2, 1, "ndim")
  #   mask = (d > (center - width / 2)) & (d <= (center + width / 2))

  # Painting density plane
  density_plane = mtf.zeros(mesh, shape=lp_shape)
  for block_size_dim in lp_shape[-2:]:
    density_plane = mtf.pad(density_plane, [halo_size, halo_size],
                          block_size_dim.name)
  density_plane = mesh_utils.cic_paint2d(density_plane, 
                                         xy, halo_size)
  # Halo exchange
  for blocks_dim, block_size_dim in zip(lp_shape[1:3], density_plane.shape[-2:]):
    density_plane = mpm.halo_reduce(density_plane, blocks_dim, block_size_dim,
                                       halo_size)
  # Remove borders
  for block_size_dim in lp_shape[-2:]:
    density_plane = mtf.slice(density_plane, halo_size, block_size_dim.size,
                                  block_size_dim.name)

  # Apply density normalization
  density_plane = density_plane / ((nc / plane_resolution) *
                                   (nc / plane_resolution) * (nc))

  return density_plane

