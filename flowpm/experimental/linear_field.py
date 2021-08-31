def linear_field(mesh,
                 hr_shape,
                 lr_shape,
                 boxsize,
                 nc,
                 pk,
                 kvec_lr,
                 kvec_hr,
                 halo_size,
                 post_filtering=True,
                 downsampling_factor=2,
                 antialias=True,
                 seed=None,
                 dtype=tf.float32,
                 return_random_field=False):
  """Generates a linear field with a given linear power spectrum
  """

  # Element-wise function that applies a Fourier kernel
  def _cwise_fn(kfield, pk, kx, ky, kz):
    kx = tf.reshape(kx, [-1, 1, 1])
    ky = tf.reshape(ky, [1, -1, 1])
    kz = tf.reshape(kz, [1, 1, -1])
    kk = tf.sqrt((kx / boxsize * nc)**2 + (ky / boxsize * nc)**2 +
                 (kz / boxsize * nc)**2)
    shape = kk.shape
    kk = tf.reshape(kk, [-1])
    pkmesh = tfp.math.interp_regular_1d_grid(
        x=kk,
        x_ref_min=1e-05,
        x_ref_max=1000.0,
        y_ref=pk,
        grid_regularizing_transform=tf.log)
    pkmesh = tf.reshape(pkmesh, shape)
    kfield = kfield * tf.cast((pkmesh / boxsize**3)**0.5, tf.complex64)
    return kfield

  # Generates the random field
  random_field = mtf.random_normal(
      mesh, shape=hr_shape, mean=0, stddev=nc**1.5, dtype=tf.float32)
  field = random_field
  # Apply padding and perform halo exchange with neighbors
  # TODO: Figure out how to deal with the tensor size limitations
  for block_size_dim in hr_shape[-3:]:
    field = mtf.pad(field, [halo_size, halo_size], block_size_dim.name)
  for blocks_dim, block_size_dim in zip(hr_shape[1:4], field.shape[-3:]):
    field = mesh_ops.halo_reduce(field, blocks_dim, block_size_dim, halo_size)

  # We have two strategies to separate scales, before or after filtering
  field = mtf.reshape(field, field.shape + [mtf.Dimension('h_dim', 1)])
  if post_filtering:
    high = field
    low = mesh_utils.downsample(field, downsampling_factor, antialias=antialias)
  else:
    low, high = mesh_utils.split_scales(
        field, downsampling_factor, antialias=antialias)
  low = mtf.reshape(low, low.shape[:-1])
  high = mtf.reshape(high, high.shape[:-1])

  # Remove padding and redistribute the low resolution cube accross processes
  for block_size_dim in hr_shape[-3:]:
    low = mtf.slice(low, halo_size // 2**downsampling_factor,
                    block_size_dim.size // 2**downsampling_factor,
                    block_size_dim.name)

  low_hr_shape = low.shape
  # Reshape hack
  low = mtf.slicewise(
      lambda x: x[:, 0, 0, 0], [low],
      output_dtype=tf.float32,
      output_shape=lr_shape,
      name='my_dumb_reshape',
      splittable_dims=lr_shape[:-1] + hr_shape[:4])
  #low = mtf.reshape(low, lr_shape)

  # Apply power spectrum on both grids
  klow = mesh_utils.r2c3d(low)
  khigh = mesh_utils.r2c3d(high)
  klow = mtf.cwise(_cwise_fn, [klow, pk] + kvec_lr, output_dtype=tf.complex64)
  khigh = mtf.cwise(_cwise_fn, [khigh, pk] + kvec_hr, output_dtype=tf.complex64)
  low = mesh_utils.c2r3d(klow)
  high = mesh_utils.c2r3d(khigh)

  if return_random_field:
    return low, high, random_field
  else:
    return low, high
