"""TensorFlow implementation of the interpolation function"""

import tensorflow as tf

def interp_tf(x, xp, fp):
    """Returns the one-dimensional piecewise linear interpolant to a function 
        with given discrete data points (xp, fp), evaluated at x.
  
      Parameters
      ----------
      x: array_like or tf.TensorArray
      The x-coordinates at which to evaluate the interpolated values.
        
      xp: array_like or tf.TensorArray
      The x-coordinates of the data points.

      fp: array_like or tf.TensorArray
      The y-coordinates of the data points, same length as xp.
      
      Returns
      -------
      Scalar float Tensor
      The interpolated values, same shape as x.
  
    """
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    xp = tf.convert_to_tensor(xp, dtype=tf.float32)
    fp = tf.convert_to_tensor(fp, dtype=tf.float32)
    # First we find the nearest neighbour
    ind = tf.math.argmin((tf.expand_dims(x,1) - 
                          tf.expand_dims(xp,0)) ** 2, axis=-1)
    # Perform linear interpolation
    ind = tf.clip_by_value(ind, 1, len(xp) - 2)
    xi = tf.gather(xp, ind)
    # Figure out if we are on the right or the left of nearest
    s = tf.cast(tf.math.sign(tf.clip_by_value(x, xp[-2], xp[1]) - xi),dtype=tf.int64)
    fp0= tf.gather(fp,ind)
    fp1= tf.gather(fp, ind + tf.cast(tf.sign(s),dtype=tf.int64)) - fp0
    xp0= tf.gather(xp, ind)
    xp1= tf.gather(xp, ind + tf.cast(tf.sign(s),dtype=tf.int64)) - xp0
    a = (fp1)/(xp1)
    b = fp0-a*xp0
    return a*x+b