import tensorflow as tf


def simps(f, a, b, N=128):
  """Approximate the integral of f(x) from a to b by Simpson's rule.
    Simpson's rule approximates the integral \int_a^b f(x) dx by the sum:
    (dx/3) \sum_{k=1}^{N/2} (f(x_{2i-2} + 4f(x_{2i-1}) + f(x_{2i}))
    where x_i = a + i*dx and dx = (b - a)/N.
    Parameters
    ----------
    f : function
        Vectorized function of a single variable
    a , b : numbers
        Interval of integration [a,b]
    N : (even) integer
        Number of subintervals of [a,b]
    Returns
    -------
    float
        Approximation of the integral of f(x) from a to b using
        Simpson's rule with N subintervals of equal length.
    Examples
    --------
    >>> simps(lambda x : 3*x**2,0,1,10)
    1.0
    Notes:
    ------
    Stolen from: https://www.math.ubc.ca/~pwalls/math-python/integration/simpsons-rule/
    """
  if N % 2 == 1:
    raise ValueError("N must be an even integer.")
  dx = (b - a) / N
  x = tf.linspace(a, b, N + 1)
  y = f(x)
  S = dx / 3 * tf.reduce_sum(y[0:-1:2] + 4 * y[1::2] + y[2::2], axis=0)
  return S


def trapz(y, x):
  """ Unequal space trapezoidal rule.
    Approximate the integral of y with respect to x based on the trapezoidal rule.
    x and y must be to the same length. 
    Trapezoidal rule's rule approximates the integral \int_a^b f(x) dx by the sum:
    (\sum_{k=1}^{N} (x_{i-1}-x_{i}))(f(x_{i-1}) + f(x_{i}))/2
    Parameters
    ----------
    y : array_like or tf.TensorArray
        vector of dependent variables

    x : array_like or tf.TensorArray
        vector of independent variables
    Returns
    -------
    float or array_like or tf.TensorArray
        Approximation of the integral of y with respect to x using
        trapezoidal's rule with subintervals of unequal length.
    """
  T = tf.reduce_sum(
      (tf.reshape(x[1:] - x[:-1], [-1, 1, 1])) * (y[1:] + y[:-1]) / 2, axis=0)
  return T
