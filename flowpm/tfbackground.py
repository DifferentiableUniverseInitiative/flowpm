"""TensorFlow implementation of Cosmology Computations"""
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import flowpm.constants as constants
from flowpm.scipy.interpolate import interp_tf
from flowpm.cosmology import Cosmology


def fde(cosmo, a, epsilon=1e-5):
  r"""Evolution parameter for the Dark Energy density.

    Parameters
    ----------
    cosmo: Cosmology
      Cosmological parameters structure

    a : array_like or tf.TensorArray
        Scale factor

    epsilon: float value
            Small number to make sure we are not dividing by 0 and avoid a singularity

    Returns
    -------
    f : Scalar float Tensor.
        The evolution parameter of the Dark Energy density as a function
        of scale factor

    Notes
    -----

    For a given parametrisation of the Dark Energy equation of state,
    the scaling of the Dark Energy density with time can be written as:

    .. math::

        \rho_{de}(a) \propto a^{f(a)}

    (see :cite:`2005:Percival`) where :math:`f(a)` is computed as
    :math:`f(a) = \frac{-3}{\ln(a)} \int_0^{\ln(a)} [1 + w(a^\prime)]
    d \ln(a^\prime)`. In the case of Linder's parametrisation for the
    dark energy in Eq. :eq:`linderParam` :math:`f(a)` becomes:

    .. math::

        f(a) = -3(1 + w_0) + 3 w \left[ \frac{a - 1}{ \ln(a) } - 1 \right]
    """
  a = tf.convert_to_tensor(a, dtype=tf.float32)
  return (-3.0 * (1.0 + cosmo.w0) + 3.0 * cosmo.wa *
          ((a - 1.0) / tf.math.log(a - epsilon) - 1.0))


def w(cosmo, a):
  """Dark Energy equation of state parameter using the Linder
    parametrisation.

    Parameters
    ----------
    cosmo: Cosmology
      Cosmological parameters structure

    a : array_like or tf.TensorArray
        Scale factor

    Returns
    -------
    w : Scalar float Tensor
        The Dark Energy equation of state parameter at the specified
        scale factor

    Notes
    -----

    The Linder parametrization :cite:`2003:Linder` for the Dark Energy
    equation of state :math:`p = w \rho` is given by:

    .. math::

        w(a) = w_0 + w (1 -a)
  """
  a = tf.convert_to_tensor(a, dtype=tf.float32)
  return cosmo.w0 + cosmo.wa * (1.0 - a)


def E(cosmo, a):
  r"""The scale factor dependent factor E(a) in the Hubble
    parameter.

    Parameters
    ----------
    cosmo: Cosmology
      Cosmological parameters structure

    a : array_like or tf.TensorArray
        Scale factor

    Returns
    -------
    E^2 : Scalar float Tensor
        Square of the scaling of the Hubble constant as a function of
        scale factor

    Notes
    -----

    The Hubble parameter at scale factor `a` is given by
    :math:`H^2(a) = E^2(a) H_o^2` where :math:`E^2` is obtained through
    Friedman's Equation (see :cite:`2005:Percival`) :

    .. math::

        E(a) = sqrt(\Omega_m a^{-3} + \Omega_k a^{-2} + \Omega_{de} a^{f(a)})

    where :math:`f(a)` is the Dark Energy evolution parameter computed
    by :py:meth:`.f_de`.
    """
  a = tf.convert_to_tensor(a, dtype=tf.float32)
  return (tf.math.sqrt(cosmo.Omega_m / tf.math.pow(a, 3) +
                       cosmo.Omega_k / tf.math.pow(a, 2) +
                       cosmo.Omega_de * tf.math.pow(a, fde(cosmo, a))))


def H(cosmo, a):
  r"""Hubble parameter [km/s/(Mpc/h)] at scale factor `a`

    Parameters
    ----------
    cosmo: Cosmology
      Cosmological parameters structure

    a : array_like or tf.TensorArray
        Scale factor

    Returns
    -------
    H : Scalar float Tensor
        Hubble parameter at the requested scale factor.
    """
  a = tf.convert_to_tensor(a, dtype=tf.float32)
  return constants.H0 * cosmo.h * (E(cosmo, a))


def dfde(cosmo, a, epsilon=1e-5):
  r"""Derivative of the evolution parameter for the Dark Energy density
    f(a) with respect to the scale factor.

    Parameters
    ----------
    cosmo: Cosmology
      Cosmological parameters structure

    a : array_like or tf.TensorArray
        Scale factor

    epsilon: float value
            Small number to make sure we are not dividing by 0 and avoid a singularity

    Returns
    -------
    df(a)/da :  Scalar float Tensor
        Derivative of the evolution parameter for the Dark Energy density
        with respect to the scale factor.

    Notes
    -----

    The expression for :math:`\frac{df(a)}{da}` is:

    .. math::

        \frac{df}{da}(a) = =\frac{3w_a \left( \ln(a-\epsilon)-
    \frac{a-1}{a-\epsilon}\right)}{\ln^2(a-\epsilon)}
    """
  a = tf.convert_to_tensor(a, dtype=tf.float32)
  return (3 * cosmo.wa * (tf.math.log(a - epsilon) - (a - 1) / (a - epsilon)) /
          tf.math.pow(tf.math.log(a - epsilon), 2))


def dEa(cosmo, a):
  r"""Derivative of the scale factor dependent factor E(a) in the Hubble
    parameter with respect to the
    scale factor.

    Parameters
    ----------
    cosmo: Cosmology
      Cosmological parameters structure

    a : array_like or tf.TensorArray
        Scale factor

    Returns
    -------
    dE(a)/da :Scalar float Tensor
        Derivative of the scale factor dependent factor in the Hubble
      parameter with respect to the scale factor.

    Notes
    -----

    The expression for :math:`\frac{dE}{da}` is:

    .. math::

        \frac{dE(a)}{da}=\frac{-3a^{-4}\Omega_{0m}
        -2a^{-3}\Omega_{0k}
        +f'_{de}\Omega_{0de}a^{f_{de}(a)}}{2E(a)}
    """
  a = tf.convert_to_tensor(a, dtype=tf.float32)
  return 0.5 * (-3 * cosmo.Omega_m / tf.math.pow(a, 4) -
                2 * cosmo.Omega_k / tf.math.pow(a, 3) + dfde(cosmo, a) *
                cosmo.Omega_de * tf.math.pow(a, fde(cosmo, a))) / E(cosmo, a)


def Omega_m_a(cosmo, a):
  r"""Matter density at scale factor `a`.

    Parameters
    ----------
    cosmo: Cosmology
      Cosmological parameters structure

    a : array_like or tf.TensorArray
        Scale factor

    Returns
    -------
    Omega_m : Scalar float Tensor
        Non-relativistic matter density at the requested scale factor

    Notes
    -----
    The evolution of matter density :math:`\Omega_m(a)` is given by:

    .. math::

        \Omega_m(a) = \frac{\Omega_{0,m} a^{-3}}{E^2(a)}

    see :cite:`2005:Percival` Eq. (6)
    """
  a = tf.convert_to_tensor(a, dtype=tf.float32)
  return cosmo.Omega_m * tf.math.pow(a, -3) / E(cosmo, a)**2


def Omega_de_a(cosmo, a):
  r"""Dark Energy density at scale factor `a`.

    Parameters
    ----------
    cosmo: Cosmology
      Cosmological parameters structure

    a : array_like or tf.TensorArray
        Scale factor

    Returns
    -------
    Omega_de : Scalar float Tensor
        Dark Energy density at the requested scale factor

    Notes
    -----
    The evolution of Dark Energy density :math:`\Omega_{de}(a)` is given
    by:

    .. math::

        \Omega_{de}(a) = \frac{\Omega_{0,de} a^{f(a)}}{E^2(a)}

    where :math:`f(a)` is the Dark Energy evolution parameter computed by
    :py:meth:`.f_de` (see :cite:`2005:Percival` Eq. (6)).
    """
  a = tf.convert_to_tensor(a, dtype=tf.float32)
  return cosmo.Omega_de * tf.math.pow(a, fde(cosmo, a)) / E(cosmo, a)**2


def dchioverda(cosmo, a):
  r"""Derivative of the radial comoving distance with respect to the
    scale factor.

    Parameters
    ----------
    cosmo: Cosmology
      Cosmological parameters structure

    a : array_like or tf.TensorArray
        Scale factor

    Returns
    -------
    dchi/da : tf.TensorArray
        Derivative of the radial comoving distance with respect to the
        scale factor at the specified scale factor.

    Notes
    -----

    The expression for :math:`\frac{d \chi}{da}` is:

    .. math::

        \frac{d \chi}{da}(a) = \frac{R_H}{a^2 E(a)}
    """
  a = tf.convert_to_tensor(a, dtype=tf.float32)
  return constants.rh / (a**2 * E(cosmo, a))


@tf.function
def _distance_computation_func(a, rtol=1e-3, **kwcosmo):
  """ Computes integral required by radial comoving distance.

  Parameters
  ----------
  a: array_like
    Output scale factors

  rtol: float, optional
        Parameters determing the error control performed by the solver

  kwcosmo: keyword args
    Cosmological parameter values.

  Returns
  -------
  chi: array_like
    Radial comoving distance computed at desired scale factors.
  """
  a = tf.convert_to_tensor(a, dtype=tf.float32)

  @tf.function
  def dchioverdlna(x, y, **kwcosmo):
    # Instantiate a cosmology object
    cosmo = Cosmology(**kwcosmo)
    xa = tf.math.exp(x)
    return dchioverda(cosmo, xa) * xa

  solver = tfp.math.ode.BDF(rtol=rtol)

  #  # Run the ODE
  chitab = solver.solve(dchioverdlna,
                        tf.math.log(a)[0],
                        0.0,
                        tf.math.log(a),
                        constants=kwcosmo)
  chitab = chitab.states[-1] - chitab.states

  return chitab


def rad_comoving_distance(cosmo, a, log10_amin=-3, steps=256, rtol=1e-3):
  r"""Radial comoving distance in [Mpc/h] for a given scale factor.

    Parameters
    ----------
    cosmo: Cosmology
      Cosmological parameters structure

    a : array_like or tf.TensorArray
        Scale factor

    log10_amin: integer
                 Starting value of the log-scale spaced sequence.

    steps:integer, optional
        Number of samples to generate.

    rtol: float, optional
          Parameters determing the error control performed by the solver

    Returns
    -------
    chi : tf.TensorArray,
        Radial comoving distance corresponding to the specified scale
        factor.

    Notes
    -----
    The radial comoving distance is computed by performing the following
    integration:

    .. math::

        \chi(a) =  R_H \int_a^1 \frac{da^\prime}{{a^\prime}^2 E(a^\prime)}
    """
  if "background.radial_comoving_distance" not in cosmo._workspace.keys():
    atab = tf.convert_to_tensor(np.logspace(log10_amin, 0.0, steps),
                                dtype=tf.float32)

    chitab = _distance_computation_func(atab, rtol=rtol, **cosmo.to_dict())

    cache = {"a": atab[::-1], "chi": chitab[::-1]}
    cosmo._workspace["tfbackground.radial_comoving_distance"] = cache
  else:
    cache = cosmo._workspace["background.radial_comoving_distance"]
  # Return the results as an interpolation of the table)
  lna = tf.math.log(a)
  inter = tfp.math.interp_regular_1d_grid(tf.cast(lna, dtype=tf.float32),
                                          tf.math.log(cache["a"])[0],
                                          tf.math.log(cache["a"])[-1],
                                          cache["chi"])
  return tf.clip_by_value(inter, 0.0, 1000000)


def a_of_chi(cosmo, chi):
  r"""Computes the scale factor for corresponding (array) of radial comoving
    distance by reverse linear interpolation.

    Parameters:
    -----------
    cosmo: Cosmology
      Cosmological parameters

    chi: array_like or tf.TensorArray
      radial comoving distance to query.

    Returns:
    --------
    a : tf.TensorArray
      Scale factors corresponding to requested distances
    """
  # Check if distances have already been computed, force computation otherwise
  if "background.radial_comoving_distance" not in cosmo._workspace.keys():
    rad_comoving_distance(cosmo, 1.0)
  cache = cosmo._workspace["tfbackground.radial_comoving_distance"]
  chi = tf.cast(chi, dtype=tf.float32)
  return interp_tf(chi, cache["chi"], cache["a"])


def transverse_comoving_distance(cosmo, a):
  r"""Transverse comoving distance in [Mpc/h] for a given scale factor.

    Parameters
    ----------
    cosmo: Cosmology
      Cosmological parameters

    a : tf.TensorArray
        Scale factor

    Returns
    -------
    f_k : tf.TensorArray
        Transverse comoving distance corresponding to the specified
        scale factor.

    Notes
    -----
    The transverse comoving distance depends on the curvature of the
    universe and is related to the radial comoving distance through:

    .. math::

        f_k(a) = \left\lbrace
        \begin{matrix}
        R_H \frac{1}{\sqrt{\Omega_k}}\sinh(\sqrt{|\Omega_k|}\chi(a)R_H)&
            \mbox{for }\Omega_k > 0 \\
        \chi(a)&
            \mbox{for } \Omega_k = 0 \\
        R_H \frac{1}{\sqrt{\Omega_k}} \sin(\sqrt{|\Omega_k|}\chi(a)R_H)&
            \mbox{for } \Omega_k < 0
        \end{matrix}
        \right.
    """
  chi = rad_comoving_distance(cosmo, a)
  if cosmo.Omega_k < 0:  # Open universe
    return constants.rh / tf.math.sqrt(cosmo.Omega_k) * tf.math.sinh(
        cosmo.sqrtk * chi / constants.rh)
  elif cosmo.Omega_k > 0:  # Closed Universe
    return constants.rh / tf.math.sqrt(cosmo.Omega_k) * tf.math.sin(
        cosmo.sqrtk * chi / constants.rh)
  else:
    return chi


def angular_diameter_distance(cosmo, a):
  r"""Angular diameter distance in [Mpc/h] for a given scale factor.

    Parameters
    ----------
    cosmo: Cosmology
      Cosmological parameters structure

    a : tf.TensorArray
        Scale factor

    Returns
    -------
    d_A : tf.TensorArray

    Notes
    -----
    Angular diameter distance is expressed in terms of the transverse
    comoving distance as:

    .. math::

        d_A(a) = a f_k(a)
    """
  return a * transverse_comoving_distance(cosmo, a)


# Equation 1.96 from Florent Leclercq thesis
@tf.function
def growth_ode(a, y, **kwcosmo):
  """Define the ode functions that will be used to compute the linear growth factor D_1(a) and
    second-order growth factor D_2(a) at a given scale factor
    Parameters
    ----------
    a: array_like or tf.TensorArray
      Scale factor

    y: tf.TensorArray
    Contain the value of y for each desired scale factors in a, with the initial value y0 in the first row

    cosmo: Cosmology
      Cosmological parameters structure

    Notes
    -----
    Linear growth factor D_1(a) is given by
    .. math::
    a^2\frac{d^2 D_1}{da^2}+
    \left( \Omega_{\Lambda}(a)-
    \frac{ \Omega_{m}(a)}{2} +2
    \right) a \frac{dD_1}{da}=\frac{3}{2}  \Omega_{m}(a)D_1
     (see :cite:`Florent Leclercq thesis` Eq. (1.96))
    """
  a = tf.convert_to_tensor(a, dtype=tf.float32)
  # Instantiate a cosmology object
  cosmo = Cosmology(**kwcosmo)
  # Extracting entries
  (d1, d2), (d1_f, d2_f) = y
  # ODE for d1
  dy1dt = d1_f, 1.5 * Omega_m_a(cosmo, a) * d1 / tf.pow(a, 2) - (d1_f / a) * (
      Omega_de_a(cosmo, a) - 0.5 * Omega_m_a(cosmo, a) + 2)
  # ODE for d2
  dy2dt = d2_f, 1.5 * Omega_m_a(cosmo, a) * d2 / tf.pow(a, 2) - (
      d2_f / a) * (Omega_de_a(cosmo, a) - 0.5 * Omega_m_a(cosmo, a) +
                   2) - 1.5 * (Omega_m_a(cosmo, a) * d1**2) / tf.pow(a, 2)

  # Concatenate output
  dydt = [[dy1dt[0], dy2dt[0]], [dy1dt[1], dy2dt[1]]]
  return dydt


@tf.function
def odesolve_func(a, rtol=1e-4, **kwcosmo):
  """ Solves the growth ODE system for a given cosmology at the requested
    scale factors.

    Parameters
    ----------
    a: array_like
      Output scale factors, note that the ODE is initialized at a[0]

    rtol: float, optional
          Parameters determing the error control performed by the solver
    kwcosmo: keyword args
      Cosmological parameter values.

    Returns
    -------
    (D1, D1f), (D2, D2f): dictionary
      First and second order growth factors, and their derivatives, computed at
      the requested scale factors.
    """
  a = tf.convert_to_tensor(a, dtype=tf.float32)
  # Matter dominated initial condition.
  # Row 1: Initial condition of first(column 1) and second order (column 2) growth factors
  # Row 2: Initial condition of derivative of first (column 1) and second order (column 2) growth factors
  y0 = [[a[0], -3. / 7 * a[0]**2], [1.0, -6. / 7 * a[0]]]
  # Instantiate the solver
  solver = tfp.math.ode.BDF(rtol=rtol)

  # Run the ODE
  results = solver.solve(growth_ode,
                         a[0],
                         y0,
                         solution_times=a,
                         constants=kwcosmo)

  # While we are at it, compute second order derivatives growth
  second_order_results = growth_ode(results.times, results.states, **kwcosmo)

  # Normalize the ODE to present time
  # For first order growth and its derivative
  D1 = results.states[0][0] / results.states[0][0][-1]
  D1f = results.states[1][0] / results.states[0][0][-1]
  F1p = second_order_results[1][0] / results.states[0][0][-1]

  # For second order growth and its derivative
  D2 = results.states[0][1] / results.states[0][1][-1]
  D2f = results.states[1][1] / results.states[0][1][-1]
  F2p = second_order_results[1][1] / results.states[0][1][-1]

  return {
      'a': a,
      'D1': D1,
      'D1f': D1f,
      'D2': D2,
      'D2f': D2f,
      'F1p': F1p,
      'F2p': F2p
  }


def maybe_compute_ODE(cosmo, log10_amin=-2, steps=256):
  """
    Either computes or returns the cached ODE solution
    """
  if 'cache_ODE' in cosmo._workspace:
    # If cache is found in the cosmo dictionary it means the ODE has already
    # been computed
    cache = cosmo._workspace['cache_ODE']
    # Checking that the stored ODE results have the right lenght
    assert cache['a'].shape[0] == steps
  else:
    # Otherwise, we compute it now, and save the results for later
    a = tf.convert_to_tensor(np.logspace(log10_amin, 0., steps),
                             dtype=tf.float32)
    cache = odesolve_func(a, **cosmo.to_dict())
    cosmo._workspace['cache_ODE'] = cache
  return cache


def D1(cosmo, a):
  """ Normalised first order growth factor.

    Parameters
    ----------
    cosmo: dict
      Cosmology dictionary.
    a : tf.TensorArray
        Scale factor.

    Returns
    -------
    Scalar float Tensor
        normalised D1.

    Notes
    -----

    The expression for :math:`D_{1norm}(a)` is:

    .. math::

        D_{1norm}(a)=\frac{D_1(a)}{D_1(a=1)}

    """
  a = tf.convert_to_tensor(a, dtype=tf.float32)
  # Maybe compute ODE or use stored result
  cache = maybe_compute_ODE(cosmo)
  lna = tf.math.log(a)
  return tfp.math.interp_regular_1d_grid(lna, tf.math.log(cache['a'][0]),
                                         tf.math.log(cache['a'][-1]),
                                         cache['D1'])


def D2(cosmo, a):
  """ Normalised second order growth factor

    Parameters
    ----------
    cosmo: dict
      Cosmology dictionary.
    a : tf.TensorArray
        Scale factor.

    Returns
    -------
    Scalar float Tensor
        normalised D2.

    Notes
    -----

    The expression for :math:`D_{2norm}(a)` is:

    .. math::

        D_{2norm}(a)=\frac{D_2(a)}{D_2(a=1)}

    """
  a = tf.convert_to_tensor(a, dtype=tf.float32)
  # Maybe compute ODE or use stored result
  cache = maybe_compute_ODE(cosmo)
  lna = tf.math.log(a)
  return tfp.math.interp_regular_1d_grid(lna, tf.math.log(cache['a'][0]),
                                         tf.math.log(cache['a'][-1]),
                                         cache['D2'])


def D1f(cosmo, a):
  """ Derivative of the first order growth factor respect to scale factor a

    Parameters
    ----------
    cosmo: dict
      Cosmology dictionary.

    a : tf.TensorArray
        Scale factor.

    Returns
    -------
    Scalar float Tensor
        normalised derivative D1.

    Notes
    -----

    The expression for :math:`D'_{1norm}(a)` is:

    .. math::

        D'_{1norm}(a)=\frac{D'_1(a)}{D_1(a=1)}

    """
  a = tf.convert_to_tensor(a, dtype=tf.float32)
  # Maybe compute ODE or use stored result
  cache = maybe_compute_ODE(cosmo)
  lna = tf.math.log(a)
  return tfp.math.interp_regular_1d_grid(lna, tf.math.log(cache['a'][0]),
                                         tf.math.log(cache['a'][-1]),
                                         cache['D1f'])


def D2f(cosmo, a):
  """ Derivative of the second order growth factor respect to scale factor a

    Parameters
    ----------
    cosmo: dict
      Cosmology dictionary.

    a : tf.TensorArray
        Scale factor.

    Returns
    -------
    Scalar float Tensor
        normalised derivative D2.

    Notes
    -----

    The expression for :math:`D'_{2norm}(a)` is:

    .. math::

        D'_{2norm}(a)=\frac{D'_2(a)}{D_2(a=1)}

    """
  a = tf.convert_to_tensor(a, dtype=tf.float32)
  # Maybe compute ODE or use stored result
  cache = maybe_compute_ODE(cosmo)
  lna = tf.math.log(a)
  return tfp.math.interp_regular_1d_grid(lna, tf.math.log(cache['a'][0]),
                                         tf.math.log(cache['a'][-1]),
                                         cache['D2f'])


def f1(cosmo, a):
  """ Linear order growth rate

    Parameters
    ----------
    cosmo: dict
      Cosmology dictionary.

    a : tf.TensorArray
        Scale factor.

    Returns
    -------
    Scalar float Tensor
        Linear order growth rate.

    Notes
    -----

    The expression for :math:`f_{1}(a)` is:

    .. math::

        f{1}(a)=\frac{D'_1(a)}{D_1(a=1)}*a

    """
  a = tf.convert_to_tensor(a, dtype=tf.float32)
  return D1f(cosmo, a) * a / D1(cosmo, a)


def f2(cosmo, a):
  """ Second order growth rate.

    Parameters
    ----------
    cosmo: dict
      Cosmology dictionary.

    a : tf.TensorArray
        Scale factor.

    Returns
    -------
    Scalar float Tensor
        Linear order growth rate.

    Notes
    -----

    The expression for :math:`f_{2}(a)` is:

    .. math::

        f{2}(a)=\frac{D'_2(a)}{D_2(a=1)}*a

    """
  a = tf.convert_to_tensor(a, dtype=tf.float32)
  return D2f(cosmo, a) * a / D2(cosmo, a)


def Gf(cosmo, a):
  """
    FastPM growth factor function

    Parameters
    ----------
    cosmo: dict
      Cosmology dictionary.

    a : tf.TensorArray
       Scale factor.

    Returns
    -------
    Scalar float Tensor : FastPM growth factor function.

    Notes
    -----

    The expression for :math:`Gf(a)` is:

    .. math::
        Gf(a)=D'_{1norm}*a**3*E(a)
    """
  a = tf.convert_to_tensor(a, dtype=tf.float32)
  return D1f(cosmo, a) * a**3 * E(cosmo, a)


def Gf2(cosmo, a):
  """
    FastPM second order growth factor function

    Parameters
    ----------
    cosmo: dict
      Cosmology dictionary.

    a : tf.TensorArray
       Scale factor.

    Returns
    -------
    Scalar float Tensor : FastPM second order growth factor function.

    Notes
    -----

         The expression for :math:`Gf_2(a)` is:

    .. math::
        Gf_2(a)=D'_{2norm}*a**3*E(a)
    """
  a = tf.convert_to_tensor(a, dtype=tf.float32)
  return D2f(cosmo, a) * a**3 * E(cosmo, a)


def gf(cosmo, a):
  """
    Derivative of Gf against a

            Parameters
            ----------
            cosmo: dict
               Cosmology dictionary.

            a : tf.TensorArray
               Scale factor.

            Returns
            -------
            Scalar float Tensor : the derivative of Gf against a.

            Notes
            -----

         The expression for :math:`gf(a)` is:

    .. math::
        gf(a)=\frac{dGF}{da}= D^{''}_1 * a ** 3 *E(a) +D'_{1norm}*a ** 3 * E'(a)
                +   3 * a ** 2 * E(a)*D'_{1norm}

    """
  a = tf.convert_to_tensor(a, dtype=tf.float32)
  # Maybe compute ODE or use stored result
  cache = maybe_compute_ODE(cosmo)
  lna = tf.math.log(a)
  d1f = tfp.math.interp_regular_1d_grid(lna, tf.math.log(cache['a'][0]),
                                        tf.math.log(cache['a'][-1]),
                                        cache['D1f'])
  f1p = tfp.math.interp_regular_1d_grid(lna, tf.math.log(cache['a'][0]),
                                        tf.math.log(cache['a'][-1]),
                                        cache['F1p'])
  return (f1p * a**3 * E(cosmo, a) + d1f * a**3 * dEa(cosmo, a) +
          3 * a**2 * E(cosmo, a) * d1f)


def gf2(cosmo, a):
  """
    Derivative of Gf2 against a

            Parameters
            ----------
            cosmo: dict
              Cosmology dictionary.

            a : tf.TensorArray
               Scale factor.

            Returns
            -------
            Scalar float Tensor : the derivative of Gf2 against a.

            Notes
            -----

         The expression for :math:`gf2(a)` is:

    .. math::
        gf_2(a)=\frac{dGF_2}{da}= D^{''}_2 * a ** 3 *E(a) +D'_{2norm}*a ** 3 * E'(a)
                +   3 * a ** 2 * E(a)*D'_{2norm}

    """
  a = tf.convert_to_tensor(a, dtype=tf.float32)
  # Maybe compute ODE or use stored result
  cache = maybe_compute_ODE(cosmo)
  lna = tf.math.log(a)
  d2f = tfp.math.interp_regular_1d_grid(lna, tf.math.log(cache['a'][0]),
                                        tf.math.log(cache['a'][-1]),
                                        cache['D2f'])
  f2p = tfp.math.interp_regular_1d_grid(lna, tf.math.log(cache['a'][0]),
                                        tf.math.log(cache['a'][-1]),
                                        cache['F2p'])
  return (f2p * a**3 * E(cosmo, a) + d2f * a**3 * dEa(cosmo, a) +
          3 * a**2 * E(cosmo, a) * d2f)
