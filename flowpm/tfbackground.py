"""TensorFlow implementation of Cosmology Computations"""
import numpy as np
import tensorflow as tf
import flowpm
import tensorflow_probability as tfp

#Planck 2015 results
cosmo={"w0":-1.0,
       "wa":0.0,
       "H0":100,
       "h":0.6774,
       "Omega0_b":0.04860,
       "Omega0_c":0.2589,
       "Omega0_m":0.3075,
       "Omega0_k":0.0,
       "Omega0_de":0.6925,
       "n_s":0.9667,
       "sigma8":0.8159}

def fde(cosmo,a,epsilon=1e-5):
   r"""Evolution parameter for the Dark Energy density.

    Parameters
    ----------
    a : array_like or tf.TensorArray
        Scale factor

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
   a=tf.convert_to_tensor(a,dtype=tf.float32)
   w0=tf.convert_to_tensor(cosmo["w0"],dtype=tf.float32)
   wa=tf.convert_to_tensor(cosmo["wa"],dtype=tf.float32)
   return (-3.0*(1.0+w0)+ 3.0*wa*((a-1.0)/tf.math.log(a-epsilon)-1.0))




def w(cosmo,a):
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
  a=tf.convert_to_tensor(a,dtype=tf.float32)
  w0=tf.convert_to_tensor(cosmo["w0"],dtype=tf.float32)
  wa=tf.convert_to_tensor(cosmo["wa"],dtype=tf.float32)
  return w0+wa*(1.0-a)

def E(cosmo,a):
  r"""The scale factor dependent factor E(a) in the Hubble
    parameter.

    Parameters
    ----------
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
  a=tf.convert_to_tensor(a,dtype=tf.float32)
  return(tf.math.sqrt(
        cosmo["Omega0_m"]/tf.math.pow(a, 3)
        +cosmo["Omega0_k"]/tf.math.pow(a, 2)
        +cosmo["Omega0_de"]*tf.math.pow(a, fde(cosmo,a))))

def H(cosmo, a):
    r"""Hubble parameter [km/s/(Mpc/h)] at scale factor `a`

    Parameters
    ----------
    a : array_like or tf.TensorArray
        Scale factor

    Returns
    -------
    H : Scalar float Tensor
        Hubble parameter at the requested scale factor.
    """
    a=tf.convert_to_tensor(a,dtype=tf.float32)
    return cosmo["H0"]*cosmo["h"] *(E(cosmo, a))


def dfde(cosmo,a,epsilon=1e-5):
    r"""Derivative of the evolution parameter for the Dark Energy density
    f(a) with respect to the scale factor.

    Parameters
    ----------
    a : array_like or tf.TensorArray
        Scale factor

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
    a=tf.convert_to_tensor(a,dtype=tf.float32)
    wa=tf.convert_to_tensor(cosmo["wa"],dtype=tf.float32)
    return (3*wa*
            (tf.math.log(a-epsilon)-(a-1)/(a-epsilon))
            /tf.math.pow(tf.math.log(a-epsilon),2))


def dEa(cosmo,a,epsilon=1e-5):
    r"""Derivative of the scale factor dependent factor E(a) in the Hubble
    parameter with respect to the
    scale factor.

    Parameters
    ----------
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
    a=tf.convert_to_tensor(a,dtype=tf.float32)
    return 0.5*(-3*cosmo["Omega0_m"]/tf.math.pow(a, 4)-2*cosmo["Omega0_k"]/tf.math.pow(a, 3)
    +dfde(cosmo,a)*cosmo["Omega0_de"]*tf.math.pow(a, fde(cosmo,a)))/E(cosmo,a)


def Omega_m_a(cosmo,a):
    r"""Matter density at scale factor `a`.

    Parameters
    ----------
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
    a=tf.convert_to_tensor(a,dtype=tf.float32)
    return cosmo["Omega0_m"]*tf.math.pow(a,-3)/E(cosmo,a)**2


def Omega_de_a(cosmo,a):
    r"""Dark Energy density at scale factor `a`.

    Parameters
    ----------
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
    a=tf.convert_to_tensor(a,dtype=tf.float32)
    return cosmo["Omega0_de"]*tf.math.pow(a,fde(cosmo,a))/E(cosmo,a)**2

#Equation 1.96 from Florent Leclercq thesis
@tf.function
def growth_func(a,y):
    """Compute linear growth factor D_1(a) and second-order growth factor D_2(a) at a given scale factor
    Parameters
    ----------
    a: array_like or tf.TensorArray
      Scale factor
    amin: float
      Mininum scale factor, default 1e-3
    Returns
    -------
    D1:  Scalar float Tensor
        First order growth factor computed at requested scale factor
    D2: Scalar float Tensor
        Second orderd growth factor computed at requested scale factor
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
    y=tf.convert_to_tensor(y,dtype=tf.float32)
    a=tf.convert_to_tensor(a,dtype=tf.float32)
    d = y[0]
    d_f = y[1]
    # EXtracting d1 and derivative
    d1 = d[0]
    d1_f = d_f[0]
    # Extractinct d2 and derivative
    d2 = d[1]
    d2_f = d_f[1]
    # ODE for d1
    dy1dt= d1_f,1.5*Omega_m_a(cosmo,a)*d1/tf.pow(a,2)-(d1_f/a)*(Omega_de_a(cosmo,a)-0.5*Omega_m_a(cosmo,a)+2)
    # ODE for d2
    dy2dt = d2_f,1.5*Omega_m_a(cosmo,a)*d2/tf.pow(a,2)-(d2_f/a)*(Omega_de_a(cosmo,a)-0.5*Omega_m_a(cosmo,a)+2)  - 1.5*(Omega_m_a(cosmo,a)*d1**2)/tf.pow(a,2)

    # Concatenate output
    dydt = [[dy1dt[0], dy2dt[0]],
            [dy1dt[1], dy2dt[1]]]
    return dydt

@tf.function
def odesolve_func(atab,y0):
    solver=tfp.math.ode.BDF(rtol=1e-4)
    results = solver.solve(growth_func,atab[0], y0, solution_times=atab)
    return results

if __name__ == "__main__":
  """ Tests implementation and draws first and second order growth.
  """
  import matplotlib.pyplot as plt
  log10_amin=-3
  steps=128
  atab =np.logspace(log10_amin, 0.0, steps)
  y0=tf.constant([[atab[0], -3./7 * 0.001**2], [1.0, -6. / 7 *0.001]],dtype=tf.float32)

  results_func=odesolve_func(atab,y0)

  #normalise D1 and D2 such that D1(a=1) = 1 and D2(a=1) = 1
  D1_norm=results_func.states[:,0,0]/results_func.states[-1,0,0]
  D2_norm=results_func.states[:,0,1]/results_func.states[-1,0,1]
  d1_fn=results_func.states[:,1,0]/results_func.states[-1,0,0]
  d2_fn=results_func.states[:,1,1]/results_func.states[-1,0,1]

  plt.plot(results_func.times,D1_norm,label='$D_1$')
  plt.plot(results_func.times,D2_norm,label='$D_2$')
  plt.xlabel('Scale factor')
  plt.ylabel('Growth function')
  plt.legend()
  plt.show()
       
       
       
       

################################################################


#for first derivative of D1 and D2 against log a
def F1(a):
    """
    First derivative of D1 against log a
    """
    #results_func=odesolve_func(atab,y0)
    return results_func.states[:,1,0]*a/results_func.states[:,0,0]


def F2(a):
    """
    First derivative of D2 against log a
    """
    #results_func=odesolve_func(atab,y0)
    return results_func.states[:,1,1]*a/results_func.states[:,0,1]

###############################################################
def Gf(a):
    """
    FastPM growth factor function
    """
    #results_func=odesolve_func(atab,y0)
    d1_f=results_func.states[:,1,0]
    d1_f_norm=d1_f/results_func.states[-1,0,0]
    return (d1_f_norm)*a**3*E(cosmo,a)

def Gf2(a):
    """ Gf but for second order LPT
    """
    #results_func=odesolve_func(atab,y0)
    d2_f=results_func.states[:,1,1]
    d2_f_norm=d2_f/results_func.states[-1,0,1]
    return (d2_f_norm)*a**3*E(cosmo,a) 
  
################################################################

def gf(a):
    D1_order2=1.5*Omega_m_a(cosmo,a)*D1/tf.pow(a,2)-(d1_f/a)*(Omega_de_a(cosmo,a)-0.5*Omega_m_a(cosmo,a)+2)
    D1_order2=D1_order2/results_func.states[-1,0,0]
    return  (D1_order2 * a ** 3 *E(cosmo,a) +  d1_fn*a ** 3 * dEa(cosmo,a)
                +   3 * a ** 2 * E(cosmo,a)*d1_fn)



def gf2(a):
    D2_order2=1.5*Omega_m_a(cosmo,a)*D2/tf.pow(a,2)-(d2_f/a)*(Omega_de_a(cosmo,a)-0.5*Omega_m_a(cosmo,a)+2)- 1.5*(Omega_m_a(cosmo,a)*D1**2)/tf.pow(a,2) 
    D2_order2=D2_order2/results_func.states[-1,0,1]
    return  (D2_order2 * a ** 3 *E(cosmo,a) +  d2_fn*a ** 3 * dEa(cosmo,a)
                +   3 * a ** 2 * E(cosmo,a)*d2_fn)

