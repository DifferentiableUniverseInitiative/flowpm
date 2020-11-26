import tensorflow as tf
import numpy as np

import flowpm.tfbackground as bkgrd
import flowpm.constants as const
from flowpm.scipy.integrate import simps

def primordial_matter_power(cosmo, k):
    r"""Primordial power spectrum
    Pk = k^n

    Parameters
    ----------
    cosmo: dictionary
      Input cosmology dictionary.
    k: array_like
      Input scale at which to evaluate the PPS

    Returns:
    pk: array_like
      Primordial power spectrum  evaluated at requested scales
    """
    k = tf.convert_to_tensor(k, dtype=tf.float32)
    return k ** cosmo['n_s']

def Eisenstein_Hu(cosmo, k, type="eisenhu_osc"):
    """Computes the Eisenstein & Hu matter transfer function.
    Parameters
    ----------
    cosmo: dictionary
      Background cosmology
    k: array_like
      Wave number in h Mpc^{-1}
    type: str, optional
      Type of transfer function. Either 'eisenhu' or 'eisenhu_osc'
      (def: 'eisenhu_osc')
    Returns
    -------
    T: array_like
      Value of the transfer function at the requested wave number
    Notes
    -----
    The Eisenstein & Hu transfer functions are computed using the fitting
    formulae of :cite:`1998:EisensteinHu`
    """
    k = tf.convert_to_tensor(k, dtype=tf.float32)
    #############################################
    # Quantities computed from 1998:EisensteinHu
    # Provides : - k_eq   : scale of the particle horizon at equality epoch
    #            - z_eq   : redshift of equality epoch
    #            - R_eq   : ratio of the baryon to photon momentum density
    #                       at z_eq
    #            - z_d    : redshift of drag epoch
    #            - R_d    : ratio of the baryon to photon momentum density
    #                       at z_d
    #            - sh_d   : sound horizon at drag epoch
    #            - k_silk : Silk damping scale
    T_2_7_sqr = (const.tcmb / 2.7) ** 2
    h2 = cosmo["h"] ** 2
    w_m = cosmo["Omega0_m"] * h2
    w_b = cosmo["Omega0_b"] * h2
    fb = cosmo["Omega0_b"] / cosmo["Omega0_m"]
    fc = (cosmo["Omega0_m"] - cosmo["Omega0_b"]) / cosmo["Omega0_m"]

    k_eq = 7.46e-2 * w_m / T_2_7_sqr / cosmo["h"]  # Eq. (3) [h/Mpc]
    z_eq = 2.50e4 * w_m / (T_2_7_sqr) ** 2  # Eq. (2)

    # z drag from Eq. (4)
    b1 = 0.313 * tf.math.pow(w_m, -0.419) * (1.0 + 0.607 * tf.math.pow(w_m, 0.674))
    b2 = 0.238 * tf.math.pow(w_m, 0.223)
    z_d = (
        1291.0
        * tf.math.pow(w_m, 0.251)
        / (1.0 + 0.659 * tf.math.pow(w_m, 0.828))
        * (1.0 + b1 * tf.math.pow(w_b, b2))
    )

    # Ratio of the baryon to photon momentum density at z_d  Eq. (5)
    R_d = 31.5 * w_b / (T_2_7_sqr) ** 2 * (1.0e3 / z_d)
    # Ratio of the baryon to photon momentum density at z_eq Eq. (5)
    R_eq = 31.5 * w_b / (T_2_7_sqr) ** 2 * (1.0e3 / z_eq)
    # Sound horizon at drag epoch in h^-1 Mpc Eq. (6)
    sh_d = (
        2.0
        / (3.0 * k_eq)
        * tf.math.sqrt(6.0 / R_eq)
        * tf.math.log((np.sqrt(1.0 + R_d) + tf.math.sqrt(R_eq + R_d)) / (1.0 + tf.math.sqrt(R_eq)))
    )
    # Eq. (7) but in [hMpc^{-1}]
    k_silk = (
        1.6
        * tf.math.pow(w_b, 0.52)
        * tf.math.pow(w_m, 0.73)
        * (1.0 + tf.math.pow(10.4 * w_m, -0.95))
        / cosmo["h"]
    )
    #############################################

    alpha_gamma = (
        1.0
        - 0.328 * tf.math.log(431.0 * w_m) * w_b / w_m
        + 0.38 * tf.math.log(22.3 * w_m) * (cosmo["Omega0_b"] / cosmo["Omega0_m"]) ** 2
    )
    gamma_eff = (
        cosmo["Omega0_m"]
        * cosmo["h"]
        * (alpha_gamma + (1.0 - alpha_gamma) / (1.0 + (0.43 * k * sh_d) ** 4))
    )

    if type == "eisenhu":

        q = k * tf.math.pow(const.tcmb / 2.7, 2) / gamma_eff

        # EH98 (29) #
        L = tf.math.log(2.0 * tf.math.exp(1.0) + 1.8 * q)
        C = 14.2 + 731.0 / (1.0 + 62.5 * q)
        res = L / (L + C * q * q)

    elif type == "eisenhu_osc":
        # Cold dark matter transfer function

        # EH98 (11, 12)
        a1 = tf.math.pow(46.9 * w_m, 0.670) * (1.0 + tf.math.pow(32.1 * w_m, -0.532))
        a2 = tf.math.pow(12.0 * w_m, 0.424) * (1.0 + tf.math.pow(45.0 * w_m, -0.582))
        alpha_c = tf.math.pow(a1, -fb) * tf.math.pow(a2, -(fb ** 3))
        b1 = 0.944 / (1.0 + tf.math.pow(458.0 * w_m, -0.708))
        b2 = tf.math.pow(0.395 * w_m, -0.0266)
        beta_c = 1.0 + b1 * (tf.math.pow(fc, b2) - 1.0)
        beta_c = 1.0 / beta_c

        # EH98 (19). [k] = h/Mpc
        def T_tilde(k1, alpha, beta):
            # EH98 (10); [q] = 1 BUT [k] = h/Mpc
            q = k1 / (13.41 * k_eq)
            L = tf.math.log(tf.math.exp(1.0) + 1.8 * beta * q)
            C = 14.2 / alpha + 386.0 / (1.0 + 69.9 * tf.math.pow(q, 1.08))
            T0 = L / (L + C * q * q)
            return T0

        # EH98 (17, 18)
        f = 1.0 / (1.0 + (k * sh_d / 5.4) ** 4)
        Tc = f * T_tilde(k, 1.0, beta_c) + (1.0 - f) * T_tilde(k, alpha_c, beta_c)

        # Baryon transfer function
        # EH98 (19, 14, 21)
        y = (1.0 + z_eq) / (1.0 + z_d)
        x = tf.math.sqrt(1.0 + y)
        G_EH98 = y * (-6.0 * x + (2.0 + 3.0 * y) * tf.math.log((x + 1.0) / (x - 1.0)))
        alpha_b = 2.07 * k_eq * sh_d * tf.math.pow(1.0 + R_d, -0.75) * G_EH98

        beta_node = 8.41 * tf.math.pow(w_m, 0.435)
        tilde_s = sh_d / tf.math.pow(1.0 + (beta_node / (k * sh_d)) ** 3, 1.0 / 3.0)

        beta_b = 0.5 + fb + (3.0 - 2.0 * fb) * tf.math.sqrt((17.2 * w_m) ** 2 + 1.0)

        # [tilde_s] = Mpc/h
        Tb = (
            T_tilde(k, 1.0, 1.0) / (1.0 + (k * sh_d / 5.2) ** 2)
            + alpha_b
            / (1.0 + (beta_b / (k * sh_d)) ** 3)
            * tf.math.exp(- tf.math.pow(k / k_silk, 1.4))
        ) * (tf.math.sin(k * tilde_s )/(k * tilde_s + 1e-9)) # TODO: Replace by sinc when possible

        # Total transfer function
        res = fb * Tb + fc * Tc
    else:
        raise NotImplementedError
    return res

def linear_matter_power(cosmo, k, a=1.0, transfer_fn=Eisenstein_Hu, **kwargs):
    r"""Computes the linear matter power spectrum.
    Parameters
    ----------
    k: array_like
        Wave number in h Mpc^{-1}
    a: array_like, optional
        Scale factor (def: 1.0)
    transfer_fn: transfer_fn(cosmo, k, **kwargs)
        Transfer function
    Returns
    -------
    pk: array_like
        Linear matter power spectrum at the specified scale
        and scale factor.
    """
    k=tf.convert_to_tensor(k,dtype=tf.float32)
    a=tf.convert_to_tensor(a,dtype=tf.float32)

    g = bkgrd.D1(cosmo, a)
    t = transfer_fn(cosmo, k, **kwargs)

    pknorm = cosmo["sigma8"] ** 2 / sigmasqr(cosmo, 8.0, transfer_fn, **kwargs)

    pk = primordial_matter_power(cosmo, k) * t ** 2 * g ** 2

    # Apply normalisation
    pk = pk * pknorm
    return tf.squeeze(pk)

def sigmasqr(cosmo, R, transfer_fn, kmin=0.0001, kmax=1000.0, ksteps=5, **kwargs):
    """Computes the energy of the fluctuations within a sphere of R h^{-1} Mpc
    .. math::
       \\sigma^2(R)= \\frac{1}{2 \\pi^2} \\int_0^\\infty \\frac{dk}{k} k^3 P(k,z) W^2(kR)
    where
    .. math::
       W(kR) = \\frac{3j_1(kR)}{kR}
    """
    def int_sigma(logk):
        k = np.exp(logk)
        x = k * R
        w = 3.0 * (np.sin(x) - x * np.cos(x)) / (x * x * x)
        pk = transfer_fn(cosmo, k, **kwargs) ** 2 * primordial_matter_power(cosmo, k)
        return k * (k * w) ** 2 * pk

    y = simps(int_sigma, np.log10(kmin), np.log10(kmax), 256)
    return 1.0 / (2.0 * np.pi ** 2.0) * y
