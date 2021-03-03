import tensorflow as tf
from functools import partial


class Cosmology:
    def __init__(self, Omega_c, Omega_b, h, n_s, sigma8, Omega_k, w0, wa):
        """
        Cosmology object, stores primary and derived cosmological parameters.
        Parameters:
        -----------
        Omega_c, float
          Cold dark matter density fraction.
        Omega_b, float
          Baryonic matter density fraction.
        h, float
          Hubble constant divided by 100 km/s/Mpc; unitless.
        n_s, float
          Primordial scalar perturbation spectral index.
        sigma8, float
          Variance of matter density perturbations at an 8 Mpc/h scale
        Omega_k, float
          Curvature density fraction.
        w0, float
          First order term of dark energy equation
        wa, float
          Second order term of dark energy equation of state
        """
        # Store primary parameters
        self._Omega_c = tf.convert_to_tensor(Omega_c, dtype=tf.float32)
        self._Omega_b = tf.convert_to_tensor(Omega_b, dtype=tf.float32)
        self._h = tf.convert_to_tensor(h, dtype=tf.float32)
        self._n_s = tf.convert_to_tensor(n_s, dtype=tf.float32)
        self._sigma8 = tf.convert_to_tensor(sigma8, dtype=tf.float32)
        self._Omega_k = tf.convert_to_tensor(Omega_k, dtype=tf.float32)
        self._w0 = tf.convert_to_tensor(w0, dtype=tf.float32)
        self._wa = tf.convert_to_tensor(wa, dtype=tf.float32)

        self._flags = {}

        # Create a workspace where functions can store some precomputed
        # results
        self._workspace = {}

    def __str__(self):
        return (
            "Cosmological parameters: \n"
            + "    h:        "
            + str(self.h)
            + " \n"
            + "    Omega_b:  "
            + str(self.Omega_b)
            + " \n"
            + "    Omega_c:  "
            + str(self.Omega_c)
            + " \n"
            + "    Omega_k:  "
            + str(self.Omega_k)
            + " \n"
            + "    w0:       "
            + str(self.w0)
            + " \n"
            + "    wa:       "
            + str(self.wa)
            + " \n"
            + "    n:        "
            + str(self.n_s)
            + " \n"
            + "    sigma8:   "
            + str(self.sigma8)
        )

    def __repr__(self):
        return self.__str__()

    # Cosmological parameters, base and derived
    @property
    def Omega(self):
        return 1.0 - self._Omega_k

    @property
    def Omega_b(self):
        return self._Omega_b

    @property
    def Omega_c(self):
        return self._Omega_c

    @property
    def Omega_m(self):
        return self._Omega_b + self._Omega_c

    @property
    def Omega_de(self):
        return self.Omega - self.Omega_m

    @property
    def Omega_k(self):
        return self._Omega_k

    @property
    def k(self):
        if self.Omega > 1.0:  # Closed universe
            k = 1.0
        elif self.Omega == 1.0:  # Flat universe
            k = 0
        elif self.Omega < 1.0:  # Open Universe
            k = -1.0
        return k

    @property
    def sqrtk(self):
        return tf.math.sqrt(tf.math.abs(self._Omega_k))

    @property
    def h(self):
        return self._h

    @property
    def w0(self):
        return self._w0

    @property
    def wa(self):
        return self._wa

    @property
    def n_s(self):
        return self._n_s

    @property
    def sigma8(self):
        return self._sigma8


# To add new cosmologies, we just set the parameters to some default values
# using partial

# Planck 2015 paper XII Table 4 final column (best fit)
Planck15 = partial(
    Cosmology,
    Omega_c=0.2589,
    Omega_b=0.04860,
    Omega_k=0.0,
    h=0.6774,
    n_s=0.9667,
    sigma8=0.8159,
    w0=-1.0,
    wa=0.0,
)
