import tensorflow as tf
from flowpm.scipy.integrate import simps

def smail_nz(z,a,b,z0):
    """Defines a smail distribution with these arguments
    
    Parameters:
    -----------
    
    z: array_like or tf.TensorArray
       Photometric redshift
        
             
    a: float value
        Parameter of Smail distribution
    
    b: float value
        Parameter of Smail distribution
    
    z0: float value
        Parameter of Smail distribution
        
    Notes
    -----

    The expression for :math:`n(z)` is:

    .. math::

      \n(z)=z^{a}\exp{-(z/z_0)^b}
    """

    smail_nz=[]
    zmax=10.0
    def smail(z,a,b,z0):
        return (z) ** a * tf.math.exp(-(((z) / z0) ** b))
    for i in range(len(z0)):
        norm = simps(lambda t: smail(t,a,b,z0[i]), 0.0, 10, 256)
        norm_smail=tf.cast(smail(z,a,b,z0[i]),dtype=tf.float32)/tf.cast(norm,dtype=tf.float32)
        smail_nz.append(norm_smail)
    return tf.stack(smail_nz, axis=0)
    
def systematic_shift(z,bias):
    """Defines a smail distribution with these arguments
    
    Parameters:
    -----------
    
    z: array_like or tf.TensorArray
       Photometric redshift array
        
             
    bias: float value
        Nuisance parameters defining the uncertainty of the redshift distributions
    
    """
    z = tf.convert_to_tensor(z, dtype=tf.float32)
    return (tf.clip_by_value(z - bias, 0,50))