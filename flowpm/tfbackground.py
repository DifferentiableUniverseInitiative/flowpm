"""TensorFlow implementation of Cosmology Computations"""
import tensorflow as tf

def fde(cosmo,a,epsilon=1e-5):
  """TODO: add documentation
  """
  a=tf.convert_to_tensor(a,dtype=tf.float32)
  w0=tf.convert_to_tensor(cosmo["w0"],dtype=tf.float32)
  wa=tf.convert_to_tensor(cosmo["wa"],dtype=tf.float32)
  return (-3.0*(1.0+w0)+
            3.0*wa*((a-1.0)/tf.math.log(a-epsilon)-1.0))

def w(cosmo,a):
  """TODO: add documentation
  """
  a=tf.convert_to_tensor(a,dtype=tf.float32)
  w0=tf.convert_to_tensor(cosmo["w0"],dtype=tf.float32)
  wa=tf.convert_to_tensor(cosmo["wa"],dtype=tf.float32)
  return w0+wa*(1.0-a)

def E(cosmo,a):
  """TODO: add documentation
  """
  a=tf.convert_to_tensor(a,dtype=tf.float32)
  return(tf.math.sqrt(
        cosmo["Omega0_m"]/tf.pow(a, 3)
        +cosmo["Omega0_k"]/tf.pow(a, 2)
        +cosmo["Omega0_de"]*tf.pow(a, fde(cosmo,a))))

def dfde(cosmo,a,epsilon=1e-5):
  """TODO: add documentation
  """
  a=tf.convert_to_tensor(a,dtype=tf.float32)
  wa=tf.convert_to_tensor(cosmo["wa"],dtype=tf.float32)
  return (3*wa*
            (tf.math.log(a-epsilon)-(a-1)/(a-epsilon))
            /tf.math.pow(tf.math.log(a-epsilon),2))

def dEa(cosmo,a,epsilon=1e-5):
  """TODO: add documentation
  """
  a=tf.convert_to_tensor(a,dtype=tf.float32)
  return 0.5*(-3*Omega0_m/tf.pow(a, 4)-2*Omega0_k/tf.pow(a, 3)
          + dfde(cosmo,a)*Omega0_de*tf.pow(a, fde(cosmo,a)))
