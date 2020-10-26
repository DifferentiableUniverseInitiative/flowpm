import tensorflow as tf
import tensorflow_probability as tfp
from nbodykit import cosmology
import numpy as np
from flowpm.utils import r2c2d, c2r2d


def generate(di, df, ds, boxsize, boxsize2D, ):
    """ Returns a list of rotation matrices and shifts that are applied to the box
    di : distance to inital redshift (before taking fastpm step)
    df : distance to final redshift (after taking fastpm step)
    ds : Source distance
    boxsize: size of computational box
    """
    # Generate the possible rotation matrices
    x      = np.asarray([1,0,0],dtype=int)
    y      = np.asarray([0,1,0],dtype=int)
    z      = np.asarray([0,0,1],dtype=int)

    # shuffle directions, only 90 deg rotations, makes a total of 6
    M_matrices = [np.asarray([x,y,z],dtype=int), np.asarray([x,z,y],dtype=int),np.asarray([z,y,x],dtype=int),np.asarray([z,x,y],dtype=int), \
                         np.asarray([y,x,z],dtype=int), np.asarray([y,z,x],dtype=int)]
    # Generate possible xy shifts
    I = np.zeros(3)
    fac =0.5
    xyshifts = [np.asarray([fac,fac,0.],dtype=float),np.asarray([-fac,fac,0.],dtype=float),np.asarray([-fac,-fac,0.],dtype=float),np.asarray([fac,-fac,0.],dtype=float)]

    # Find the maximum number of rotations needed in the z direction
    vert_num = ds * boxsize2D / boxsize[-1]

    print('rotations available: %d'%len(M_matrices))
    print('rotations required: %d'%np.ceil(ds/boxsize[-1]))

    try:
        assert(len(M_matrices)*boxsize[-1]>ds)
        print('sufficient number of rotations to fill lightcone.')
    except:
        print('insufficient number of rotations to fill the lightcone.')

    if df>ds:
        return 0, 0
    else:
        shift_ini = np.floor(max(di,ds)/boxsize[-1])
        shift_end = np.floor(df/boxsize[-1])
        M = []
        if vert_num==1:
            for ii in np.arange(shift_end,shift_ini+1,dtype=int):
                M.append((M_matrices[ii%len(M_matrices)], I+ii*z))
        elif vert_num==2:
            for ii in np.arange(shift_end,shift_ini+1,dtype=int):
                for jj in range(4):
                    M.append((M_matrices[ii%len(M_matrices)], I+ii*z+xyshifts[jj]))
        else:
            raise ValueError('vertical number of boxes must be 1 or 2, but is %d'%vert_num)

        return M



def rotate(x, M, boxsize, boxshift, name='rotate'):
    """
    rotates, shift, and separates particle coordinates into distance and xy position
    x:        particle positions
    M:        rotation matrix
    boxshift: shift vector
    """
    with tf.name_scope(name):
        y  = tf.einsum('ij,kj->ki', M, x)
        y  = tf.add(y, boxsize*boxshift)
        d  = tf.gather(y, 2, axis=1, name='gather_z')
        xy = tf.gather(y, (0, 1), axis=1, name='gather_xy')
        return xy, d

def z_chi(d, cosmo, name='z_chi'):
    with tf.name_scope(name):
        # redshift as a function of comsoving distance for underlying cosmology
        z_int          = np.logspace(-12,np.log10(1500),40000)
        chis           = cosmo.comoving_distance(z_int) #Mpc/h
        z = tfp.math.interp_regular_1d_grid(d, 1e-12, 1.5e3, tf.convert_to_tensor(chis, dtype='float'), name='interpolation')
        return z

def wlen(d, ds, cosmo, boxsize, boxsize2D, mesh2D, name='efficiency_kernel'):
    """
    returns the correctly weighted lensing efficiency kernel
    d:   particle distance (assuming parllel projection)
    ds:  source redshift
    """
    with tf.name_scope(name):
        # Find the redshift at ds
        z= z_chi(d, cosmo)
        # Find the number density of simulated particles
        nbar = tf.shape(d)/boxsize[0]**3

        #Find angular pixel area for projection
        A = mesh2D[0]**2/boxsize2D**2
        columndens = tf.multiply(tf.pow(d,2),float(nbar*A)) #particles/Volume*angular pixel area* distance^2 -> 1/L units
        w          = tf.divide(tf.multiply(tf.multiply(tf.subtract(ds, d), tf.divide(d, ds)), (1.+z)), columndens) #distance
        return w
