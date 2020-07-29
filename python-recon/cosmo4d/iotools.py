import numpy
from nbodykit.source.mesh.field import FieldMesh
from nbodykit.source.mesh.bigfile import BigFileMesh 
from nbodykit.source.catalog.file import BigFileCatalog

def save_map(s, filename, dataset):
    x = FieldMesh(s)
    x.save(filename, dataset=dataset, mode='real')

def load_map(filename, dataset):
    x = BigFileMesh(filename, dataset)
    return x.paint(mode='real')

def load_catalog(filename, dataset):
    x = BigFileCatalog(filename, header='Header')[dataset].compute()
    return x

def distribute_image(pm, image):
    # create a distributed image object, as an ndarray
    from skimage.transform import resize

    pm2d = pm.resize((pm.Nmesh[0], pm.Nmesh[1], 1))
    fimage = resize(image, pm.Nmesh[:2], order=0, mode='wrap', preserve_range=True).ravel()
    image1 = pm2d.create(mode='real')

    allsizes = pm.comm.allgather(image1.size)
    fimage = fimage[sum(allsizes[:pm.comm.rank]):sum(allsizes[:pm.comm.rank+1])]
    image1.unravel(fimage)
    return image1[...].copy()

def collect_image(pm, image):
    pm2d = pm.resize((pm.Nmesh[0], pm.Nmesh[1], 1))
    im = numpy.zeros(pm2d.Nmesh, dtype=pm.dtype)[..., 0].copy()
    image1 = pm2d.create(mode='real')
    image1[...] = image[...]
    return image1.preview(axes=(0, 1))


def create_figure(figsize, gssize):
    from matplotlib.figure import Figure
    from matplotlib.gridspec import GridSpec
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize)
    canvas = FigureCanvasAgg(fig)
    gs = GridSpec(*gssize)
    return fig, gs
