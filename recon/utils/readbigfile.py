import numpy as np

def readhead(path):
    shape = None
    fnames = []
    with open(path +'/attr-v2') as  f:
        for line in f.readlines():
            if 'ndarray.shape' in line:
                shape = tuple(int(i) for i in line.split('[')[1].split()[:-1])
    with open(path +'header') as  f:
        for line in f.readlines():
            if 'DTYPE' in line: dtype = line.split()[-1]
            elif 'NFILE' in line: nf = int(line.split()[-1])
            elif 'NMEMB' in line:
                if shape is None: shape = tuple([-1, int(line.split()[-1])])
            else: fnames.append(line[:6])
    return dtype, nf, shape, fnames


def readbigfile(path):
    dtype, nf, shape, fnames = readhead(path)
    data = []
    for i in range(nf):
        data.append(np.fromfile(path + '%s'%fnames[i], dtype=dtype))
    data = np.concatenate(data)
    data = np.reshape(data, shape)
    return data



