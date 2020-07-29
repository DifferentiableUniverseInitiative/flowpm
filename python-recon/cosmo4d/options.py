from abopt.abopt2 import Preconditioner

def Pvp(x, direction):
    if direction == -1:
        return x.r2c() * x.Nmesh.prod()
    else:
        return x.r2c()

def vPp(x, direction):
    if direction == -1:
        return x.c2r()
    else:
        return x.c2r() / x.Nmesh.prod()

UseComplexSpaceOptimizer = Preconditioner(Pvp=Pvp, vPp=vPp)
#UseComplexSpaceOptimizer = Preconditioner(Pvp=lambda x: x.r2c(), Qvp=lambda x:x.r2c() * x.Nmesh.prod(), vQp=lambda x:x.c2r())
UseRealSpaceOptimizer = None
