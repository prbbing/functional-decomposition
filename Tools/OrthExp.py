import numpy               as np
import numexpr             as ne
import Tools.Base          as Base

from   Tools.PrintMgr      import *
from   numpy.core.umath    import euler_gamma
from   fractions           import Fraction
from   math                import log, sqrt, ceil

####
## An implementation of an orthonormal basis:
##   the orthogonal exponentials.
####

#### Orthogonal exponentials as functions
class ExpDecompFn (Base.Basis):
    _param = Base.Basis._param + ( "x0", "Lambda", "Alpha" )

    # User-facing functions.
    def Values(self):           return self['t'][ self.N % 2 ] * np.sqrt(self.NormSq(self.N)) * self['xf']

    def Zeros (self, shape=()): return np.zeros(shape + self['x'].shape)

    def NormSq(self, N):        return 2. / N if N > 0 else 0
    def Base0 (self, out):      out[:] = 0
    def Base1 (self, out):      out[:] = self['rz']
    def Raise (self, out):      return ne.evaluate( "exp( -(x-x0) * xf / Alpha)", self.ParamVal, out=out)
    def Xfrm  (self, out):      return ne.evaluate( "(Alpha/Lambda) * ((x-x0)/Lambda)**(Alpha-1)", self.ParamVal, out=out)

    # The recurrence relation itself.
    def Recur(self, N, En, Ep, out):
        A      = float(4*N + 2) / N
        B      = float(4*N)     / (2*N - 1)
        C      = float(2*N + 1) / (2*N - 1)

        ne.evaluate( "A*rz*En - B*En - C*Ep", global_dict = self, out=out)

    def Moment(self):           return np.dot(self['t'][ self.N % 2 ], self['w']) * sqrt(self.NormSq(self.N))

    def __init__(self, x, w, **kwargs):
        Base.Basis.__init__(self, **kwargs)

        self['x'] = x
        self['w'] = w

    def Reinit(self, x, w):
        self['x'] = x
        self['w'] = w

#### Class to elevate a decomposition to it's matrix form
class ExpDecompMxForm (Base.Basis):
    def Values(self):           return self['t'][ self.N % 2 ][:,1:self["Nbasis"]+1]
    def Zeros (self, shape=()): return np.zeros(shape + self['x'].shape)

    def Base0 (self, out):      out[:] = 0
    def Base1 (self, out):      out[:] = 0; self.Recur(0, self['x'], self['t'][0], out)
    def Raise (self, out):
        n           = np.arange(self["Nrow"]-2, dtype=np.float)

        out         = np.zeros((2, self["Nrow"]))
        out[0,1:-1] = np.sqrt(n*(n+1)) / (      2*(2*n+1) ) #sub/sup diagonal
        out[1,1:-1] =           2*n**2 / ((2*n-1)*(2*n+1) ) #diagonal
        out[1,-2]   = 0

        return out
    def Xfrm  (self, out):      return None

    # The recurrence relation itself.
    def Recur(self, N, En, Ep, out):
        N = float(N)
        if N > 0:
            e  = sqrt(     N / (N+1) )
            f  = sqrt( (N-1) / (N+1) )

            A  = e * (4*N + 2) / N
            B  = e * (4*N)     / (2*N - 1)
            C  = f * (2*N + 1) / (2*N - 1)
        else:
            A,B,C = 2.0,0.0,0.0

        dg  = self['rz'][1,1:-1]
        us  = self['rz'][0,1:-1]
        ds  = self['rz'][0,0:-2]

        x   = En[:,1:-1]
        u   = En[:,2:  ]
        d   = En[:, :-2]
        p   = Ep[:,1:-1]

        return ne.evaluate("A*(dg*x + us*u + ds*d) - B*x - C*p ", out=out[:,1:-1])

    def __init__(self, *mom, **kwargs):
        Base.Basis.__init__(self, **kwargs)

        Nmom          = max([x.size for x in mom])
        Nbasis        = self["Nbasis"]
        Nrow          = Nmom + Nbasis + 2

        self["Nrow"]  = Nrow
        self['x']     = np.zeros((len(mom), Nrow))

        for n,x in enumerate(mom):
            self['x'][n,1:len(x)+1] = x

#### Decomposition of a simple exponential e**(-x/Lambda) 
class ExpPrior (Base.Basis):
    _param = Base.Basis._param + ( "Alpha", "Lambda" )

    def Moment(self):
        if self.N != 1:
            return 0
        return 2**(1./2 - 1./self['Alpha'])

#### Transformation matrix generator for orthogonal exponentials.
class ExpDecompTransform(Base.Transform):
    _param = Base.Transform._param + ("Lambda", "Alpha")

    # Return a high-accuracy rational approximation of log(n) for integer n
    # Use recursion and cache results to enhance performance.
    def _log(self, n, numIni=96, numMin=48):
        if n <= 1: return 0
        if n not in self:
            num     = max( numIni - (n - 2), numMin )
            s       = [ Fraction(2, 2*k+1) * Fraction(1, 2*n-1) ** (2*k+1) for k in range(num) ]
            self[n] = self._log(n-1) + sum( s )
        return self[n]

    # Recursion relations for series coefficients
    def CoeffOrth(self, n):
        r = [ 0, Fraction((-1)**(n+1) * n) ]

        for m in range(1, n):
            r.append(  Fraction(m**2 - n**2, m*(m+1)) * r[-1]  )
        return r

    def CoeffDeriv(self, n):
        r = [ 0, Fraction((-1)**n * n) ]

        for m in range(1, n):
            r.append(  Fraction(m**2 - n**2, m**2) * r[-1]  )
        return r

    def Norm (self, *arg):   return sqrt(np.prod([ 2*a for a in arg]))

    # Infinitesimal transformations
    @Base.Transform.OrthogonalizeHankel
    def Alpha    (self, n):   return (1 - Fraction(euler_gamma) - self._log(n)) / n**2 if n > 0 else 0
    @Base.Transform.OrthogonalizeHankel
    def Lambda   (self, n):   return Fraction(1, n**2)    if n > 0 else 0

    # Infinitesimal transformation parameters
    # 'fin' is the final value for the parameter;
    # 'ini' is a dict with the initial value for all parameters
    def parAlpha (self, fin, ini): return log( fin / ini["Alpha"] )
    def parLambda(self, fin, ini): return -ini["Alpha"] * log(fin / ini["Lambda"])

#### A decomposer factory using the orthonormal exponentials
class ExpDecompFactory ( Base.DecompFactory ):
    _param    = tuple(set( Base.DecompFactory._param
                          + ExpDecompFn._param
                          + ExpDecompMxForm._param
                          + ExpPrior._param))
    _fitparam = ("Alpha", "Lambda")

    # Methods to create the function, matrix and weight objects with correct parameters.
    def Fn    (self, x, w, **kw): return ExpDecompFn        (x, w, **self._arg(kw) )
    def MxForm(self, *x,   **kw): return ExpDecompMxForm    (*x,   **self._arg(kw) )
    def Pri   (self,       **kw): return ExpPrior           (      **self._arg(kw) )
    def Xfrm  (self,       **kw): return ExpDecompTransform (      **self._arg(kw) )
 
