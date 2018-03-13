import numpy               as np
import numexpr             as ne
import Tools.CacheMgr      as Cache

from   Tools.PrintMgr      import *
from   scipy.sparse.linalg import expm_multiply
from   scipy.linalg        import solve
from   multiprocessing     import Pool
from   abc                 import abstractmethod
from   os                  import environ

####
## Base classes for implementing an orthonormal basis.
##  DecompFactory: object to track all parameters for the basis
##          and conveniently create basis objects with
##          those parameters
##
##  Basis: an iterator object used by child classes to
##          implement three-term recursion relations.
####
class ParametricObject(object):
    # List of parameters - should be overridden by subclass.
    _param    = ( )
    _fitparam = ( )
      
    # Wrap a few dict functions.
    def update(self, ind):  return  self.ParamVal.update(ind)
    def items (self):       return  self.ParamVal.items()

    # Define getitem and setitem for parameters.
    def __getitem__ (self, k):       return self.ParamVal[k]
    def __setitem__ (self, k, v):    self.ParamVal[k] = v
    def __contains__(self, k):       return k in self.ParamVal

    # Get the argument dict, updated by the contents of 'kw'
    def _arg(self, kw):
        d = self.ParamVal.copy()
        d.update(kw)
        return d

    def __init__(self, **kwargs):
        self.ParamVal = { p : kwargs[p] for p in self._param }

    def __str__(self):
        s  = [ str_nl % "className" + type(self).__name__ ]
        s += [ str_nl % k + str(self[k]) for k in self._param if k not in self._fitparam ]
        s += [ str_el % k + "%.3f" % self[k] for k in self._fitparam ]

        return '\n'.join(s)

    # Format as a list of floats joined by '_'.  The
    #  str(float()) construction ensures that numpy
    #  singletons are printed consistently
    def __format__(self, fmt):
        s = []
        s += [ str(float(self[k])) for k in self._fitparam ]
        s += [ str(self[k])        for k in self._param if k not in self._fitparam ]

        return "_".join(s)
        #return "_".join([ str(float(self[p])) for p in self._param])

class DecompFactory(ParametricObject):
    # These should be overridded by subclasses.
    _param    = ("Nbasis", "Nxfrm", "Ncheck")  # List of all parameters for constituents.

    # Methods to create the function, matrix and weight objects with correct parameters.
    # Should be over-ridden by subclasses.
    @abstractmethod
    def Fn     (self, x, w, **kw):   return
    @abstractmethod
    def MxForm (self,       **kw):   return
    @abstractmethod
    def Pri    (self,       **kw):   return
    @abstractmethod
    def Xfrm   (self,       **kw):   return

    # Return the matrix-form of Mom
    def MomMx  (self, Mom, i, j, **kwargs):
        out = kwargs.get("out", None)
        N   = kwargs.get("N", self["Nxfrm"])
        j   = min(j, self.TDot.shape[-1])
        return np.dot( self.TDot[:N,:N,i:j], Mom[i:j], out=out)

    # return the truncated series and matrix-form of bkg and signals, assuming that
    #   bkg is truncated above n
    #   sig is truncated below n.
    def CovTensor(self, *Mom):
        Nb  = Mom[0].size
        Ct  = np.zeros((Nb, Nb, len(Mom)))
        for r in self.MxForm(*Mom, Nbasis=Nb):
            Ct[r.N]  += r.Values().T
        return Ct

    def CovMatrix(self, Mom):
        Nb      = Mom.size
        Ct      = -np.outer(Mom[:Nb], Mom[:Nb])
        Ct[0,0] = 1
        for r in self.MxForm(Mom, Nbasis=Nb):
            Ct[r.N]  += r.Values()[0]
        return Ct

    def OpMatrix(self, N, **kwargs):
        M = kwargs.get("M", N)
        v = np.zeros((N,N,M))
        F = self.MxForm( *[x for x in np.eye(N)[:M]], Nbasis=N )

        for x in F:
           v[x.N] = x.Values().T
        return v

    # Decompose the dataset 'x' with weights 'w' and return the moments.
    def Decompose(self, x, w, cksize=2**20, **kw):
        Nb   = kw.pop("Nbasis", 0)
        Fn   = self.Fn(x[:cksize+1], w[:cksize+1], Nbasis=Nb if Nb > 0 else self["Nbasis"], **kw)
        Mom  = np.zeros( (Fn["Nbasis"],) )

        for i in range(0, x.size, cksize): 
            pdot()
            Fn.Reinit( x[i:i+cksize+1 ], w[i:i+cksize+1] )
            for D in Fn: Mom[D.N] += D.Moment()

        return Mom

    # Decompose the dataset and cache the results.
    @Cache.Element("{self.CacheDir}", "Decompositions", "{self}", "{2:s}-{Nbasis}.npy")
    def CachedDecompose(self, x, w, name, cksize=2**20, Nbasis=0, **kwargs):
        return self.Decompose(x, w, cksize, Nbasis=Nbasis, **kwargs)

    def __init__(self, **kwargs):
        self.Nthread    = kwargs.get('Nthread',  1)
        self.CacheDir   = kwargs.get('CacheDir', "tmp")
        self.FDDir      = environ.get('FD_DIR', ".")

        ParametricObject.__init__(self, **kwargs)
        ne.set_num_threads(self.Nthread)

        self.TDot       = self.OpMatrix( self["Nxfrm"] )
        self.TDotF      = self.OpMatrix( self["Nbasis"], M=self["Ncheck"] )

        self['Factory'] = self

class Basis(ParametricObject):
    _param      = ("Nbasis", )

    # User-facing functions.
    def Values(self):                 return self.t[ self.N % 2 ] * np.sqrt(self.NormSq(self.N))

    # These should be implemented by subclasses.
    def Zeros (self, shape=()):       return np.zeros(shape + (1,))
    def NormSq(self, N):              return None
    def Base0 (self, out):            return None
    def Base1 (self, out):            return None
    def Raise (self, out):            return None
    def Xfrm  (self, out):            return None
    def Recur (self, N, En, Ep, out): return

    # Implementation of iterator interface
    def __iter__(self, **kwargs):
        self.N       = -1

        for k, v in kwargs.items():
            self[k] = v

        if 'xf' in self.ParamVal and 'x' in self.ParamVal and self['x'].size == self['xf'].size:
            self.Xfrm  (self['xf'])
            self.Raise (self['rz'])
        else:
            self['xf'] = self.Xfrm(None)
            self['rz'] = self.Raise(None)
            self['t']  = self.Zeros((2,))
        self.Base0 (self['t'][0])
        self.Base1 (self['t'][1])
      
        return self

    def next(self):
        next    = self['t'][ (self.N + 1) % 2 ]
        this    = self['t'][ (self.N    ) % 2 ]
        prev    = self['t'][ (self.N - 1) % 2 ]

        if   self.N < 0:                   self.Base0(next)
        elif self.N == 0:                  self.Base1(next)
        elif self.N < self['Nbasis']:      self.Recur(self.N, this, prev, next)
        if   self.N == self['Nbasis'] - 1: raise StopIteration

        self.N += 1
        return self

    def __str__(self):
        s  = [ str_nl % "className" + type(self).__name__ ]
        s += [ str_nl % k           + str(self[k]) for k in self._param ]

        return '\n'.join(s)


# These go with the Transform object.  Must be globals for Multiprocessing
gK = {}
def gEle(n, m): return gK['obj'].Ele(n, m, **gK)

class Transform(ParametricObject):
    _param = ( )

    # Transform a moment vector into a new basis with specified parameters.
    def __call__(self, *vec, **kwargs):
        inv = kwargs.get("inv", False)

        ini = kwargs        if inv else self.ParamVal
        fin = self.ParamVal if inv else kwargs
        ret = np.stack(vec, axis=1)[:self["Nxfrm"]]
        
        for p in self._param:
            try:
                arg = getattr(self, "par" + p)(fin[p], ini)
                ret = expm_multiply( arg * self.KMx[p].T, ret )
            except KeyError:
                pass
        return tuple(ret.T)

    # The actual element computation
    @Cache.AtomicElement("{self.Factory.FDDir}", "data", "ele-cache-{name:s}", "{0:d}-{1:d}.json")
    def Ele(self, n, m, **kwargs):
        h   = kwargs.get("h")
        v   = np.outer(self.O[n][::-1], self.D[m])
        s   = [ np.trace(v, offset=k) for k in range(-v.shape[0]+1, v.shape[1]) ]

        return np.dot(h[:len(s)], s) * self.Norm(n, m)

    @staticmethod
    def OrthogonalizeHankel(func):
        @Cache.Element("{self.Factory.FDDir}", "data", "xfrm-cache-{0:s}-{1:d}.npy")
        def wrap(self, name, N):
            pini(name + " xfrm moment")
            global gK
            gK = {
               'h'   : [ pdot(func(self, k)) for k in range(2*N-1) ],
               'name': name,
               'obj' : self
            }
            pend()

            pini(name + " xfrm matrix", interval=N)
            p  = Pool( self.Factory.Nthread )
            r  = [ p.apply_async(gEle, (n, m), callback=pdot) for n in range(N) for m in range(N) ]
            p.close()
            r  = np.array([ x.get() for x in r]).reshape((N, N))
            pend()

            return r
        return wrap

    # Calculate infinitesimal transformations
    def __init__(self, **kwargs):
        ParametricObject.__init__(self, **kwargs)

        self["Nxfrm"]   = kwargs["Nxfrm"]
        self.Factory    = kwargs["Factory"]
        self.O          = [ self.CoeffOrth (n) for n in range(self["Nxfrm"]) ]
        self.D          = [ self.CoeffDeriv(n) for n in range(self["Nxfrm"]) ]
        self.KMx        = { n : getattr(self, n)(n, self["Nxfrm"]) for n in self._param }

