import numpy               as np
import numexpr             as ne
import Tools.CacheMgr      as Cache

from   Tools.PrintMgr      import *
from   Tools.Base          import ParametricObject
from   scipy.optimize      import minimize
from   scipy.linalg        import solve
from   numpy.linalg        import slogdet, multi_dot
from   scipy.linalg        import cho_factor, cho_solve
from   math                import pi, e, sqrt
from   scipy.special       import erfinv

import time

###
## Conduct a signal scan on a DataSet and store the results.
###
class SignalScan(ParametricObject):
    def _setShortCov(self, SigMoms, P):
        n        = self.DataSet.N
        m        = P.shape[0]
        Sct      = self.Factory.CovTensor( SigMoms[-1] )

        self.Bcs = np.zeros((m, m, n))
        self.Scs = np.zeros((m, m, m))

        for i in range(n):
            self.Bcs[:,:,i] = multi_dot( ( P, self.Bct[i,n:,n:], P.T ) )
        for i in range(m-1):
            self.Scs[:,:,i] = multi_dot( ( P, self.Sct[i,n:,n:], P.T ) )
        self.Scs[:,:,m-1]   = multi_dot( ( P, Sct[n:,n:,0], P.T ) )

    # covariance helper function
    def _cov(self, t):
        n   = self.DataSet.N
        Bcs = self.Bcs[:,:,:n]
        Scs = self.Scs

        return Bcs.dot(self.b) + Scs.dot(t) - np.outer(t, t)

    # Return the log-likelihood of data assuming t
    def _gvecprob(self, t):
        cov   = self._cov(t) / self.DataSet.Nint
        delta = self.x - t

        if self.Lumi > 0:
            cov += self.LumiUnc * np.outer(t, t) / self.DataSet.Nint

        try:
            ch = cho_factor(cov)
            c  = delta.dot(cho_solve(ch, delta))
            l  = np.log(np.diag(ch[0])).sum()

            return c/2 + l
        except (np.linalg.linalg.LinAlgError, ValueError):
            return np.inf

    def UL(self, x, p = 0.05):
        self.x = x
        s      = x.dot(self.dvec)

        proj   = x - s * self.dvec          # NP signal projected off
        base   = proj if s < 0 else x       # Base point for limit
        ini    = base + 2*self.Sb

        ptgt   = self._gvecprob(base) - sqrt(2) * erfinv(p-1)
        gcon   = lambda t: ptgt - self._gvecprob(t)
        gobj   = lambda t: -self.DataSet.Nint*np.dot(t, self.dvec)

        res    = minimize(gobj, ini, constraints={'type': 'eq', 'fun': gcon },
                                     options={'maxiter': 200, 'eps': 1e-12})
        if not res.success:
            print res
        return -res.fun / self.Lumi if self.Lumi > 0 else - res.fun

    # Implementation of iterator.
    def next(self):
        self.idx      += 1

        if self.idx >= len(self.Names):
            raise StopIteration

        Data           = self.DataSet
        SigName        = self.Names[self.idx]
        signals        = [ Data[name] for name in Data.GetActive(SigName) ]

        DataMom        = getattr(Data, Data.attr)
        SigMoms, P     = Data.NormSignalEstimators(SigName)

        self._setShortCov(SigMoms, P)

        self.dvec      = np.array( [ x.name == SigName for x in signals ] )
        Rb             = np.array( [ x.Moment for x in signals ] )
        Rs             = P.dot(DataMom[Data.N:])

        # Nominal background
        self.b         = DataMom[:Data.N] - Rb.dot( SigMoms[:,:Data.N] )

        # Store the output values and calculate limits
        i              = self.idx
        S              = multi_dot((self.dvec, self._cov(Rb), self.dvec))
        self.Sb        = sqrt(S / Data.Nint)

        self.Mass  [i] = Data[SigName].Mass
        self.Yield [i] = Rs.dot(self.dvec) * Data.Nint
        self.Unc   [i] = sqrt(S * Data.Nint)
        self.ObsLim[i] =   self.UL( Rs )
        self.ExpLim[i] = [ self.UL( Rb + v*self.Sb*self.dvec) for v in range(-2, 3) ]

        return SigName, self.Yield[i], self.Unc[i], self.ObsLim[i], self.ExpLim[i]

    # Initialize covariances for shared signals.  Transposes copies
    #   are stored in memory, because this optimizes memory access
    #   and greatly speeds up the sums in _setShortCov.
    def __iter__(self):
        sl       = self.DataSet.GetActive()
        Sig      = np.array([ self.DataSet[s].Sig for s in sl ])
        self.Bct = self.Factory.TDotF.T[:self.Factory["Ncheck"]]
        self.Sct = self.Factory.CovTensor( *Sig ).T.copy()

        return self

    def __init__(self, Factory, DataSet, *arg, **kwargs):
        ParametricObject.__init__(self)

        self.Factory    = Factory
        self.DataSet    = DataSet
        self.Lumi       = float(kwargs.get("Lumi",    0.0))
        self.LumiUnc    = float(kwargs.get("LumiUnc", 0.0))

        self.Names      = arg
        self.Mass       = np.zeros( (len(arg),   ) )
        self.Yield      = np.zeros( (len(arg),   ) )
        self.Unc        = np.zeros( (len(arg),   ) )
        self.ExpLim     = np.zeros( (len(arg), 5 ) )
        self.ObsLim     = np.zeros( (len(arg),   ) )

        self.idx        = -1

###
## Optimize hyperparameters on a specified DataSet.
###
class Optimizer(ParametricObject):
    # Transform the dataset to the specified hyperparameters
    def UpdateXfrm(self, reduced=True, **kwargs):
        self.update( kwargs )
        self.PriorGen.update(self.ParamVal)
        self.Prior.FromIter(self.PriorGen)

        Act        = self.DataSet.GetActive() if reduced else self.DataSet.Signals

        inm        = [ self.DataSet[name].Mom for name in Act ]
        outm       = self.Xfrm(self.DataSet.Mom, *inm, **self.ParamVal)
        Mom, sigs  = outm[0], outm[1:]

        for name, m in zip(Act, sigs):
            self.DataSet[name].MomX[len(m):] = 0
            self.DataSet[name].MomX[:len(m)] = m

        self.DataSet.MomX[:] = Mom
        self.DataSet.Full.Set (Mom=Mom)

    # Scan for the optimal number of moments.
    @Cache.Element("{self.Factory.CacheDir}", "LLH", "{self.Factory}", "{self}", "{self.DataSet}.json")
    def ScanN(self, reduced=True, **kwargs):
        L = np.full((self.Factory["Ncheck"],), np.inf)
        D = self.DataSet

        self.UpdateXfrm(reduced=reduced)

        for j in range(2, self.Factory["Ncheck"]):
            try:
                D.SetN(j)

                Raw  = D.TestS.Chi2(D.Full)
                Pen  = j * np.log(j / 2*pi*e) / 2  # prior strength = j

                pdot()
            except (np.linalg.linalg.LinAlgError, ValueError):
                pdot(pchar="x")
                continue
            L[j] = Raw + Pen
        j = np.nanargmin(L)

        return j, L[j]

    # Optimize the hyperparameters
    @pstage("Optimizing Hyperparameters")
    def FitW(self):
        ini = [ self.Factory[p] for p in self.Factory._fitparam ]
        res = minimize(self.ObjFunc, ini, method='Nelder-Mead', options={'xatol': 1e-2})
        par = dict(zip(self.Factory._fitparam, res.x))

        # Now set to the best parameters
        self.update( par )
        self.UpdateXfrm()
        
        N, L = self.ScanN()
        self.DataSet.SetN( N )

        return L, par

    # Scan through a grid of hyperparameters
    @pstage("Scanning Hyperparameters")
    def ScanW(self, *par):
        self.Nfex = par[0].size

        LLH  = np.vectorize(lambda *arg: self.ObjFunc(arg) )(*par)
        best = np.where(LLH == LLH.min())
        x    = [ v[best] for v in par ]

        return LLH, dict(zip(self.Factory._fitparam, x))

    def ObjFunc(self, arg):
        prst()
        pstr(str_nl % "Nint" + "%.2f" % self.DataSet.Nint)
        pstr(str_nl % "Signals" + " ".join(self.DataSet.GetActive()))
        pstr(str(self))

        pini("MOMENT SCAN")

        self.update( dict( zip(self.Factory._fitparam, arg) ) )
        self.Xfrm.update(self.Factory.ParamVal)
        j, L       = self.ScanN()

        pstr(str_nl % "Nmom" + "%2d"  % j)
        pstr(str_nl % "LLH"  + "%.2f" % L)
        pstr(str_nl % "Nfev" +  "%d / %d"  % (self.Nfev, self.Nfex))

        self.Nfev += 1

        return L

    def __init__(self, Factory, DataSet, **kwargs):
        self.DataSet   = DataSet
        self.Factory   = Factory
        self.Nfev      = 0
        self.Nfex      = 0

        # Copy parameters from the Factory object.
        self._param    = Factory._fitparam
        self._fitparam = Factory._fitparam
        ParametricObject.__init__(self, **Factory.ParamVal)

        self.Prior     = TruncatedSeries(self.Factory, np.zeros((Factory["Nbasis"],)), 1.0, Nmax=2 )
        self.PriorGen  = Factory.Pri()
        self.Xfrm      = self.Factory.Xfrm()

###
## An object to hold data to decompose along with signal objects.
###
class DataSet(ParametricObject):
    def AddParametricSignal(self, name, func, **kwargs):
        if name in self:
            return
        self[name] = ParametricSignal(self.Factory, name, func, **kwargs)
        self.Signals.append(name)

    def DelSignal(self, name):
        try:
            list.remove(name)
        except ValueError:
            pass
        try:
           del self[name]
        except ValueError:
           pass

    def GetActive(self, *args):  return [ n for n in self.Signals if self[n].Active or self[n].name in args ]

    # Decompose the dataset and active signals
    @pstage("Decomposing")
    def Decompose(self, reduced=True, xonly=False, cksize=2**20):
        pini("Data Moments")
        N             = self.Factory["Nxfrm"]
        Nb            = self.Factory["Nxfrm"] if xonly else self.Factory["Nbasis"]

        self.Mom      = np.zeros((self.Factory["Nbasis"],))
        self.MomX     = np.zeros((self.Factory["Nxfrm"],))

        self.Mom[:Nb] = self.Factory.CachedDecompose(self.x, self.w, str(self.uid), cksize=cksize, Nbasis=Nb)

        self.Full     = TruncatedSeries(self.Factory, self.Mom, self.Nint, Nmax=N )
        self.TestS    = TruncatedSeries(self.Factory, self.Mom, self.Nint, Nmax=N )
        self.TestB    = TruncatedSeries(self.Factory, self.Mom, self.Nint, Nmax=N )
        pend()

        Act = self.GetActive() if reduced else self.Signals
        for name  in Act:
            pini("%s Moments" % name)
            self[name].Decompose(cksize=cksize, Nbasis=Nb)
            pend()

    # Solve for the raw signal estimators.  Use Cholesky decomposition,
    #   as it is much faster than the alternative solvers.
    @pstage("Preparing Signal Estimators")
    def PrepSignalEstimators(self, reduced=True, verbose=False):
        D     = getattr(self, self.attr).copy()
        n, N  = self.N, D.size
        Act   = self.GetActive() if reduced else self.Signals

        D[n:] = 0
        LCov  = self.Factory.TDotF[:N,:N,:n].dot( D[:n] ) - np.outer(D, D)
        LCov += np.eye( N ) * n / self.Nint
        Ch    = cho_factor( LCov )

        if verbose: pini("Solving")
        for name in Act:
            sig     = self[name]
            sig.Sig = getattr(sig, self.attr)                           # set the moments to use.
            sig.Res = sig.Sig[n:]                                       # sig res
            sig.Est = cho_solve(Ch, sig.Sig.T)[n:]

            if verbose: pdot()
        if verbose: pend()

    # Solve for the normalized signal estimators
    def NormSignalEstimators(self, *extrasignals):
        sl   = self.GetActive(*extrasignals)

        Sig  = np.array([ self[s].Sig for s in sl ])                    # raw signals
        Res  = np.array([ self[s].Res for s in sl ])                    # sig residuals
        Est  = np.array([ self[s].Est for s in sl ])                    # sig raw estimators

        P    = solve ( Res.dot(Est.T), Est, assume_a='pos')             # normalized estimators

        return Sig, P

    # Extract the active signals
    def ExtractSignals(self, Data, *extrasignals):
        N            = self.N
        Sig, P       = self.NormSignalEstimators(self, *extrasignals)
        R            = P.dot(Data[self.N:])

        self.FullSig = R.dot(Sig)
        self.P       = P

        for i, name in enumerate( self.GetActive(*extrasignals) ): 
            self[name].Moment = R[i]
            self[name].Yield  = R[i] * self.Nint

    # Return covariance matrix (in events)
    def Covariance(self):
        sigs = self.GetActive()
        N    = self.N

        if len(sigs) == 0:
            self.Cov  = [[]]
            self.Unc  = []
            self.Corr = [[]]
        else:
            Mf        = self.Factory.CovMatrix( getattr(self, self.attr) )
            self.Cov  = self.Nint * multi_dot (( self.P, Mf[N:,N:], self.P.T ))
            self.Unc  = np.sqrt(np.diag(self.Cov))
            self.Corr = self.Cov / np.outer(self.Unc, self.Unc)

        for n, name in enumerate(sigs):
            self[name].Unc = self.Unc[n]

    # Set the number of moments
    def SetN(self, N, attr="MomX"):
        self.N = N
        Mom    = getattr(self, attr)
        isSig  = np.arange(Mom.size) >= N
        dMom   = Mom * (~isSig)

        self.attr = attr

        if len(self.Signals) > 0:
            self.PrepSignalEstimators(verbose=False)
            self.ExtractSignals(Mom)
        else:
            self.FullSig = np.zeros_like(dMom)

        self.TestB.Set(Mom=dMom - self.FullSig * (~isSig), Nmax=N)
        self.TestS.Set(Mom=dMom + self.FullSig * ( isSig) )

    def __init__(self, x, Factory, **kwargs):
        w              = kwargs.get('w', np.ones(x.shape[-1]))
        sel            = (w != 0) * (x > Factory["x0"] )
        self.w         = np.compress(sel, w, axis=-1)
        self.x         = np.compress(sel, x, axis=-1)

        # Record some vital stats
        self.uid       = self.x.dot(self.w)  # Use the weighted sum as a datset identifier.
        self.Nint      = self.w.sum()
        self.w        /= self.Nint

        self.Factory   = Factory
        self.Signals   = []

        ParametricObject.__init__(self, **Factory.ParamVal)

    # Format as a list of floats joined by '_'.  The
    #  str(float()) construction ensures that numpy
    #  singletons are p.copy()rinted consistently
    def __format__(self, fmt):
        id_str = [ str(self.uid) ] + self.GetActive()
        return "_".join( id_str )

###
## A parametric signal model.
###
class ParametricSignal(object):
    def Decompose(self, cksize=2**20, Nbasis=0, **kwargs):
        Mom                 = self.CachedDecompose(cksize, Nbasis=Nbasis, **kwargs)
        self.Mom[:len(Mom)] = Mom
        self.Mom[len(Mom):] = 0
        self.MomX[:]        = 0

    # Decompose the signal sample data.
    @Cache.Element("{self.Factory.CacheDir}", "Decompositions", "{self.Factory}", "{self.name}.npy")
    def CachedDecompose(self, cksize=2**20, **kwargs):
        Nb           = kwargs.pop("Nbasis", 0)
        Mom          = np.zeros((self.Factory["Nbasis"],))
        sumW         = 0.0
        sumW2        = 0.0
        Neff         = 0
        cksize       = min(cksize, self.Npt)
        Fn           = self.Factory.Fn(np.zeros((cksize,)), w=np.zeros((cksize,)),
                                       Nbasis=Nb if Nb > 0 else self.Factory["Nbasis"])

        while Neff < self.Npt:
            Fn['x']    = np.random.normal(loc=self.mu, scale=self.sigma, size=cksize)
            Fn['w']    = self.func(Fn['x']) / self._gauss(Fn['x'])

            k          = np.nonzero(Fn['x'] <= self.Factory["x0"])
            Fn['w'][k] = 0
            Fn['x'][k] = 2*self.Factory["x0"]

            sumW      += Fn['w'].sum()
            sumW2     += np.dot(Fn['w'], Fn['w'])
            Neff       = int(round( (sumW*sumW)/sumW2 ))

            for D in Fn: Mom[D.N] += D.Moment()
            pdot()

        return Mom / sumW

    # Gaussian PDF
    def _gauss(self, x):
        u, s = self.mu, self.sigma
        return np.exp( -0.5*( (x-u)/s )**2 ) / sqrt(2*pi*s*s)

    # Set a signal model and generate
    def __init__(self, Factory, name, func, mu, sigma, **kwargs):
        self.Factory = Factory
        self.name    = name
        self.func    = func
        self.mu      = mu
        self.sigma   = sigma
        self.Mass    = float(kwargs.get("Mass", self.mu)) 
        self.Npt     = int(  kwargs.get("NumPoints", 2**22))
        self.Active  = bool( kwargs.get("Active", False))

        self.Moment  = 0
        self.Yield   = 0

        self.Mom     = np.zeros((self.Factory["Nbasis"],))
        self.MomX    = np.zeros((self.Factory["Nxfrm"],))

###
## A truncated orthonormal series
###
class TruncatedSeries(object):
    # Evaluate series
    def __call__(self, x, trunc=False):
        Mom      = self.MomAct if trunc else self.MomU
        Val      = np.zeros_like(x)
        w        = np.ones_like(x)

        for D in self.Factory.Fn(x, w, Nbasis=Mom.size):
            if D.N >= self.Nmin:
                Val  += D.Values() * Mom[D.N]

        return Val

    # Get the common index range between self and other
    def _ci(self, othr):
        return ( max(self.Nmin, othr.Nmin),
                 max(self.Nmax, othr.Nmax))

    # Get the entropy of this TruncatedSeries.
    def Entropy(self, **kwargs):
       j     = kwargs.get('j', self.Nmin)
       k     = kwargs.get('k', self.Nmax)
       Cov   = self.Cov[j:k,j:k]

       return slogdet(2*pi*e*Cov)[1] / 2

    # Dkl(othr||self) --> prior.KL(posterior). If specified, scale the
    #  statistical precision of 'self' by Scalee
    def KL(self, othr):
        j, k        = self._ci(othr)
        delta       = self.MomAct[j:k] - othr.MomAct[j:k]

        ChSelf      = cho_factor(self.Cov[j:k,j:k])
        h           = cho_solve(ChSelf, delta)
        r           = cho_solve(ChSelf, othr.Cov[j:k,j:k])

        return (np.trace(r) + delta.dot(h) - slogdet(r)[1] - k + j) / 2

    #Log-likelihood of othr with respect to self.
    def Chi2(self, othr):
        j, k        = self._ci(othr)
        delta       = self.MomAct[j:k] - othr.MomAct[j:k]

        Ch          = cho_factor(self.Cov[j:k,j:k])
        h           = cho_solve(Ch, delta)

        return delta.dot(h) / 2

    # Negative log-likelihood of othr with respect to self.
    def LogP(self, othr):
        j, k        = self._ci(othr)
        delta       = self.MomAct[j:k] - othr.MomAct[j:k]

        Ch          = cho_factor(self.Cov[j:k,j:k])
        h           = cho_solve(Ch, delta)
        l           = 2*np.log(np.diag(Ch[0])).sum()

        return (  (k-j)*np.log(2*pi) + l + delta.dot(h)) / 2

    # Set the number of active moments.
    def Set(self, **kwargs):
        self.Nmin   = kwargs.get('Nmin', self.Nmin)
        self.Nmax   = kwargs.get('Nmax', self.Nmax)
        self.Mom    = kwargs.get('Mom',  self.Mom).copy()

        # Truncate or pad with zeros as necessary.  Keep a copy of the original.
        self.MomU = self.Mom.copy()
        self.Mom.resize( (self.Factory["Nbasis"],), )

        R           = np.arange(self.Mom.size, dtype=np.int)
        self.MomAct = self.Mom * (self.Nmin <= R) * (R < self.Nmax)

        # Build covariance matrix
        N           = self.Cov.shape[0]
        self.Mx     = self.Factory.MomMx( self.Mom, self.Nmin, self.Nmax, out=self.Mx)
        self.Cov    = (self.Mx - np.outer(self.MomAct[:N], self.MomAct[:N])) / self.StatPre

    # Set the moments from an iterator
    def FromIter(self, iter):
        Mom         = np.asarray( [ D.Moment() for D in iter ] )
        self.Set(Mom=Mom)

    # Initialize by taking the decomposition of the dataset `basis'.
    #   Store all moments.  The `active' moments are [self.Nmin, self.Nmax)
    def __init__(self, Factory, Moments, StatPre=1, **kwargs):
        self.Factory = Factory
        self.StatPre = StatPre
        self.Nmin    = kwargs.get("Nmin", 1)
        self.Nmax    = kwargs.get("Nmax", Factory["Nbasis"])

        Nbasis       = Factory["Nbasis"]
        Nxfrm        = Factory["Nxfrm"]

        self.Mx      = np.zeros( (Nxfrm, Nxfrm) )
        self.Cov     = np.zeros( (Nxfrm, Nxfrm) )     # Covariance (weighted)
        self.MomAct  = np.zeros( (Nbasis,) )          # Weighted, truncated moments
        self.MomU    = Moments.copy()
        self.Mom     = Moments.copy()

        self.Set()

