#!/usr/bin/env python

import os, sys, signal, argparse, ConfigParser
import numpy                           as np
import matplotlib.pyplot               as plt
import Plots.Plots                     as Plots

from   Tools.Decomp                    import DataSet, Optimizer, SignalScan
from   Tools.OrthExp                   import ExpDecompFactory
from   Tools.ConfigIter                import ConfigIter
from   Tools.PrintMgr                  import *
from   Plots.Plots                     import *

from   matplotlib.backends.backend_pdf import PdfPages

# Load a specified variable from base dir
def loadVar(varName, DSList, Dec):
    Path = [ os.path.join(ArgC.base, "Data", n, varName + ".npy") for n in DSList ]
    Ar   = [ np.load(p, mmap_mode = 'r')[::int(d)] for p, d in zip(Path, Dec) ]

    return np.concatenate(Ar)

# Test a condition on on cached variables
def testVar(cond, DSList, Dec):
    Keep = []

    for n, d in zip(DSList, Dec):
        Path  = os.path.join ( ArgC.base, "Data", n )
        Names = [ os.path.splitext(k)[0] for k in os.listdir ( Path ) ]
        Full  = { k : os.path.join(Path, k + ".npy") for k in Names }
        Vars  = { k : np.load(Full[k], mmap_mode = 'r')[::int(d)] for k in Names }

        Keep.append(eval(cond, Vars))
    return np.concatenate(Keep)

# Evaluate a signal function string.
def _fmtSignalFunc(x, **kwargs):
    try:
        return eval(x.format(**kwargs))
    except AttributeError:
        return x

def _addSignalFunc(D, func, name, M, **kw):
    Active = kw.get("Active", (M is None))
    arg    = { k : _fmtSignalFunc(v, Mass=M, **kw) for k, v in c.items() }
    ffunc  = _fmtSignalFunc(func, **arg)

    D.AddParametricSignal(name, ffunc, Active=Active, **arg)

########################################
########################################

# Stop stack trace on ctrl-c
def nice_quit(signal, frame):
    sys.exit(0)
signal.signal(signal.SIGINT, nice_quit)

# Lower default NP print precision.
np.set_printoptions(precision=4)

######
# Parse command line parameters and config files
######
ArgP      = argparse.ArgumentParser(description='=== Functional Decomposition Fitter ===')
ArgP.add_argument('--base', type=str, default=".", help="FD base directory.")
ArgP.add_argument('--show', action="store_true",   help="Display plots interactively.")
ArgC      = ArgP.parse_args()

Config    = ConfigParser.ConfigParser()
Config.optionxform = str
Config.read( os.path.join(ArgC.base, "base.conf") )

PlotStyle = Config.get("General", "PlotStyle")
try:            plt.style.use( PlotStyle )
except IOError: plt.style.use( os.path.join(ArgC.base, PlotStyle) )

# Reduce numpy default print precision to make logs a bit neater.
np.set_printoptions(precision=2)





######
# Initialize the factory and signal models.
######

# Get initial hyperparameter scan range
ranges  = { p : r for p, r in Config.items("HyperParameterScan") }
rlist   = [ ranges[p] for p in ExpDecompFactory._fitparam ]
A, L    = eval( "np.mgrid[%s]" % ",".join(rlist) )

# Create Factory configuration
fConf   = { k: eval(v) for k, v in Config.items('ExpDecompFactory') }
for p, r in zip(ExpDecompFactory._fitparam, rlist):
    fConf[p] = eval( "np.mgrid[%s][0]" % r )
fConf['CacheDir'] = os.path.join(ArgC.base, "Cache")

# Read input variables.
SetList   = Config.items("InputFiles")
varName   = Config.get("General", "Variable")
wgtName   = Config.get("General", "Weight")
Scale     = Config.get("General", "Scale")
Lumi      = Config.get("General", "Lumi",      0.0)
LumiUnc   = Config.get("General", "LumiUnc",   0.0)
XSecUnits = Config.get("General", "XSecUnits") if Lumi > 0 else "Events"


print
print
print "Input files:", " ".join( [n for n, d in SetList] )

x       = loadVar(varName, *zip(*SetList))
w       = loadVar(wgtName, *zip(*SetList)) * float(Scale)
cutflow = [ ("Initial", w.sum()) ]

for name, cond in Config.items("Cuts"):
    w *= testVar(cond, *zip(*SetList))
    cutflow.append((name, w.sum()))

# Create objects
Factory = ExpDecompFactory( **fConf )
Tr      = Factory.Xfrm()
D       = DataSet(x, Factory, w=w)
FOpt    = Optimizer(Factory, D)

Names   = {}
Scans   = {}

# Initialize signal peaks
for c, name in ConfigIter(Config, "ParametricSignal", "ParametricSignalDefault"):
   func = "lambda x:" + c.pop("func")
   Scan = c.pop("Scan", None)

   if Scan is None:
       _addSignalFunc(D, func, name, None, **c)
   else:
       Names[name] = [ name + "%.3f" % M for M in Scan ]

       for M, fname in zip( Scan, Names[name] ):
           _addSignalFunc(D, func, fname, M, **c)

print

######
# Decompose
######

### decompose; scan hyperparameters; decompose; fit from best location.
D.Decompose(xonly=True)
LLH, PBest        = FOpt.ScanW(A, L)
Factory.update( PBest )
D.Decompose(xonly=True)
LBest, PBest      = FOpt.FitW()
NBest             = D.N

### Re-decompose, extract signals, and validate that xfrm gave good moment estimates.
MomX = D.Full.Mom.copy()
Factory.update( PBest)
D.Decompose(reduced=False)
FOpt.UpdateXfrm(**PBest)
D.SetN(N=NBest, attr="Mom")    # attr="Mom": use the full expansions from here on out
MomT = D.Full.Mom.copy()

D.Covariance()
if len(D.Signals) > 0:
    D.PrepSignalEstimators(reduced=False, verbose=True)

### Calculate yields, uncertainties and CLs
fmt = {
  "wht": "\x1b[37m %8.1f \x1b[0m",
  "grn": "\x1b[32m %8.1f \x1b[0m",
  "yel": "\x1b[33m %8.1f \x1b[0m",
  "red": "\x1b[31m %8.1f \x1b[0m",
}
lfmt = [ "red", "yel", "grn", "wht", "grn", "yel", "red" ]

print
print
print "=====> YIELD AND LIMITS <====="
print
print "%-16s: %8s +- %8s (%5s) [ %9s  ] [ %9s  %9s  %9s  %9s  %9s  ]" % (
       "Signal", "Yield", "Unc", "Sig.", "Obs. CL95",
       "-2sigma", "-1sigma", "Exp. CL95", "+1sigma", "+2sigma")

for scan_name, sig_names in Names.items():
    Scans[scan_name] = SignalScan(Factory, D, *sig_names, Lumi=Lumi, LumiUnc=LumiUnc)

    t1 = time.time()
    for name, yld, unc, obs, exp in Scans[scan_name]:
        t2   = time.time()
        sig  = yld / unc
        isig = 3 + np.clip(int(sig) + (1 if sig > 0 else -1), -3, 3)

        print "%-16s: %8.1f +- % 8.1f (% 4.2f)" % (name, yld, unc, sig),
        print "[", fmt[lfmt[ isig ]] % obs, "] [",
        for l, e in zip(lfmt[1:-1], exp):
            print fmt[l] % e,
        print "] (%4.2fs)" % (t2-t1)

        t1 = time.time()

######
# Output fit results
######
Nxfrm = Factory["Nxfrm"]
print
print
print "=====> CUTFLOW <====="
for c in cutflow:
    print "% 12s: %.2f" % c
print
print
print "=====> SIGNAL RESULTS <====="
for name in D.GetActive():
    s = D[name]
    print "% 12s: %.2f +- %.2f" % (name, s.Yield, s.Unc)
print
print
print "=====> COVARIANCE < ====="
print D.Cov
print
print
print "=====> CORRELATION < ====="
print D.Corr
print
print
print "=====> FRACTIONAL DIFFERENCE BETWEEN TRANSFORMED AND DIRECT MOMENTS <====="
print ((MomX[1:Nxfrm] - MomT[1:Nxfrm])/MomT[1:Nxfrm])

######
# Plotting
######
print
print
print "=====> PLOTTING < ====="
def op(*x):
    return os.path.join(ArgC.base, "Output", *x)

try:
    os.mkdir(op())
except OSError:
    pass

pdf = PdfPages(op('all.pdf'))
ini =   fConf["Lambda"],   fConf["Alpha"]
fin = Factory["Lambda"], Factory["Alpha"]

Plots.cutflow      (cutflow,             pdf=pdf, fname=op('cutflow.pdf'))
Plots.scan         (L, A, LLH, ini, fin, pdf=pdf, fname=op('hyperparameter_scan.pdf'))
Plots.summary_table(D,                   pdf=pdf, fname=op('signal_summary.pdf'))

for p, file in ConfigIter(Config, "Plot", "PlotDefault"):
    Pull = p.pop("DrawPull", True)

    try:
        Type = p.pop("Type")
    except KeyError:
        print "ERROR: Must specify 'Type' key in config for %s.  Valid types are:" % file
        print "   Fit  (requires keys: 'Bins')"
        print "   Scan (requires keys: 'Scans')"
        continue

    if Type == "Fit":
        h, res = Plots.fit (D,      pdf=pdf, fname=op(file), **p)
        if Pull:
            Plots.pull     (h, res, pdf=pdf, fname=op("pull-" + file) )
    elif Type == "Scan":
        Plots.mass_scan    (Scans,  pdf=pdf, fname=op(file), Units=XSecUnits, **p)
    elif Type == "Estimators":
        Plots.estimators   (D,      pdf=pdf, fname=op(file), **p)
    elif Type == "Moments":
        Plots.moments      (D,      pdf=pdf, fname=op(file), **p)

pdf.close()

