#!/usr/bin/python

import numpy.random
import numpy             as np
import matplotlib.pyplot as plt

from   Tools.OrthExp     import ExpDecompFn

end    = 10        # Maximum value to evaluate
Npt    = int(1e6)  # Number of points to use
Nbasis = 101       # Number of basis elements to evaluate

toplot = {   1 : 'green',
            50 : 'orange',
             2 : 'black',
            10 : 'red',
             0 : 'blue',
         }

np.set_printoptions(precision=3, linewidth=160)


x      = np.linspace(0.001, end, Npt)
w      = np.ones((Npt,)) / Npt

Decomp = ExpDecompFn( x=x, w=w, Nbasis=Nbasis, Lambda=1, x0=0, Alpha=1.0 )

######
for D in Decomp:
    if D.N in toplot:
        plt.plot(x, Decomp.Values(), lw=0.5, ls='--', color=toplot[D.N], zorder=-D.N, label='$E_{%d}$' % D.N)
    print D.Values()
    print "%d: %f" % (D.N, Decomp.Moment())

plt.xlim( 0, end)
#plt.xlim( 1e-3, end)
#plt.xscale('log')
plt.ylim(-1, 1)
plt.legend()
plt.show()

