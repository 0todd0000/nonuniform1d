
import numpy as np
from matplotlib import pyplot,rc
from spm1d import rft1d   #www.spm1d.org
rc('font',**{'family':'DejaVu Sans', 'serif':['Times']})
rc('text', usetex=True)




#(0) Generate random data:
np.random.seed(0)
Q      = 101
W      = 20
u      = 0.6  #threshold
y      = rft1d.random.randn1d(1, Q, W, pad=True)



#(1) Plot:
# pyplot.close('all')
fig   = pyplot.figure(figsize=(6,4))
fig.canvas.set_window_title('Figure 4') 
color0      = '0.6'
### create axes and domain:
ax    = pyplot.axes([0.11,0.14,0.87,0.84])
q     = np.arange(Q)
u     = u * np.ones(Q)
### plot:
ax.plot(q, y, color='k', lw=2)
ax.fill_between(q, y, u, where=(y>=u), interpolate=True, facecolor=color0)
ax.plot(y.argmax(), y.max(), 'o', markersize=5, markerfacecolor='w', markeredgecolor=color0)
ax.hlines(0, 0, 100, color='k', linestyle='-', lw=0.5)
ax.hlines(u[0], 0, 100, color=color0, linestyle='--')
### labels:
ax.annotate('Suprathreshold cluster', xy=(37.5, 0.7), xytext=(50, 0.9), color='k', arrowprops=dict(facecolor=color0, shrink=0.01), size=12)
ax.text(83, 0.65, 'Threshold  $u$', color='0.0', size=12)
ax.text(33, y.max()+0.05, '$t_{\mathrm{max}}$', color='0.0', size=14)
ax.plot([25.7,38.4], [0.5]*2, '-', color=color0, lw=3)
ax.text(32, 0.37, 'Extent', color='k', ha='center', size=12)
### axes labels:
ax.set_xlabel('Time  (\%)', size=16)
ax.set_ylabel('$t$ value', size=16)
pyplot.setp(ax, xlim=(0,100), ylim=(-1.2,1.2))
pyplot.show()


