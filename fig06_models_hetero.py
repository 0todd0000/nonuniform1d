
import numpy as np
from matplotlib import pyplot,rc
from spm1d import rft1d  #www.spm1d.org
import nonuniform1d      #(in this repository)
import myplot            #(in this repository)
rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)



#(0) Generate random data:
J,Q   = 5, 101
w0,w1 = 2, 25
np.random.seed(0)
yA    = rft1d.randn1d(J, Q, w0, pad=True)
yB    = rft1d.randn1d(J, Q, w1, pad=True)
y0    = yB + 0.1 * yA
y1    = yB + 0.5 * yA
y2    = yB + 2.0 * yA
### hetero + nonuniform
np.random.seed(11)
s0,s1 = 10, 30
w     = nonuniform1d.generate_fwhm_continuum('step', Q, s0, s1)
yC    = rft1d.randn1d(J, Q, w0, pad=True)
yD    = nonuniform1d.randn1dnu(J, w)
y3    = 0.1*yC + yD



### plot:
# pyplot.close('all')
fontname = 'Times New Roman'
fig      = myplot.MyFigure(figsize=(6.5,1.8), axx=np.linspace(0.070,0.775,4), axy=[0.22], axw=0.21, axh=0.67, fontname=fontname, set_font=True, set_visible=True)
fig.set_window_title('Figure 6') 
AX       = fig.AX.flatten()
ax0,ax1,ax2,ax3 = AX
[ax.axhline(0, color='0.7', ls='--')  for ax in AX]
for ax,y in zip(AX,[y0,y1,y2,y3]):
	ax.plot(y.T, 'k-', lw=0.5)
### datum:
[ax.axhline(0, color='k', ls='-')  for ax in AX]
### axis limits:
pyplot.setp(AX, xlim=(0,100))
### axis labels:
[ax.set_xlabel(r'Position (\%)', size=11, name=fontname)  for ax in AX]
ax0.set_ylabel('Dependent variable', size=11, name=fontname)
### panel labels:
labels  = 'Amp = 0.1', 'Amp = 0.5', 'Amp = 2.0', 'Hetero + nonuniform'
[ax.text(-0.05, 1.05, '(%s) %s' %(chr(97+i),label), transform=ax.transAxes, name=fontname, size=11)  for i,(ax,label) in enumerate(zip(AX,labels))]
pyplot.show()





