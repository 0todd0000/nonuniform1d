
import os
import numpy as np
from matplotlib import pyplot
import tables
import spm1d         #www.spm1d.org
import nonuniform1d  #(in this repository)
import myplot        #(in this repository)




#(0) Simulated nonuniform data:
np.random.seed(41)
J     = 12
Q     = 101
w0    = nonuniform1d.generate_fwhm_continuum('step', Q, 3, 50)
y0    = nonuniform1d.randn1dnu(J, w0)



#(1) Load Neptune data:
dir0        = os.path.dirname( __file__ )
fnameH5     = os.path.join(dir0, 'Neptune2009means.h5') 
with tables.open_file(fnameH5, mode='r') as fid:
	Y       = fid.get_node('/Y').read()
	SUBJ    = fid.get_node('/SUBJ').read()
	TASK    = fid.get_node('/TASK').read()
y1    = Y[TASK==0][:,:,8]
r1    = y1 - y1.mean(axis=0)
w1    = nonuniform1d.estimate_fwhm(r1, mean=False)



#(2) Plot:
# pyplot.close('all')
fontname = 'Times New Roman'
fig      = myplot.MyFigure(figsize=(8,5), axx=[0.06, 0.56], axy=[0.55, 0.09], axw=0.42, axh=0.43, fontname='Times New Roman', set_font=True, set_visible=False)
fig.set_window_title('Figure 1') 
AX       = fig.AX.flatten()
ax0,ax1,ax2,ax3  = AX
ax0.plot(y0.T, 'k', lw=0.5)
ax1.plot(y1.T, 'k', lw=0.5)
ax2.plot(w0, 'k', lw=3)
ax3.plot(w1.T, 'k', lw=3)
[ax.axhline(0, color='k', ls=':')  for ax in AX]
[ax.set_xticklabels([])  for ax in [ax0,ax1]]
pyplot.setp(ax1.get_yticklabels(), visible=True)
pyplot.setp(AX, xlim=(0,100))
pyplot.setp([ax2,ax3], ylim=(-5,90))
ax1.set_ylim(-100, 3400)
labels = '(a) Simulated', '(b) Neptune et al. (1999)', '(c) True smoothness', '(d) Estimated smoothness'
[ax.text(0.03, 0.9, label, name=fontname, transform=ax.transAxes)   for i,(ax,label) in enumerate(zip(AX,labels))]
labels = 'Continuum position  (%)', 'Time  (%)'
[ax.set_xlabel(label, name=fontname, size=12)  for ax,label in zip([ax2,ax3],labels)]
labels = 'Gaussian noise value', 'Ground reaction force  (N)', 'FWHM', 'FWHM'
[ax.set_ylabel(label, name=fontname, size=12)  for ax,label in zip(AX,labels)]
pyplot.show()


