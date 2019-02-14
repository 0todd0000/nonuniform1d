
import numpy as np
from matplotlib import pyplot
import spm1d         #www.spm1d.org
import nonuniform1d  #(in this repository)
import myplot        #(in this repository)




#(0) Stationary mean, uniform smoothness:
np.random.seed(0)
J,Q   = 12, 101
y0    = spm1d.rft1d.randn1d(J, Q, 30, pad=True)


#(1) Stationary mean, nonuniform smoothness:
np.random.seed(81)
w1    = nonuniform1d.generate_fwhm_continuum('step', Q, 3, 50)
y1    = nonuniform1d.randn1dnu(J, w1)


#(2) Nonstationary mean, uniform smoothness:
y           = spm1d.rft1d.randn1d(J, Q, 10) 
drift       = 0.2 * np.exp( np.linspace(0, 3, Q) )
y2          = y + drift


#(3) Heterogeneous smoothness:
w3          = [3]*6 + [40]*6
y3          = np.array([spm1d.rft1d.randn1d(1, Q, w)  for w in w3])




#(4) Plot:
# pyplot.close('all')
fontname = 'Times New Roman'
lcolor   = '0.7'
fig      = myplot.MyFigure(figsize=(7,5), axx=[0.07,0.54], axy=[0.56,0.09], axw=0.45, axh=0.43, fontname=fontname, set_font=True, set_visible=False)
fig.set_window_title('Figure 2') 
AX       = fig.AX.flatten()
ax0,ax1,ax2,ax3 = AX
### plot:
h0 = ax0.plot(y0.T, lw=0.5, color=lcolor)[0]
h1 = ax0.plot(y0.mean(axis=0), 'k', lw=3)[0]
ax0.legend([h0,h1], ['Single observations', 'Sample mean'], prop=dict(family=fontname, size=10), loc='lower center')

ax1.plot(y1.T, lw=0.5, color=lcolor)
ax1.plot(y1.mean(axis=0), 'k', lw=3)

ax2.plot(y2.T, lw=0.5, color=lcolor)
ax2.plot(y2.mean(axis=0), 'k', lw=3)

ax3.plot(y3[:6].T, lw=0.5, color=lcolor)
ax3.plot(y3[6:].T, lw=0.5, color=lcolor)
ax3.plot(y3.mean(axis=0), 'k', lw=3)

pyplot.setp([ax0,ax1,ax2,ax3], ylim=(-4.9, 4.9))
### datum:
[ax.axhline(0, color='k', ls='--')  for ax in AX]
### annotate:
[ax.set_xticklabels([])  for ax in [ax0,ax1]]
[ax.set_yticklabels([])  for ax in [ax1,ax3]]
[ax.set_ylabel('DV value', size=11, name=fontname)  for ax in [ax0,ax2]]
labels = '(a)  Mean: Stationary\n       Smoothness: Uniform', '(b)  Mean: Stationary\n       Smoothness: Nonuniform', '(c)  Mean: Nonstationary\n       Smoothness: Uniform', '(d)  Mean: Stationary\n       Smoothness: Heterogeneous'
[ax.text(0.01, 0.86, label, transform=ax.transAxes, name=fontname, size=11)  for i,(ax,label) in enumerate(zip(AX,labels))]
[ax.set_xlabel('Continuum position  (%)', name=fontname, size=11)  for ax in [ax2,ax3]]

pyplot.show()







