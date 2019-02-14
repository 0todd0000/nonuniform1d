
import numpy as np
from matplotlib import pyplot
import nonuniform1d  #(in this repository)
import myplot        #(in this repository)





#(0) Create models:
np.random.seed(84)
J,Q   = 12, 101
s0,s1 = 6.2, 67
### smoothness continua:
w0    = nonuniform1d.generate_fwhm_continuum('linear', Q, s0, s1)
w1    = nonuniform1d.generate_fwhm_continuum('exponential', Q, s0, s1)
w2    = nonuniform1d.generate_fwhm_continuum('gaussian', Q, 50, 14, s0, s1)
w3    = nonuniform1d.generate_fwhm_continuum('step', Q, s0, s1)
w4    = nonuniform1d.generate_fwhm_continuum('double_step', Q, s0, 20, 45)
### random data:
y0    = nonuniform1d.randn1dnu(J, w0)
y1    = nonuniform1d.randn1dnu(J, w1)
y2    = nonuniform1d.randn1dnu(J, w2)
y3    = nonuniform1d.randn1dnu(J, w3)
y4    = nonuniform1d.randn1dnu(J, w4)




#(1) Plot:
# pyplot.close('all')
fontname = 'Times New Roman'
fig      = myplot.MyFigure(figsize=(8,3), axx=np.linspace(0.065,0.815,5), axy=[0.59,0.14], axw=0.17, axh=0.4, fontname=fontname, set_font=True, set_visible=True)
fig.set_window_title('Figure 5') 
AX       = fig.AX
ax0,ax1,ax2,ax3,ax4 = AX[0]
ax5,ax6,ax7,ax8,ax9 = AX[1]
### plot:
for ax,w in zip(AX[0],[w0,w1,w2,w3,w4]):
	ax.plot(w, 'k', lw=2)
for ax,y in zip(AX[1],[y0,y1,y2,y3,y4]):
	ax.plot(y.T, 'k', lw=0.5)
### datum:
[ax.axhline(0, color='k', ls='--')  for ax in AX.flatten()]
### axis limits:
pyplot.setp(AX[0], xlim=(0,100), ylim=(-3,81))
pyplot.setp(AX[1], xlim=(0,100), ylim=(-5,5))
### axis labels:
[ax.set_xlabel('Time (%)', size=11, name=fontname)  for ax in AX[1]]
ax0.set_ylabel('FWHM', size=11, name=fontname)
ax5.set_ylabel('Dependent variable', size=11, name=fontname)
### panel labels:
labels  = 'Linear', 'Exponential', 'Gaussian', 'Sigmoid step', 'Double step'
[ax.text(0.03, 0.89, '(%s) %s' %(chr(97+i),label), transform=ax.transAxes, name=fontname, size=11)  for i,(ax,label) in enumerate(zip(AX[0],labels))]
pyplot.show()









