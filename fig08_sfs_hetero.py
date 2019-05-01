'''
README!
In order to replicate the figure from the paper,
the number of simulation iterations must be set
to 10,000 using the "nIter" variable.

This script uses only 50 iterations so that
the script finishes executing relatively quickly.
'''


import numpy as np
from matplotlib import pyplot,rc
from spm1d import rft1d  #www.spm1d.org
import nonuniform1d      #(in this repository)
import myplot            #(in this repository)
rc('font',**{'family':'DejaVu Sans', 'serif':['Times']})
rc('text', usetex=True)




#(0) Simulate heterogeneous 1D data:
J,Q    = 12, 101
w0,w1  = 2, 25
nIter  = 500    #change this to 10000 to replicate the results from the paper
sqrtN  = J**0.5
amps   = [0.1, 0.5, 2.0]
T,W    = [],[]
for i,amp in enumerate(amps):
	np.random.seed(10+i)
	t,w   = [],[]
	for i in range(nIter):
		y0    = rft1d.randn1d(J, Q, w0)  #high frequency noise
		y1    = rft1d.randn1d(J, Q, w1)  #low frequency noise
		y     = amp * y0 + y1            #heterogeneous noise
		tt    = y.mean(axis=0) / y.std(ddof=1, axis=0) * sqrtN
		t.append( tt.max() )
		w.append( rft1d.geom.estimate_fwhm(y) )
	T.append(t)
	W.append(w)




#(1) Simulate heterogeneous + nonuniform 1D data:
s0,s1  = 5, 30
w0     = s1
w1     = nonuniform1d.generate_fwhm_continuum('step', Q, s0, s1)
np.random.seed(13)
t,w   = [],[]
for i in range(nIter):
	y0    = rft1d.randn1d(J, Q, w0)  #high frequency noise
	y1    = nonuniform1d.randn1dnu(J, w1)   #nonuniform noise with a different freuqncy
	y     = amp * y0 + y1            #heterogeneous noise
	tt    = y.mean(axis=0) / y.std(ddof=1, axis=0) * sqrtN
	t.append( tt.max() )
	w.append( rft1d.geom.estimate_fwhm(y) )
T.append(t)
W.append(w)
T,W    = np.asarray(T), np.asarray(W)




#(2) Compute survival functions:
heights     = np.linspace(2, 5, 21)
df          = J-1
SF,SFE,FPR  = [], [], []
for t,w in zip(T,W):
	wmean   = w.mean()
	sf      = np.array(  [ (t>u).mean()  for u in heights]  )
	sfe     = rft1d.t.sf(heights, df, Q, wmean)  #theoretical SF
	tc      = rft1d.t.isf(0.05, df, Q, wmean) #critical value
	fpr     = (t>tc).mean()   #false positive rate
	SF.append( sf )
	SFE.append( sfe )
	FPR.append( fpr )
SF,SFE,FPR  = np.array(SF), np.array(SFE), np.array(FPR)





#(3) Plot:
# pyplot.close('all')
fontname = 'Times New Roman'
fig      = myplot.MyFigure(figsize=(6.5,1.8), axx=np.linspace(0.080,0.78,4), axy=[0.18], axw=0.21, axh=0.7, fontname=fontname, set_font=True, set_visible=True)
fig.set_window_title('Figure 8') 
AX       = fig.AX.flatten()
ax0,ax1,ax2,ax3 = AX
[ax.axhline(0.05, color='0.7', ls='--')  for ax in AX]

for ax,sf,sfe,fpr in zip(AX,SF,SFE,FPR):
	ax.plot(heights[::2], sf[::2],   'ko', ms=3, label=r'Simulated')
	ax.plot(heights, sfe,  'k-', lw=0.5, label=r'Theoretical')
	ax.text(3.1, 0.40,r'FPR$_\textrm{sim}$ = %.3f' %fpr, size=9)
	ax.set_ylim(0, 0.6)

pyplot.setp(AX, xticks=[2,3,4,5])
leg = ax0.legend(loc='lower left', bbox_to_anchor=(0.2,0.20), frameon=False)
pyplot.setp(leg.get_texts(), size=8)

[ax.text(0.5, -0.24, r'$u$', size=12, transform=ax.transAxes)  for ax in AX]
ax0.set_ylabel(r'$P (t_\mathrm{max} > u)$', size=12, ha='center')

labels   = 'Amp = 0.1', 'Amp = 0.5', 'Amp = 2.0', 'Hetero + nonuniform'
[ax.text(-0.02, 1.05, '(%s) %s'%(chr(97+i), label), size=10, transform=ax.transAxes, ha='left')  for i,(ax,label) in enumerate(zip(AX,labels))]

pyplot.show()





