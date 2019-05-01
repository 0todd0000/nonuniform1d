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




#(0) Create nonuniform randomness models:
np.random.seed(123456789)
J,Q    = 12, 101
nIter  = 50    #change this to 10000 to replicate the results from the paper
s0,s1  = 10, 25
w0     = nonuniform1d.generate_fwhm_continuum('linear', Q, s0, s1)
w1     = nonuniform1d.generate_fwhm_continuum('exponential', Q, s0, s1)
w2     = nonuniform1d.generate_fwhm_continuum('gaussian', Q, 50, 14, s0, s1)
w3     = nonuniform1d.generate_fwhm_continuum('step', Q, s0, s1)
w4     = nonuniform1d.generate_fwhm_continuum('double_step', Q, s0, 20, 45)




#(1) Simulate random data for each model
# save maximum test stat and smoothness estimate for each iteration
models = [w0,w1,w2,w3,w4]
sqrtN  = J**0.5
T,W    = [],[]
for model in models:
	t,w   = [],[]
	for i in range(nIter):
		y  = nonuniform1d.randn1dnu(J, model)
		tt = y.mean(axis=0) / y.std(ddof=1, axis=0) * sqrtN
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
fig      = myplot.MyFigure(figsize=(8,1.8), axx=np.linspace(0.065,0.815,5), axy=[0.18], axw=0.17, axh=0.7, fontname=fontname, set_font=True, set_visible=True)
fig.set_window_title('Figure 7') 
AX       = fig.AX.flatten()
ax0,ax1,ax2,ax3,ax4 = AX
[ax.axhline(0.05, color='0.7', ls='--')  for ax in AX]
for ax,sf,sfe,fpr in zip(AX,SF,SFE,FPR):
	ax.plot(heights[::2], sf[::2],   'ko', ms=3, label=r'Simulated')
	ax.plot(heights, sfe,  'k-', lw=0.5, label=r'Theoretical')
	ax.text(3.1, 0.15,r'FPR$_\textrm{sim}$ = %.3f' %fpr, size=9)

pyplot.setp(AX, xticks=[2,3,4,5])
leg      = ax0.legend(frameon=False)
pyplot.setp(leg.get_texts(), size=8)

[ax.text(0.5, -0.24, r'$u$', size=12, transform=ax.transAxes)  for ax in AX]
ax0.set_ylabel(r'$P (t_\mathrm{max} > u)$', size=12, ha='center')

labels   = 'Linear', 'Exponential', 'Gaussian', 'Step', 'Double step'
[ax.text(0.5, 1.05, '(%s) %s'%(chr(97+i), label), size=10, transform=ax.transAxes, ha='center')  for i,(ax,label) in enumerate(zip(AX,labels))]

for ax in AX:
	ax.text(2.0, 0.07, r'$p>0.05$', size=8, color='0.7')
	ax.text(2.0, 0.01, r'$p<0.05$', size=8, color='0.7')

pyplot.show()




