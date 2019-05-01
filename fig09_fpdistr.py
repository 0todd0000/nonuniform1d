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
rc('font',**{'serif':['Times']})
rc('text', usetex=True)


def tstat(y):
	return y.mean(axis=0) / (  y.std(ddof=1, axis=0)/ (y.shape[0])**0.5  )



#(0) Simulate datasets (uniform and nonuniform)
np.random.seed(0)
J,Q   = 8, 101
w0    = 20 * np.ones(Q)
w1    = nonuniform1d.generate_fwhm_continuum('step', Q, 10, 30)
nIter = 50
T,W   = [],[]
for w in [w0,w1]:
	t,we  = [],[]
	for i in range(nIter):
		y  = nonuniform1d.randn1dnu(J, w)
		t.append( tstat(y) )
		we.append( rft1d.geom.estimate_fwhm(y) )
	T.append(t)
	W.append(we)
T,W   = np.array(T), np.array(W)




#(1) Compute false positive distributions:
t0,t1     = T
w0,w1     = W
### crtical thresholds
tstar0    = rft1d.t.isf(0.05, J-1, Q, w0.mean())
tstar1    = rft1d.t.isf(0.05, J-1, Q, w1.mean())
### false positive binary continua:
fp0       = t0 > tstar0
fp1       = t1 > tstar1
### overall false positive rate:
fpr0      = np.any(fp0, axis=1).mean()
fpr1      = np.any(fp1, axis=1).mean()
### tmax distribution:
m0,m1     = t0.max(axis=1), t1.max(axis=1)
q0,q1     = t0.argmax(axis=1), t1.argmax(axis=1)
i0,i1     = m0 > tstar0, m1 > tstar1
q0,q1     = q0[i0], q1[i1]
countsmx0 = 100 * np.array([  ((q0>10*i)&(q0<=10*i+9)).sum()  for i in range(10)  ]) / nIter
countsmx1 = 100 * np.array([  ((q1>10*i)&(q1<=10*i+9)).sum()  for i in range(10)  ]) / nIter
### FP distribution (all continuum nodes):
n0        = ( t0 > tstar0 ).sum(axis=0)
n1        = ( t1 > tstar1 ).sum(axis=0)
counts0   = 100 * np.array([n0[10*i:10*i+10].sum()  for i in range(10)]) / nIter
counts1   = 100 * np.array([n1[10*i:10*i+10].sum()  for i in range(10)]) / nIter





#(2) Example single datasets:
w0       = 25
w1       = nonuniform1d.generate_fwhm_continuum('step', Q, 10, 30)
np.random.seed(7)
y0       = rft1d.randn1d(J, Q, w0, pad=True)
y0       = np.fliplr(y0)
np.random.seed(122)
y1       = nonuniform1d.randn1dnu(J, w1)
scale    = nonuniform1d.generate_fwhm_continuum('step', Q, 1.5, 0.8)
y1       = scale * y1
t0,t1    = tstat(y0), tstat(y1)







#(3) Plot:
# pyplot.close('all')
fontname = 'Times New Roman'
stmax    = r'$t_{\textrm{max}}$'
fig      = myplot.MyFigure(figsize=(5.5,8), axx=[0.105,0.56], axy=np.linspace(0.77,0.07,4), axw=0.43, axh=0.2, fontname=fontname, set_font=True, set_visible=True)
fig.set_window_title('Figure 9') 
AX       = fig.AX
ax0,ax1  = AX[0]
ax2,ax3  = AX[1]
ax4,ax5  = AX[2]
ax6,ax7  = AX[3]

### plot example random data:
for ax,y in zip([ax0,ax1], [y0,y1]):
	ax.axhline(0, color='k', lw=0.5, ls='--')
	ax.plot(y.T, '0.5', lw=0.5)
	ax.plot(y.mean(axis=0), 'k-', lw=3, label='Mean')
	ax.set_ylim(-3.5, 3.5)


### plot t stat:
for ax,t,tstar,txx in zip([ax2,ax3], [t0,t1], [tstar0,tstar1],[40,80]):
	ax.axhline(0, color='k', lw=0.5, ls='--')
	ax.plot(t, 'k-', lw=3, label='$t$')
	ax.plot(t.argmax(), t.max(), 'ko', ms=6, mfc='w')
	ax.axhline(tstar, color='k', lw=2, ls='--')
	ax.set_ylim(-5, 6)
	ax.text(t.argmax()+2, t.max()+0.3, stmax)
	# ax.text(50, tstar+0.3, r'$t^*$  $(\alpha=0.05)$')
	ax.text(txx, tstar+0.5, r'$p < 0.05$', size=9)
	ax.text(txx, tstar-1.1, r'$p > 0.05$', size=9)


### plot maxima location distributions:
fc,ec  = '0.7', '0.6'
x  = np.arange(5, 96, 10)
for ax,counts in zip([ax4,ax5], [countsmx0, countsmx1]):
	ax.bar(x, counts, width=10, facecolor=fc, edgecolor=ec)
	ax.set_ylim(0, 1)
	ax.set_yticks(np.linspace(0, 1, 5))
	if ax==ax4:
		ax.set_yticklabels([0, '', '', '', 1])
	# ax.text(0.35, 0.9, 'False positive rate = %.1f%s'%(100*fpr, '\%'), size=10, transform=ax.transAxes)

### plot all suprathreshold node location distributions:
x  = np.arange(5, 96, 10)
for ax,counts in zip([ax6,ax7], [counts0, counts1]):
	ax.bar(x, counts, width=10, facecolor=fc, edgecolor=ec)
	ax.set_ylim(0, 4)


### legends:
ax = AX[0,0]
leg = ax.legend(loc='lower left', bbox_to_anchor=(0.03,-0.01), frameon=False)
pyplot.setp(leg.get_texts(), size=8)

### x axis labels:
[ax.set_xlabel(r'Time (\%)', size=12)  for ax in AX[-1]]

### y axis labels:
labels = 'Dependent variable', 't value', 'False positive frequency\n(\% simulations)', 'False positive frequency\n(\% simulations)'
[ax.set_ylabel(label, size=12)  for ax,label in zip(AX[:,0],labels)]

### column labels:
labels = 'Uniform', 'Nonuniform   (Model=Step)'
[ax.text(0.5, 1.05, label, size=12, transform=ax.transAxes, ha='center')  for ax,label in zip(AX[0],labels)]

### panel labels:
for i,ax in enumerate(AX.flatten()):
	ax.text(0.05, 0.9, '(%s)'%chr(97+i), transform=ax.transAxes, size=12)

### clarification labels:
labels  = 'Only %s'%stmax, 'All time nodes'
for ax,label in zip([ax5,ax7],labels):
	ax.text(-0.05, 0.75, label, transform=ax.transAxes, ha='center', bbox=dict(facecolor='0.9', alpha=0.9))



pyplot.show()




