'''
README!
In order to replicate the figure from the paper,
the number of simulation iterations must be set
to 10,000 using the "nIter" variable.

This script uses only 500 iterations so that
the script finishes executing relatively quickly.
'''


import numpy as np
from matplotlib import pyplot,rc
from spm1d import rft1d  #www.spm1d.org
import nonuniform1d      #(in this repository)
import myplot            #(in this repository)
rc('font',**{'family':'DejaVu Sans', 'serif':['Times']})
rc('text', usetex=True)

eps         = np.finfo(float).eps   #smallest float




def get_cluster_extents_locations(T,tstar):
	kmax    = []
	kall    = []
	qall    = []
	calc    = rft1d.geom.ClusterMetricCalculator()
	interp  = True
	wrap    = False
	for t in T:
		k,q = calc.cluster_extents_locations(t, tstar, interp, wrap)
		if k!=[np.nan]:
			kmax.append( max(k) )
			kall += k
			qall += q
	return np.array(kmax), np.array(kall), np.array(qall)





#(0) Set parameters:
J,Q          = 12, 101
w0,w1        = 5, 30
wmean        = 0.5 * (w1+w0)
w            = nonuniform1d.generate_fwhm_continuum('step', Q, w0, w1)
### derived parameters:
df           = J-1
sqrtN        = np.sqrt(J)
### random number generators:
randf0       = lambda : rft1d.randn1d(J, Q, wmean)
randf1       = lambda : nonuniform1d.randn1dnu(J, w)




#(1) Simulate, save t continua:
nIter        = 500
T,W          = [],[]
for randf in [randf0,randf1]:
	t,we     = [],[]
	for i in range(nIter):
		y    = randf()
		tt   = y.mean(axis=0) / y.std(ddof=1, axis=0) * sqrtN
		t.append( tt )
		we.append( rft1d.geom.estimate_fwhm(y) )
	T.append(t)
	W.append(we)
T,W          = np.asarray(T), np.asarray(W)





#(1) Cluster extents (theoretical)
w0,w1        = W.mean(axis=1)
extents      = np.linspace(eps, 12, 21)
rftcalc0     = rft1d.prob.RFTCalculator(STAT='T', df=(1,df), nodes=Q, FWHM=w0)
rftcalc1     = rft1d.prob.RFTCalculator(STAT='T', df=(1,df), nodes=Q, FWHM=w1)
tcrit0       = rftcalc0.isf(0.05)
tcrit1       = rftcalc1.isf(0.05)
P0           = np.array([rftcalc0.p.cluster(k, tcrit0)  for k in extents/w0])
P1           = np.array([rftcalc1.p.cluster(k, tcrit1)  for k in extents/w1])





#(2) Cluster extents (simulation)
T0,T1        = T
kmax0,kall0,qall0 = get_cluster_extents_locations(T0, tcrit0)
kmax1,kall1,qall1 = get_cluster_extents_locations(T1, tcrit1)
Psim0        = [(kmax0>k).sum()/nIter   for k in extents]
Psim1        = [(kmax1>k).sum()/nIter   for k in extents]
### false positive rates:
fpr0         = len(kmax0) / nIter
fpr1         = len(kmax1) / nIter






#(3) Plot:
# pyplot.close('all')
fontname = 'Times New Roman'
stmax    = r'$t_{\textrm{max}}$'
fig      = myplot.MyFigure(figsize=(5.5,6), axx=[0.115,0.56], axy=np.linspace(0.72,0.07,3), axw=0.43, axh=0.24, fontname=fontname, set_font=True, set_visible=False)
fig.set_window_title('Figure 10') 
AX       = fig.AX
ax0,ax1  = AX[0]
ax2,ax3  = AX[1]
ax4,ax5  = AX[2]


#cluster extent probabilities
for ax,p,psim in zip([ax0,ax1], [P0,P1],[Psim0,Psim1]):
	ax.plot(extents, p, '-', lw=3, color='0.7')
	ax.plot(extents, psim,  'o', color='k', ms=4)

	ax.set_ylim(0, 0.07)
	ax.axhline(0.05, ls='--', color='0.4')
	# ax.set_xlabel(r'$k$', size=11)
	ax.set_xlabel('Cluster extent (\%)', size=11)
	if ax==ax0:
		ax.legend(['Simulated', 'Theoretical'], fontsize=8)
		ax.set_ylabel('Probability', size=11)


#cluster extent distributions
fc,ec  = '0.7', '0.6'
for ax,k in zip([ax2,ax3],[kall0,kall1]):
	counts   = 100 * np.array([((k>ii) & (k<(ii+1))).sum()  for ii in range(21)]) / nIter
	ax.bar(np.arange(21), counts, width=1, facecolor=fc, edgecolor=ec)
	ax.set_ylim(0, 1.8)
	ax.set_xlabel('Cluster extent (\%)', size=11)
	if ax==ax2:
		ax.set_ylabel('Frequency\n(\% simulations)', size=11)


#cluster extent by continuum position
for ax,q,k in zip([ax4,ax5],[qall0,qall1],[kall0,kall1]):
	ax.plot(q, k, '.', color='0.5', ms=3)
	ax.set_xlabel('Time (\%)', size=11)
	ax.set_ylim([0, 30])
	if ax==ax4:
		ax.set_ylabel('Cluster extent (\%)', size=11)


### column labels:
labels = 'Uniform', 'Nonuniform   (Model=Step)'
[ax.text(0.5, 1.05, label, size=12, transform=ax.transAxes, ha='center')  for ax,label in zip(AX[0],labels)]
### panel labels:
for i,ax in enumerate(AX.flatten()):
	ax.text(0.05, 0.9, '(%s)'%chr(97+i), transform=ax.transAxes, size=12)
[ax.set_yticklabels([])  for ax in [ax1,ax3,ax5]]


pyplot.show()









