

from math import sqrt,log
import numpy as np
from matplotlib import pyplot
eps         = np.finfo(np.float).eps


def gaussian_kernel(sd):
	'''
	Create a Gaussian kernel with the specified standard deviation (sd)
	
	Modified from scipy.ndimage.filters.gaussian_filter1d
	'''
	kw = int(4.0 * sd + 0.5)   #kernel width
	weights = [0.0] * (2 * kw + 1)
	weights[kw] = 1.0
	sum = 1.0
	sd = sd * sd
	# calculate the kernel:
	for ii in range(1, kw + 1):
		tmp = np.exp(-0.5 * float(ii * ii) / sd)
		weights[kw + ii] = tmp
		weights[kw - ii] = tmp
		sum += 2.0 * tmp
	for ii in range(2 * kw + 1):
		weights[ii] /= sum
	return np.array(weights)


def fwhm_exponential(Q, w0, w1):
	'''
	Exponential smoothness model
	
	Parameters:
	
	Q  : number of continuum nodes
	w0 : initial smoothness value
	w1 : final smoothness value
	'''
	x     = np.linspace(-2, 2, Q)
	fwhm  = np.exp(x)
	fwhm  = w0 + w1 * fwhm/fwhm[-1]
	return fwhm

def fwhm_gaussian(Q, q, sd, w0, w1):
	'''
	Gaussian pulse smoothness model
	
	Parameters:
	
	Q  : number of continuum nodes
	q  : location of Gaussian kernel center
	sd : standard deviation of the kernel
	w0 : baseline smoothness value
	w1 : maximum smoothness value
	'''
	y    = np.zeros(Q)
	g    = gaussian_kernel(sd)
	n    = g.size
	i0   = q - int(n/2)
	i1   = q + int(n/2) + 1
	if i0 < 0:
		n2crop = abs(i0)
		i0 = 0
		g  = g[n2crop:]
	if i1 > Q:
		n2crop = i1 - Q
		i1     = Q
		g      = g[:-n2crop]
	y[i0:i1] = g
	amp  = w1 - w0
	fwhm = w0 + y * amp / y.max()
	return fwhm


def fwhm_linear(Q, w0, w1, q0=None, q1=None):
	'''
	Linear smoothness model
	
	Parameters:
	
	Q  : number of continuum nodes
	w0 : initial smoothness value
	w1 : final smoothness value
	q0 : optional starting node for linear increase
	q1 : optional ending node for linear increase
	'''
	q0    = 0 if (q0 is None) else q0
	q1    = Q if (q1 is None) else q1
	width = q1 - q0
	wstep = np.linspace(w0, w1, width)
	w     = w0 * np.ones(Q)
	w[q0:q1] = wstep
	w[q1:]   = w1
	return w

def fwhm_step(Q, w0, w1):
	'''
	Sigmoid step smoothness model
	
	Parameters:
	
	Q  : number of continuum nodes
	w0 : initial smoothness value
	w1 : final smoothness value
	'''
	dx    = 5
	x     = np.linspace(-5, 5, Q)
	fwhm  = w0 + (w1-w0) / (1 + np.exp(-dx*x))
	return fwhm

def fwhm_double_step(Q, w0, w1, w2):
	'''
	Double sigmoid step smoothness model
	
	Parameters:
	
	Q  : number of continuum nodes
	w0 : initial smoothness value
	w1 : intermediary smoothness value
	w2 : final smoothness value
	'''
	dx    = 5
	n     = int(Q/2)
	n0,n1 = (n+1,n) if Q%2 else (n,n)
	x0    = np.linspace(-5, +2.5, n0)
	x1    = np.linspace(-2.4, +5, n1)
	fwhm0 = w0 + (w1-w0) / (1 + np.exp(-dx*x0))
	fwhm1 = w1 + (w2-w1) / (1 + np.exp(-dx*x1))
	fwhm  = np.hstack([fwhm0,fwhm1])
	return fwhm



def generate_fwhm_continuum(type='linear', *args):
	'''
	Generate a 1D FWHM continuum as a model of underlying data smoothness
	
	Parameters:
	
	type : one of ["linear", "exponential", "gaussian", "step", "double step"]
	args : model-dependent arguments;  see documentation from fwhm* functions
	'''
	if type=='double_step':
		fn   = fwhm_double_step
	if type=='linear':
		fn   = fwhm_linear
	elif type=='exponential':
		fn   = fwhm_exponential
	elif type=='step':
		fn   = fwhm_step
	elif type=='gaussian':
		fn   = fwhm_gaussian
	return fn(*args)




def estimate_fwhm(R, mean=True):
	resels = estimate_resels(R, mean=False)
	fwhm   = 1 / resels
	if mean:
		fwhm = fwhm.mean()
	return fwhm


def estimate_resels(R, mean=True):
	'''
	Estimate local continuum smoothness.
	
	NOTE:  Only use this function to approximate local smoothness!
	Robust smoothness estimation must be done only at the continuum level,
	as implemented in **spm1d.rft1d.geom.estimate_fwhm*** 
	
	This code is adapted from **spm1d.rft1d.geom.estimate_fwhm**
	'''
	ssq    = (R**2).sum(axis=0)
	### gradient estimation (Method 2)
	dy,dx  = np.gradient(R)
	v      = (dx**2).sum(axis=0)
	# normalize:
	v     /= (ssq + eps)
	# ignore zero-variance nodes:
	i      = np.isnan(v)
	v      = v[np.logical_not(i)]
	# global resels estimate:
	resels = np.sqrt(v / (4*log(2)))
	if mean:
		resels = resels.mean()
	return resels



def randn1dnu(J, FWHM):
	'''
	Nonuniformly smooth 1D Gaussian random continuum generator
	
	Parameters:
	
	J    : sample size (integer)
	FWHM : one-dimensional NumPy array representing continuum smoothness
	
	Outputs:
	
	y    : a random sample of J continua, each with length FWHM.size
	'''
	Q     = FWHM.size
	z     = np.random.randn(Q, J)
	s     = FWHM / (  (Q-1) * sqrt( 4*log(2) )  )
	dx    = 1. / (Q -1)
	x     = np.array([dx * np.arange(Q)])
	X     = np.repeat(x, Q, 0) 
	D     =  X - np.repeat(x.T, Q, 1)  #;   %distance matrix (relative to diagonal nodes)
	A     = np.exp(-0.5*D**2 /  (s**2) )
	
	[U,V]  =  np.linalg.eig(A.T)
	U,V    = np.real(U), np.real(V)
	U[U<eps] = 0
	U,V    = np.matrix(np.diag( np.sqrt(U) )), np.matrix(V)
	C      = V * U * V.T
	y      = (C * z).T
	return np.asarray(y)