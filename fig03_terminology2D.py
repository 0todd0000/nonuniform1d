
from math import pi
import numpy as np
from scipy import signal
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot,cm
import spm1d         #www.spm1d.org
import nonuniform1d  #(in this repository)




def fn_mypulse2D(x, y, F, lamx, lamy):
	return (float(F)/(lamx*lamy)) * (1+ np.cos(2*pi/lamx*x)) * (1+ np.cos(2*pi/lamy*y))


def gen_mypulse2D(F, lamx, lamy, dt=0.1):
	tx,ty   = np.arange(-lamx/2, lamx/2+dt, dt), np.arange(-lamy/2, lamy/2+dt, dt)
	X,Y     = np.meshgrid(tx,ty)
	Z       = fn_mypulse2D(X, Y, F, lamx, lamy)
	return X,Y,Z    #N/mm2



#(0) Isotropic 2D data:
np.random.seed(0)
m,n         = 128,128
I           = np.random.randn(m,n)
lam0x,lam0y = 35,35
dt          = 1.0
Z0          = gen_mypulse2D(15, lam0x, lam0y, dt)[-1]
I0          = 1.2*signal.convolve2d(I, Z0, boundary='symm', mode='same')



#(1) Nonisotropic 2D data:
np.random.seed(2)
I1          = np.random.randn(m,n)
np.random.seed(1)
I2          = np.random.randn(m,n)
lam1x,lam1y = 80,10
lam2x,lam2y = 10,80
dt          = 1.0
Z1          = gen_mypulse2D(15, lam1x, lam1y, dt)[-1]
Z2          = gen_mypulse2D(15, lam2x, lam2y, dt)[-1]
I1          = signal.convolve2d(I1, Z1, boundary='symm', mode='same')
I2          = signal.convolve2d(I2, Z2, boundary='symm', mode='same')



#(2) Plot:
# pyplot.close('all')
fontname    = 'Times New Roman'
vmin,vmax   = -2, 2
### create figure and axes:
axx         = [0.069,0.40]
axy         = np.linspace(0.71,0.07,3)
axw         = [0.25, 0.55]
axh         = [0.25, 0.3]
fig         = pyplot.figure(figsize=(7,7))
fig.canvas.set_window_title('Figure 3') 
ax1         = [pyplot.axes([axx[1],yy,axw[1],axh[1]], projection='3d')  for yy in axy-0.04]
ax0         = [pyplot.axes([axx[0],yy,axw[0],axh[0]])  for yy in axy]
AX          = np.array([ax0,ax1]).T
### set fonts and sizes:
[pyplot.setp(ax.get_xticklabels()+ax.get_yticklabels(), name=fontname, size=8)  for ax in AX[:,0]]
[pyplot.setp(ax.get_xticklabels()+ax.get_yticklabels()+ax.get_zticklabels(), name=fontname, size=8)  for ax in AX[:,1]]
### plot images:
ticks       = [0, 32, 64, 96, 128]
ticklabels  = ['0', '', '0.5', '', '1']
for ax,I in zip(AX[:,0],[I0,I1,I2]):
	ax.imshow(I, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
	ax.set_xticks(ticks)
	ax.set_yticks(ticks)
	ax.set_xticklabels(ticklabels)
	ax.set_yticklabels(ticklabels)
	ax.set_xlabel('X', name=fontname, size=14)
	ax.set_ylabel('Y', name=fontname, size=14)
cbs       = [pyplot.colorbar(cax=pyplot.axes([0.33,yy,0.025,axh[0]]), mappable=AX[0,0].images[0])  for yy in axy]
[pyplot.setp(cb.ax.get_yticklabels(), name=fontname, size=8)  for cb in cbs]
[cb.ax.set_ylabel('DV value', name=fontname, size=14)  for cb in cbs]



### plot surfaces:
X    = np.linspace(0, 1, m)
Y    = np.linspace(0, 1, n)
X, Y = np.meshgrid(Y, X)
ticks       = [0, 0.25, 0.5, 0.75, 1]
ticklabels  = ['0', '', '0.5', '', '1']

for ax,I in zip(AX[:,1],[I0,I1,I2]):
	surf = ax.plot_surface(X, Y, I, rstride=3, cstride=3, cmap=cm.gray_r, linewidth=0.2, edgecolor='0.7', antialiased=True)
	pyplot.setp(ax, xticks=ticks, yticks=ticks, xticklabels=ticklabels, yticklabels=ticklabels)
	pyplot.setp(ax, xlim=(0,1), ylim=(0,1), zlim=(-15,15))
	ax.set_xlabel('X', name=fontname, size=14)
	ax.set_ylabel('Y', name=fontname, size=14)
	ax.set_zlabel('DV value', name=fontname, size=14)



### add panel labels:
labels   = 'Isotropic', 'Nonisotriopic  (X smoother)', 'Nonisotriopic  (Y smoother)'
yloc     = [1.14, 1.00, 1.00]
for i,(ax,label,yy) in enumerate(zip(AX[:,0], labels, yloc)):
	ax.text(1.32, yy, '(%s)  %s' %(chr(97+i), label), name=fontname, size=14, transform=ax.transAxes, va='top', bbox=dict(color='w', alpha=0.5))


### annotate:
yloc     = [0.65, 0.33]
for yy in yloc:
	AX[0,0].annotate("", xy=(0, yy), xycoords='figure fraction', xytext=(1, yy), textcoords='figure fraction', arrowprops=dict(arrowstyle="-", color='0.7') )


pyplot.show()







