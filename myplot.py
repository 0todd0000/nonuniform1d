
import numpy as np
from matplotlib import pyplot



class MyFigure(object):
	def __init__(self, figsize=None, axx=None, axy=None, axw=None, axh=None, fontname=u'Times New Roman', set_font=True, set_visible=False, projections=None):
		self.AX       = None
		self.axx      = axx
		self.axy      = axy
		self.axw      = axw
		self.axh      = axh
		self.fig      = None
		self.figsize  = figsize
		self.fontname = u'%s' %fontname
		self.nRows    = len(axy)
		self.nCols    = len(axx)
		self.projections = projections
		self._create()
		self._set_default(set_font, set_visible)
	def _create(self):
		self.fig      = pyplot.figure(figsize=self.figsize)
		AX            = []
		n             = self.nRows * self.nCols
		projections   = ([None] * n) if (self.projections is None) else self.projections
		i             = 0
		for yy in self.axy:
			ax        = []
			for xx in self.axx:
				proj  = projections[i]
				ax.append( pyplot.axes([xx,yy,self.axw,self.axh], projection=proj) )
				i    += 1
			AX.append(ax)
		self.AX       = np.array(AX)
		
	def _set_default(self, set_font, set_visible):
		if set_font:
			self.set_ticklabel_props(name=self.fontname, size=8)
		if set_visible:
			self.set_xticklabels_off()
			self.set_yticklabels_off()

	def get_axes(self):
		return self.AX.flatten().tolist()

	def plot_datum_lines(self, datum=0, **kwdargs):
		for ax in self.get_axes():
			x0,x1  = ax.get_xlim()
			ax.plot([x0,x1], [datum]*2, **kwdargs)

	def set_column_labels(self, labels=[], pos=(0.5,1.05), size=16):
		tx      = []
		for ax,label in zip( self.AX[0], labels ):
			tx.append(  ax.text(pos[0], pos[1], label, transform=ax.transAxes)  )
		pyplot.setp(tx, size=size, ha='center', name=self.fontname)
		return tx

	def set_panel_labels(self, labels=[], pos=(0.05,0.92), size=10, add_letters=True, with_box=True):
		tx      = []
		for i,(ax,label) in enumerate( zip(self.get_axes(), labels) ):
			s   = '(%s)  %s' %(chr(97+i),label) if add_letters else label
			tx.append(  ax.text(pos[0], pos[1], s, transform=ax.transAxes)  )
		pyplot.setp(tx, size=size, name=self.fontname)
		if with_box:
			pyplot.setp(tx, bbox=dict(facecolor='w'))
		return tx

	def set_xticklabels_off(self):
		pyplot.setp(self.AX[:-1], xticklabels=[])

	def set_yticklabels_off(self):
		pyplot.setp(self.AX[:,1:], yticklabels=[])

	def set_ticklabel_props(self, name=None, size=9):
		if name==None:
			name = self.fontname
		[pyplot.setp(ax.get_xticklabels()+ax.get_yticklabels(), name=u'%s'%name, size=size)  for ax in self.get_axes()]

	def set_xlabels(self, labels, size=20):
		if isinstance(labels, str):
			labels = [labels]*self.nCols
		for ax,label in zip(self.AX[-1], labels):
			ax.set_xlabel(label, name=self.fontname, size=size)

	def set_ylabels(self, labels, size=20):
		if isinstance(labels, str):
			labels = [labels]*self.nRows
		for ax,label in zip(self.AX[:,0], labels):
			ax.set_ylabel(label, name=self.fontname, size=size)


	def set_window_title(self, s):
		self.fig.canvas.set_window_title(s) 

