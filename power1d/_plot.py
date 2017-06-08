
'''
This module contains all the plotting functionality associated with **power1d**.

None of these functions are meant to be accessed by the user.  Users shoul
instead access plotting functions through **power1d** objects' "plot" methods.
For example:

>>>  baseline = power1d.baselines.Null(Q=101)
>>>  baseline.plot()

All **power1d** plotting functionality is implemented using **matplotlib**.
If you prefer a different plotting library this is the only module that needs
to be modified. Simply substitute function and method content with appropriate
commands from another library.
'''

# Copyright (C) 2017  Todd Pataky
# version: 0.1 (2017/04/01)




import numpy as np
import matplotlib
from matplotlib import pyplot
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection





def _get_colors(n, cmap='jet'):
	cmap   = pyplot.get_cmap(cmap)
	return cmap( np.linspace(0, 1, n) )


def _get_q(Q, q):
	_msg = 'q must be a one-dimensional array with length Q'
	if q is None:
		q       = np.arange(Q)
	else:
		try:
			q   = np.asarray(q)
		except:
			raise ValueError(_msg)
		assert q.ndim==1, _msg
		assert q.size==Q, _msg
		dq      = np.diff(q)
		assert np.allclose(dq, dq[0]), 'q must be a monotonically increasing array with equal spacing between nodes'
	return q





def legend_manual(ax, colors=None, labels=None, linestyles=None, markerfacecolors=None, linewidths=None, **kwdargs):
	n      = len(colors)
	if linestyles is None:
		linestyles = ['-']*n
	if linewidths is None:
		linewidths = [1]*n
	if markerfacecolors is None:
		markerfacecolors = colors
	x0,x1  = ax.get_xlim()
	y0,y1  = ax.get_ylim()
	h      = [ax.plot([x1+1,x1+2,x1+3], [y1+1,y1+2,y1+3], ls, color=color, linewidth=lw, markerfacecolor=mfc)[0]   for color,ls,lw,mfc in zip(colors,linestyles,linewidths,markerfacecolors)]
	ax.set_xlim(x0, x1)
	ax.set_ylim(y0, y1)
	return ax.legend(h, labels, **kwdargs)



class DataPlotter(object):
	def __init__(self, ax=None):
		self.ax        = self._gca(ax)
		self.x         = None
		
	@staticmethod
	def _gca(ax):
		return pyplot.gca() if ax is None else ax
	
	def _set_axlim(self):
		self._set_xlim()
		self._set_ylim()
	
	def _set_x(self, x, y):
		Q        = y.shape[0]  #size if (y.ndim==1) else y.shape[1]
		self.x   =  _get_q(Q, x)

	def _set_xlim(self):
		pyplot.setp(self.ax, xlim=(self.x.min(), self.x.max())  )

	def _set_ylim(self, pad=0.075):
		def minmax(x):
			return np.ma.min(x), np.ma.max(x)
		ax          = self.ax
		ymin,ymax   = +1e10, -1e10
		for line in ax.lines:
			y0,y1   = minmax( line.get_data()[1] )
			ymin    = min(y0, ymin)
			ymax    = max(y1, ymax)
		for collection in ax.collections:
			datalim = collection.get_datalim(ax.transData)
			y0,y1   = minmax(  np.asarray(datalim)[:,1]  )
			ymin    = min(y0, ymin)
			ymax    = max(y1, ymax)
		for text in ax.texts:
			r       = matplotlib.backend_bases.RendererBase()
			bbox    = text.get_window_extent(r)
			y0,y1   = ax.transData.inverted().transform(bbox)[:,1]
			ymin    = min(y0, ymin)
			ymax    = max(y1, ymax)
		dy = 0.075*(ymax-ymin)
		ax.set_ylim(ymin-dy, ymax+dy)
	
	def plot(self, x, y, *args, **kwdargs):
		self._set_x(x, y)
		return self.ax.plot(self.x, y, *args, **kwdargs)

	def plot_datum(self, y=0, color='k', linestyle=':'):
		self.ax.axhline(y, color=color, lw=1, linestyle=linestyle)
		
	def plot_roi(self, x, roi, facecolor='b', edgecolor='w', alpha=0.25):
		self._set_x(x, roi.value)
		### determine y axis limits:
		ylim      = (-0.05,1.05) if (len(self.ax.lines) == 0) else self.ax.get_ylim()
		### get ROI labels:
		# y         = roi.value
		L,n       = roi._get_labels()
		### create ROI polygon patches:
		poly      = []
		dx        = self.x[1] - self.x[0]
		for i in range(n):
			b     = L==(i+1)
			x0,x1 = np.argwhere(b).flatten()[[0,-1]]
			x0,x1 = self.x[[x0,x1]] + [-0.5*dx, 0.5*dx]  #edges
			y0,y1 = ylim
			verts = [(x0,y0), (x1,y0), (x1,y1), (x0,y1)]
			poly.append( Polygon(verts) )
		pyplot.setp(poly, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha)
		self.ax.add_collection( PatchCollection(poly, match_original=True) )
		self._set_xlim()
		
		
	def set_ax_prop(self, *args, **kwdargs):
		pyplot.setp(self.ax, *args, **kwdargs)







def _plot_results_one_datamodel(results, q=None):
	q     = _get_q(results.Q, q)
	### create axes:
	axx   = [0.09, 0.58]
	axy   = [0.58, 0.08]
	axw   = 0.41
	axh   = 0.41
	ax0   = pyplot.axes([axx[0], axy[0], axw, axh])
	ax2   = pyplot.axes([axx[0], axy[1], axw, axh])
	ax3   = pyplot.axes([axx[1], axy[1], axw, axh])
	AX    = [ax0,ax2,ax3]
	### plot experiment models:
	c0,c1 = '0.5', 'k'
	results.model0.data_models[0].plot(ax0, color=c0, q=q)
	results.model1.data_models[0].plot(ax0, color=c1, q=q)
	### legend:
	label0 = 'Null   --- P(reject)=%.3f' %results.p_reject0
	label1 = 'Alternative --- P(reject)=%.3f' %results.p_reject1
	legend_manual(ax0, colors=[c0,c1], labels=[label0,label1], linewidths=[2,2], fontsize=10, loc=(0.95,0.7))
	### plot "null" power continua:
	c0,c1   = 'b', 'c'
	ls0,ls1 = ':', '-'
	lw0,lw1 = 3, 3
	ax2.plot( q, results.p1d_poi0, color=c0, ls=ls0, lw=lw0, label='POI power' )
	ax2.plot( q, results.p1d_coi0, color=c1, ls=ls1, lw=lw1, label='COI power (radius=%d)' %results.coir )
	### plot "alternative" power continua:
	ax3.plot( q, results.p1d_poi1, color=c0, ls=ls0, lw=lw0 )
	ax3.plot( q, results.p1d_coi1, color=c1, ls=ls1, lw=lw1 )
	### add datum lines
	for i,ax in enumerate([ax2,ax3]):
		label = None if i>0 else 'Power datum'
		ax.hlines([0, 0.8, 1], 0, results.Q, color='k', linestyle='--', lw=0.5, label=label)
	### add POI+COI legend:
	ax2.legend(loc='center right', fontsize=8)
	### add ROIs:
	if results.roi is not None:
		if not np.all(results.roi.value):
			[results.roi.plot(ax)  for ax in AX]
	### add axis labels:
	for ax in AX:
		ax.set_xlabel('Continuum position', size=12)
	ax0.set_ylabel('Continuum value', size=12)
	ax2.set_ylabel('Power', size=12)
	### add panel labels:
	labels  = []
	labels.append(  'Null power continua'  )
	labels.append(  'Alternative power continua'  )
	tx      = [ax.text(0.05, 0.90,  label, transform=ax.transAxes)   for ax,label in zip([ax2,ax3],labels)]
	pyplot.setp([ax2,ax3], xlim=q[[0,-1]], ylim=(-0.05, 1.2), yticks=np.linspace(0,1,6))


def _plot_results_multiple_datamodels(results, q=None):
	q     = _get_q(results.Q, q)
	### create axes:
	axx   = [0.09, 0.57]
	axy   = [0.58, 0.08]
	axw   = 0.41
	axh   = 0.41
	ax0   = pyplot.axes([axx[0], axy[0], axw, axh])
	ax1   = pyplot.axes([axx[1], axy[0], axw, axh])
	ax2   = pyplot.axes([axx[0], axy[1], axw, axh])
	ax3   = pyplot.axes([axx[1], axy[1], axw, axh])
	AX    = [ax0,ax1,ax2,ax3]
	### plot experiment models:
	c0,c1 = '0.5', 'k'
	for m,c in zip(results.model0.data_models, [c0,c1]):
		m.plot(ax0, color=c, q=q)
	for m,c in zip(results.model1.data_models, [c0,c1]):
		m.plot(ax1, color=c, q=q)
	### plot "null" power continua:
	c0,c1   = 'b', 'c'
	ls0,ls1 = ':', '-'
	lw0,lw1 = 1, 3
	ax2.plot( q, results.p1d_poi0, color=c0, ls=ls0, lw=lw0, label='POI power' )
	ax2.plot( q, results.p1d_coi0, color=c1, ls=ls1, lw=lw1, label='COI power (radius=%d)' %results.coir )
	### plot "effect" power continua:
	ax3.plot( q, results.p1d_poi1, color=c0, ls=ls0, lw=lw0 )
	ax3.plot( q, results.p1d_coi1, color=c1, ls=ls1, lw=lw1 )
	### add datum lines
	for i,ax in enumerate([ax2,ax3]):
		label = None if i>0 else 'Power datum'
		ax.hlines([0, 0.8, 1], 0, results.Q, color='k', linestyle='-', label=label)
	### add POI+COI legend:
	ax2.legend(loc='center right', fontsize=8)
	### add ROIs:
	if results.roi is not None:
		if not np.all(results.roi.value):
			[results.roi.plot(ax)  for ax in AX]
	### add axis labels:
	for ax in AX[2:]:
		ax.set_xlabel('Continuum position', size=12)
	ax0.set_ylabel('Continuum value', size=12)
	ax2.set_ylabel('Power', size=12)
	### add panel labels:
	labels  = []
	labels.append(  'Null'  )
	labels.append(  'Alternative'  )
	labels.append(  'Null power continua'  )
	labels.append(  'Alternative power continua'  )
	tx      = [ax.text(0.05, 0.90,  label, transform=ax.transAxes)   for ax,label in zip(AX,labels)]
	labels2 = ['Omnibus P(reject) = %.3f' %p  for p in [results.p_reject0, results.p_reject1]]
	tx2     = [ax.text(0.5, 0.05,  label, transform=ax.transAxes, ha='center')   for ax,label in zip(AX,labels2)]
	pyplot.setp(tx2, bbox=dict(facecolor='w'))
	pyplot.setp([ax2,ax3], xlim=q[[0,-1]], ylim=(-0.05, 1.2), yticks=np.linspace(0,1,6))


