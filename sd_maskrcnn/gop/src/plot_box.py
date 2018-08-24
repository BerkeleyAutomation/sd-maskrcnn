# -*- encoding: utf-8
"""
    Copyright (c) 2014, Philipp Krähenbühl
    All rights reserved.
	
    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
        * Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
        * Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.
        * Neither the name of the Stanford University nor the
        names of its contributors may be used to endorse or promote products
        derived from this software without specific prior written permission.
	
    THIS SOFTWARE IS PROVIDED BY Philipp Krähenbühl ''AS IS'' AND ANY
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL Philipp Krähenbühl BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
	 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
	 ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
	 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
from pylab import *
import numpy as np
from sys import argv
from pickle import load
from os import path

rc('font', size=20)

MAX_WINDOWS = 2000

params = argv[1:]
output = None
if len(params) and ".pdf" in params[-1]:
	output = params[-1]
	params = params[:-1]

def createFigAx():
	fig = figure(figsize=(5,4))
	ax = fig.add_axes([0.15,0.15,0.8,0.8])
	ax.grid()
	return fig,ax

name_map = {"selsearch_small":"Selective Search","rprim_large":"Randomized Prim","objectness":"Objectness","gop_base":"GOP"}
colors = {"Selective Search":"#204a87","Randomized Prim":"#4e9a06","Objectness":"#ce5c00","GOP":"#a40000"}
bo={}
n ={}
for p in params:
	name = path.basename(p).replace(".dat","")
	if name in name_map:
		name = name_map[ name ]
	n[name],bo[name] = load( open(p,'rb') )

def plot_recall_iou( n, bo, t, legend=False ):
	fig = figure(figsize=(5,4))
	ax = fig.add_axes([0.19,0.17,0.8,0.8])
	ax.grid()
	
	for k,l in enumerate(sorted(bo.keys())):
		recall = np.mean( bo[l]>=t, axis=1 )
		nl = np.logspace(0,4,20)
		irecall = np.interp(np.log(nl),np.log(n[l]),recall)
		c = 'blue'
		if l in colors:
			c = colors[l]
		plot( nl, irecall, 'o-', zorder=10-k, label=l, lw=3, c=c )
		
	
	ax.set_xscale('log')
	if legend:
		ax.legend(loc=2,prop={'size':16}).set_zorder(20)
	ax.set_ylabel("Recall")
	ax.set_xlabel("# of boxes")
	ax.set_xlim(10,MAX_WINDOWS)
	#ax.set_ylim(0,1)
	ax.spines['top'].set_color('gray')
	ax.spines['right'].set_color('gray')
	ax.xaxis.set_ticks_position('bottom')
	ax.yaxis.set_ticks_position('left')
	ax.yaxis.set_major_locator(MultipleLocator(0.2))
	return fig

def plot_abo( n, bo, legend=False ):
	fig = figure(figsize=(5,4))
	ax = fig.add_axes([0.19,0.17,0.8,0.8])
	ax.grid()
	
	for k,l in enumerate(sorted(bo.keys())):
		abo = np.mean( bo[l], axis=1 )
		nl = np.logspace(0,4,20)
		iabo = np.interp(np.log(nl),np.log(n[l]),abo)
		c = 'blue'
		if l in colors:
			c = colors[l]
		plot( nl, iabo, 'o-', zorder=10-k, label=l, lw=3, c=c )
	
	ax.set_xscale('log')
	if legend:
		ax.legend(loc=2)
	ax.set_ylabel("Average Best Overlap")
	ax.set_xlabel("# of boxes")
	ax.set_xlim(10,MAX_WINDOWS)
	#ax.set_ylim(0,1)
	ax.spines['top'].set_color('gray')
	ax.spines['right'].set_color('gray')
	ax.xaxis.set_ticks_position('bottom')
	ax.yaxis.set_ticks_position('left')
	ax.yaxis.set_major_locator(MultipleLocator(0.2))
	return fig

def plot_recall_n( n, bo, nn, legend=False ):
	fig = figure(figsize=(5,4))
	ax = fig.add_axes([0.19,0.17,0.8,0.8])
	ax.grid()
	
	t = np.linspace(0.5,1,10)
	for k,l in enumerate(sorted(bo.keys())):
		id = np.argmin( np.abs( n[l] - nn ) )
		b = bo[l][id]
		if n[l][id] < nn:
			n1,n2 = n[l][id],n[l][id+1]
			w = (nn-n1)/(n2-n1)
			b = bo[l][id]*(1-w) + bo[l][id+1]*w
		recall = np.mean( b[None,:]>=t[:,None], axis=1 )
		c = 'blue'
		if l in colors:
			c = colors[l]
		plot( t, recall, 'o-', zorder=10-k, label=l, lw=3, c=c )
	
	if legend:
		ax.legend(loc=2)
	ax.set_ylabel("Recall")
	ax.set_xlabel("$\mathcal{J}$")
	ax.set_xlim(0.5,1)
	#ax.set_ylim(0,1)
	ax.spines['top'].set_color('gray')
	ax.spines['right'].set_color('gray')
	ax.xaxis.set_ticks_position('bottom')
	ax.yaxis.set_ticks_position('left')
	ax.yaxis.set_major_locator(MultipleLocator(0.2))
	return fig

def VUS( n, bo ):
	#t = np.linspace(0.5,1,100)
	#recall_vol = np.mean( bo[None]>=t[:,None,None], axis=2 )
	#mean_recall = np.mean( recall_vol, axis=0 )
	mean_recall = np.mean( np.maximum( 2*bo-1, 0 ), axis=1 )
	nn = np.linspace(1,MAX_WINDOWS,1000)
	nl = np.logspace(0,4,1000)
	nl = nl[nl<MAX_WINDOWS]
	return np.mean(np.interp(np.log(nn),np.log(n),mean_recall)), np.mean(np.interp(np.log(nl),np.log(n),mean_recall))

print( "VUS     Linear   Log" )
for k in sorted(bo.keys()):
	print( k, *VUS(n[k],bo[k]) )

if output!=None:
	from matplotlib.backends.backend_pdf import PdfPages
	pp = PdfPages(output)
	for t in [0.9,0.7,0.5]:
		pp.savefig( plot_recall_iou( n, bo, t, legend=t>0.8 ) )
	pp.savefig( plot_abo( n, bo ) )
	for nn in [100,500,1000]:
		pp.savefig( plot_recall_n( n, bo, nn ) )
	pp.close()
