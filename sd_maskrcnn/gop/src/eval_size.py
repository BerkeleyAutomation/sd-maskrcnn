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
from gop import *
import numpy as np
from time import sleep
from pickle import dump,load
from util import *

over_segs,segmentations,boxes = loadVOCAndOverSeg( "test", detector='mssf', year="2012" )
has_box = [len(b)>0 for b in boxes]
boxes = [np.vstack(b).astype(np.int32) if len(b)>0 else np.zeros((0,4),dtype=np.int32) for b in boxes]

prop_settings = setupBaseline( 130, 5, 0.8 )
bo,b_bo,pool_s,box_pool_s = dataset.proposeAndEvaluate( over_segs, segmentations, boxes, proposals.Proposal( prop_settings ) )

print( "ABO        ", np.mean(bo[:,0]) )
print( "cover      ", np.sum(bo[:,0]*bo[:,1])/np.sum(bo[:,1]) )
print( "recall     ", np.mean(bo[:,0]>=0.5), "\t", np.mean(bo[:,0]>=0.6), "\t", np.mean(bo[:,0]>=0.7), "\t", np.mean(bo[:,0]>=0.8), "\t", np.mean(bo[:,0]>=0.9), "\t", np.mean(bo[:,0]>=1) )
print( "# props    ", np.mean(pool_s) )

def plotAndEBar( x, y, nbins=15, scatter=True, c=None, lbl=None ):
	from pylab import plot,errorbar,xscale,fill_between,ylim
	n, p = np.histogram(np.log(x), bins=nbins)
	sy, p = np.histogram(np.log(x), bins=nbins, weights=y)
	sy2, p = np.histogram(np.log(x), bins=nbins, weights=y*y)
	mean = sy / n
	std = np.sqrt(sy2/n - mean*mean)
	if scatter:
		plot(x, y,'*')
	fill_between(np.exp((p[1:] + p[:-1])/2), mean-std, mean+std, facecolor=c, alpha=0.3)
	plot(np.exp((p[1:] + p[:-1])/2), mean, c=c, label=lbl)
	xscale('log')
	ylim(0,1)

if 1:
	from pylab import plot,figure,show, legend, savefig, xlabel, ylabel
	
	figure(figsize=(4,3)).add_axes( [0.15, 0.17, 0.8, 0.8] )
	plotAndEBar( bo[:,1], bo[:,0], scatter=False, c='r', lbl='GOP' )
	xlabel("size (#pixels)")
	ylabel("best overlap")
	legend(loc=4)
	savefig( 'size.pdf' )
	show()
