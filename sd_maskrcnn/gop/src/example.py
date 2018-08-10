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
from gop import *
import numpy as np, os
from util import *
from sys import argv
from time import time

#prop = proposals.Proposal( setupBaseline( 130, 5, 0.8 ) )
#prop = proposals.Proposal( setupBaseline( 150, 7, 0.85 ) )
prop = proposals.Proposal( setupLearned( 140, 4, 0.8 ) )
#prop = proposals.Proposal( setupLearned( 160, 6, 0.85 ) )

dir_path = os.path.dirname(os.path.realpath(__file__))
detector = contour.MultiScaleStructuredForest()
detector.load(os.path.join(dir_path, '..', 'data', 'sf.dat'))

for im in argv[1:]:
	t0 = time()
	s = segmentation.geodesicKMeans( imgproc.imread( im ), detector, 1000 )
	t1 = time()
	b = prop.propose( s )
	t2 = time()
	# If you just want the boxes use
	boxes = s.maskToBox( b )
	print( "Generated %d proposals in %0.2fs (OverSeg: %0.2fs, Prop: %0.2fs)"%(b.shape[0],t2-t0,t1-t0,t2-t1) )
	figure()
	for i in range(min(20,b.shape[0])):
		im = np.array( s.image )
		im[ b[i,s.s] ] = (255,0,0)
		ax = subplot( 4, 5, i+1 )
		ax.imshow( im )
		# Draw the bounding box
		from matplotlib.patches import FancyBboxPatch
		ax.add_patch( FancyBboxPatch( (boxes[i,0],boxes[i,1]), boxes[i,2]-boxes[i,0], boxes[i,3]-boxes[i,1], boxstyle="square,pad=0.", ec="b", fc="none", lw=2) )
show()
