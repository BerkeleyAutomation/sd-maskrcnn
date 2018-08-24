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
from util import *
import numpy as np

def trainSeed( N_SEED=200, shrink=0.5, detector='mssf' ):
	print("Training seeds")
	print("  * Loading dataset")
	over_segs,segmentations,boxes = loadVOCAndOverSeg( "train", detector=detector )
	print("  * Reprojecting ")
	print( (segmentations[0]+1).dtype )
	psegs = [over_seg.projectSegmentation( seg+1 )-1 for over_seg,seg in zip(over_segs,segmentations)]

	print("  * Shrinking segments")
	# Shrink each of the segments by about 50% [if possible]
	for over_seg,pseg in zip(over_segs,psegs):
		N,M = pseg.shape[0],np.max(pseg)+1
		# Find the distance to the object boundary (for each object)
		adj_list = [[] for i in range(N)]
		from queue import Queue
		q = Queue()
		for e in over_seg.edges:
			if pseg[e.a] != pseg[e.b]:
				q.put((e.a,0))
				q.put((e.b,0))
			else:
				adj_list[e.a].append( e.b )
				adj_list[e.b].append( e.a )
		d = -np.ones( N )
		d[ pseg<0 ] = 0
		while not q.empty():
			n,l = q.get()
			if d[n] >= 0:
				continue
			d[n] = l
			for i in adj_list[n]:
				if d[i]==-1:
					q.put((i,l+1))
		# Find the distance distribution for each object
		for l in range(M):
			dd = d[ pseg==l ]
			if dd.shape[0]>0:
				t = np.sort(dd)[int(shrink*dd.shape[0])]
				pseg[ np.logical_and( pseg==l, d<t ) ] = -2

	# Train the seeds
	print("  * Training")
	s = proposals.LearnedSeed()
	s.train( over_segs, psegs, N_SEED )

	return s

def main( argv ):
	s = trainSeed()
	s.save( '../data/seed_final.dat' )
	return 1

if __name__ == "__main__":
	from sys import argv
	r = main( argv )
	exit( r )
