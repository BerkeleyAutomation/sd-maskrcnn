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
from pickle import load
from util import *

# GOP settings
N_SEED = [1,2,5,10,20,35,50,75,100,200]

seed_functions = [load( open('../data/seed_final.dat','rb') ),proposals.GeodesicSeed(),proposals.SegmentationSeed(),proposals.RegularSeed(),proposals.RandomSeed()]
recomp = [0,0,1,1,0,0]
if __name__ == "__main__":
	over_segs,segmentations,boxes = loadVOCAndOverSeg( "test", detector='mssf', year="2012" )
	
	n_seg, n_spix, n_seed, n_prod = 0, 0, np.zeros( ( len(seed_functions), len(N_SEED) ) ), np.zeros( ( len(seed_functions), len(N_SEED) ) )
	for id,(gop,seg) in enumerate(zip( over_segs, segmentations )):
		# Count the total number of segments
		pseg = gop.projectSegmentation( seg+1 )-1
		nseg = np.max(seg)+1
		sp_seg = np.unique(pseg[pseg>=0])
		nseg_in_sp = sp_seg.shape[0]
		
		for i,sf in enumerate( seed_functions ):
			MS = int(np.max(N_SEED))
			if not recomp[i]:
				s = sf.compute( gop, MS )
			
			for j,n in enumerate( N_SEED ):
				if recomp[i]:
					ss = sf.compute( gop, n )
				else:
					ss = s[:n]
				# Find and count all the hit segments
				seed_seg = pseg[ss]
				seed_seg = np.unique(seed_seg[seed_seg>=0])
				nseg_in_seed = seed_seg.shape[0]
				
				# Update the count
				n_seed[i,j] += nseg_in_seed
				n_prod[i,j] += np.unique(ss).shape[0]

		# Update the other counts
		n_spix += nseg_in_sp
		n_seg += nseg

	print( "#SP Seed      %d / %d [%f]"%(n_spix, n_seg, 100*n_spix/n_seg) )
	print( "-"*60 )
	print( ' '*40, "".join( ["     %5d     "%n for n in N_SEED] ) )
	for i,sf in enumerate( seed_functions ):
		print( "%-40s   %s      %d"%(str(type(sf).__name__),"    ".join( ["%4d [%3.1f]"%(n, 100*n/n_seg) for n in n_seed[i]] ), n_prod[i,-1] ) )
