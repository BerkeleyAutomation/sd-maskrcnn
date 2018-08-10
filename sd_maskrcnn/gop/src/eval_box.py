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

prop_settings = proposals.ProposalSettings()
prop_settings.foreground_seeds = proposals.RegularSeed()
prop_settings.max_iou = 0.7
del prop_settings.unaries[:]

# Load the dataset
over_segs,segmentations,boxes = loadVOCAndOverSeg( "test", detector='st', year="2012_detect" )
has_box = [len(b)>0 for b in boxes]
boxes = [np.vstack(b).astype(np.int32) if len(b)>0 else np.zeros((0,4),dtype=np.int32) for b in boxes]

bos = []
ns = []
for N_S,N_T in [(1,1),(1,2),(3,2),(8,3),(12,4),(18,5),(50,5),(70,6),(125,7),(150,10),(180,15),(200,20)]:#,(200,40)]:
	if N_S>3:
		#prop_settings.foreground_seeds = proposals.GeodesicSeed()
		seed = proposals.LearnedSeed()
		seed.load( '../data/seed_final.dat' )# load( open('../data/seed_final.dat','rb') )
		prop_settings.foreground_seeds = seed
	if N_T>=5:
		prop_settings.max_iou = 0.8
	if N_T>6:
		prop_settings.max_iou = 0.9
	if N_S>40:
		del prop_settings.unaries[:]
		prop_settings.unaries.append( proposals.UnarySettings( N_S, N_T, proposals.seedUnary(), [0,1,15] ) )
		prop_settings.unaries.append( proposals.UnarySettings( 0, N_T, proposals.seedUnary(), list(range(16)), 0.1, 1  ) )
	elif N_S>1:
		del prop_settings.unaries[:]
		prop_settings.unaries.append( proposals.UnarySettings( N_S, N_T, proposals.seedUnary(), [0,15] ) )
	else:
		del prop_settings.unaries[:]
		prop_settings.unaries.append( proposals.UnarySettings( N_S, N_T, proposals.seedUnary(), [15] ) )
	
	# Generate the proposals
	bo,b_bo,pool_s,box_pool_s = dataset.proposeAndEvaluate( over_segs, [], boxes, proposals.Proposal( prop_settings ) )
	ns.append( np.mean(box_pool_s[~np.isnan(box_pool_s)]) )
	bos.append( b_bo )
	print( "N_S = %3d   N_T = %2d    # windows = %4.0f [%4.0f .. %4.0f]  ABO = %4.2f  [%0.2f]"%(N_S,N_T,np.mean(box_pool_s[~np.isnan(box_pool_s)]),np.min(box_pool_s[~np.isnan(box_pool_s)]), np.max(box_pool_s[~np.isnan(box_pool_s)]),np.mean(b_bo),np.mean(np.maximum(2*b_bo-1,0))) )

ns = np.hstack( ns )
bos = np.vstack( bos )

from sys import argv
if len(argv)>1:
	from pickle import dump
	dump( (ns,bos), open(argv[1],'wb') )
