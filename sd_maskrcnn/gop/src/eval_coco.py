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
from util import *
from time import sleep

LATEX_OUTPUT=True

T = 625

N_S,N_T,iou = 400,15,0.9
all_bo,all_pool_s = [],[]
for fold in range(dataset.cocoNFolds()):
	# Load the dataset
	over_segs,segmentations,boxes = loadCOCOAndOverSeg( "valid", N_SPIX=2000, detector='mssf', fold=fold )
	# Generate the proposals
	prop_settings = setupBaseline( N_S, N_T, iou, SEED_PROPOSAL=True )
	bo,b_bo,pool_s,box_pool_s = dataset.proposeAndEvaluate( over_segs, segmentations, [], proposals.Proposal( prop_settings ) )
	all_pool_s.extend( pool_s )
	all_bo.extend( bo )
	print( "   Baseline GOP ($%d$,$%d$)         & %d & %0.3f & %0.3f & %0.3f & %0.3f &  \\\\"%(N_S,N_T,np.mean(pool_s),np.mean(bo[:,0]),np.sum(bo[:,0]*bo[:,1])/np.sum(bo[:,1]), np.mean(bo[:,0]>=0.5), np.mean(bo[:,0]>=0.7) ) )
	del over_segs,segmentations,boxes
pool_s = np.array( all_pool_s )
bo = np.vstack( all_bo )
if LATEX_OUTPUT:
	print( "Baseline GOP ($%d$,$%d$)         & %d & %0.3f & %0.3f & %0.3f & %0.3f &  \\\\"%(N_S,N_T,np.mean(pool_s),np.mean(bo[:,0]),np.sum(bo[:,0]*bo[:,1])/np.sum(bo[:,1]), np.mean(bo[:,0]>=0.5), np.mean(bo[:,0]>=0.7) ) )
	m = bo[:,1] < T
	print( "Baseline GOP ($%d$,$%d$)  < %d  & %d & %0.3f & %0.3f & %0.3f & %0.3f &  \\\\"%(N_S,N_T,T,np.mean(pool_s),np.mean(bo[m,0]),np.sum(bo[m,0]*bo[m,1])/np.sum(bo[m,1]), np.mean(bo[m,0]>=0.5), np.mean(bo[m,0]>=0.7) ) )
	m = bo[:,1] >= T
	print( "Baseline GOP ($%d$,$%d$)  >=%d  & %d & %0.3f & %0.3f & %0.3f & %0.3f &  \\\\"%(N_S,N_T,T,np.mean(pool_s),np.mean(bo[m,0]),np.sum(bo[m,0]*bo[m,1])/np.sum(bo[m,1]), np.mean(bo[m,0]>=0.5), np.mean(bo[m,0]>=0.7) ) )
else:
	print( "ABO        ", np.mean(bo[:,0]) )
	print( "cover      ", np.sum(bo[:,0]*bo[:,1])/np.sum(bo[:,1]) )
	print( "recall     ", np.mean(bo[:,0]>=0.5), "\t", np.mean(bo[:,0]>=0.6), "\t", np.mean(bo[:,0]>=0.7), "\t", np.mean(bo[:,0]>=0.8), "\t", np.mean(bo[:,0]>=0.9), "\t", np.mean(bo[:,0]>=1) )
	print( "# props    ", np.mean(pool_s) )

	print( "box ABO    ", np.mean(b_bo) )
	print( "box recall ", np.mean(b_bo>=0.5), "\t", np.mean(b_bo>=0.6), "\t", np.mean(b_bo>=0.7), "\t", np.mean(b_bo>=0.8), "\t", np.mean(b_bo>=0.9), "\t", np.mean(b_bo>=1) )
	print( "# box      ", np.mean(box_pool_s[~np.isnan(box_pool_s)]) )
