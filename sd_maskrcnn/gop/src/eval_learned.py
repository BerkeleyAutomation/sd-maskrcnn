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
from train_seed import trainSeed
from pickle import load, dump
import train_mask

LATEX_OUTPUT=True

d='mssf'
N_MASKS = 3

# Train the seeds
try:
	seed = proposals.LearnedSeed()
	seed.load( '../data/seed_final.dat' )
except:
	print("====== Training seeds ====== ")
	seed = trainSeed( 200, detector=d )
	seed.save( '../data/seed_final.dat' )

# Training the masks
try:
	masks = []
	for i in range(N_MASKS):
		fg = proposals.binaryLearnedUnary( "../data/masks_final_%d_fg.dat"%i )
		bg = proposals.binaryLearnedUnary( "../data/masks_final_%d_bg.dat"%i )
		masks.append( (fg,bg) )
except:
	print("====== Training masks ======")
	masks = train_mask.train( N_MASKS, seed_func=seed )
	for i,u in enumerate(masks):
		proposals.saveLearnedUnary( "../data/masks_final_%d_fg.dat"%i, u[0] )
		proposals.saveLearnedUnary( "../data/masks_final_%d_bg.dat"%i, u[1] )
	print("====== Evaluation ======")

# Load the dataset
over_segs,segmentations,boxes = loadVOCAndOverSeg( "test", detector=d, year="2012" )
#over_segs,segmentations,boxes = loadCOCOAndOverSeg( "valid", detector='mssf', fold=0 )
has_box = [len(b)>0 for b in boxes]
boxes = [np.vstack(b).astype(np.int32) if len(b)>0 else np.zeros((0,4),dtype=np.int32) for b in boxes]

# Generate the proposals
s = []
s.append( (140,4,0.8) ) # ~650 props [mssf]
s.append( (160,6,0.85) ) # ~1100 props
s.append( (180,9,0.9) ) # ~2200 props
s.append( (200,15,0.9) ) # ~4400 props
for N_S,N_T,iou in s:
	prop_settings = setupLearned( N_S, N_T, iou, SEED_PROPOSAL=True )
	bo,b_bo,pool_s,box_pool_s = dataset.proposeAndEvaluate( over_segs, segmentations, boxes, proposals.Proposal( prop_settings ) )
	if LATEX_OUTPUT:
		print( "Learned GOP ($%d$,$%d$) & %d & %0.3f & %0.3f & %0.3f & %0.3f &  \\\\"%(N_S,N_T,np.mean(pool_s),np.mean(bo[:,0]),np.sum(bo[:,0]*bo[:,1])/np.sum(bo[:,1]), np.mean(bo[:,0]>=0.5), np.mean(bo[:,0]>=0.7) ) )
	else:
		print( "ABO        ", np.mean(bo[:,0]) )
		print( "cover      ", np.sum(bo[:,0]*bo[:,1])/np.sum(bo[:,1]) )
		print( "recall     ", np.mean(bo[:,0]>=0.5), "\t", np.mean(bo[:,0]>=0.6), "\t", np.mean(bo[:,0]>=0.7), "\t", np.mean(bo[:,0]>=0.8), "\t", np.mean(bo[:,0]>=0.9), "\t", np.mean(bo[:,0]>=1) )
		print( "# props    ", np.mean(pool_s) )

		print( "box ABO    ", np.mean(b_bo) )
		print( "box recall ", np.mean(b_bo>=0.5), "\t", np.mean(b_bo>=0.6), "\t", np.mean(b_bo>=0.7), "\t", np.mean(b_bo>=0.8), "\t", np.mean(b_bo>=0.9), "\t", np.mean(b_bo>=1) )
		print( "# box      ", np.mean(box_pool_s[~np.isnan(box_pool_s)]) )
