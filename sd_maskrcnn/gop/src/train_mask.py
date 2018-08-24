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
from util import *
import numpy as np
from pickle import load, dump
try:
	from joblib import Parallel, delayed
except:
	# if joblib does not exist just run it in a single thread
	delayed = lambda x: x
	def Parallel( *args, **kwargs ):
		return list
from sklearn import svm,linear_model

# Allow pickling member functions
def _pickle_method(method):
	func_name = method.__name__
	obj = method.__self__
	return _unpickle_method, (func_name, obj)

def _unpickle_method(func_name, obj):
	try:
		return obj.__getattribute__(func_name)
	except AttributeError:
		return None

import copyreg, types
copyreg.pickle(types.MethodType, _pickle_method, _unpickle_method)


N_SEED = 50
N_INITIAL = 3
N_ITER = 2
MAX_SAMPLE = 100000

def computeFeatures( features, over_seg, seed ):
	im_f = proposals.UnaryFeatures( over_seg, features )
	return [im_f.compute( s ) for s in seed]

def fitMask(obj_feature, cls, bs, obj_ids):
	# Train a model for foreground and background model
	r = []
	for c,g in zip(cls,bs):
		pos = np.vstack( [obj_feature[i][ g[i]] for i in obj_ids] )
		neg = np.vstack( [obj_feature[i][~g[i]] for i in obj_ids] )
		pick_pos,pick_neg = pos,neg
		if pos.shape[0] > MAX_SAMPLE:
			pick_pos = pos[np.random.choice(pos.shape[0],MAX_SAMPLE)]
		if neg.shape[0] > MAX_SAMPLE:
			pick_neg = neg[np.random.choice(neg.shape[0],MAX_SAMPLE)]
		f = np.vstack((pick_pos,pick_neg))
		lbl = np.hstack((np.ones(pick_pos.shape[0],dtype=int),np.zeros(pick_neg.shape[0],dtype=int)))
		c.fit( f, lbl )
		print( "    True / False pos ", np.sum( c.predict( pos ) ), " / ", np.sum( c.predict( neg ) ) )
	print()

def scoreCls( c, f, l ):
	cw = c.class_weight
	d = c.decision_function( f )
	return cw[0]*np.sum(np.maximum(1+d[l==0],0))+cw[1]*np.sum(np.maximum(1-d[l==1],0))

def scoreF( f, cls, bs ):
	score = None
	for c,g in zip(cls,bs):
		s = scoreCls( c, np.vstack((f[ g],f[~g])), np.hstack((np.ones(np.sum(g)),np.zeros(np.sum(~g)))) )
		if score==None:
			score = s
		else:
			score = (s+score)/2
	return score
	

def score( obj_feature, cls, bs, obj_ids ):
	return [scoreF( obj_feature[i], cls, (bs[0][i],bs[1][i]) ) for i in obj_ids]
	# All that pickling joblib does is just too slow
	#return Parallel(n_jobs=-1)( delayed(scoreF)( obj_feature[i], cls, (bs[0][i],bs[1][i]) ) for i in obj_ids )

def train( n_masks=3, seed_func=None ):
	if seed_func == None:
		seed_func = proposals.LearnedSeed()
		seed_func.load( '../data/seed_final.dat' )
	
	from sys import stdout		
	stdout.write( "Loading the data...                " )
	stdout.flush()
	over_segs,segmentations,boxes = loadVOCAndOverSeg( "train" , 'mssf', year="2012" )
	print('[done]')
	
	
	stdout.write( "Computing seeds...                 " )
	stdout.flush()
	seeds = list( Parallel(n_jobs=-1)( delayed(seed_func.compute)( over_seg, N_SEED ) for over_seg in over_segs ) )
	print('[done]')
	
	# Collecting all objects
	stdout.write( "Collecting objects ...             " )
	stdout.flush()
	im_id,obj_seed,fg,bg = [],[],[],[]
	missed,missed_seed = 0,0
	for i,(seed,over_seg,seg) in enumerate(zip(seeds,over_segs,segmentations)):
		pseg1 = over_seg.projectSegmentation( seg+1 )-1
		pseg2 = over_seg.projectSegmentation( seg+2 )-2
		# For all the objects add them one by one
		for s in range(np.max(seg)+1):
			if s in pseg2:
				valid_seed = seed[pseg2[seed]==s]
				if valid_seed.size > 0:
					fg.extend([pseg2==s]*valid_seed.shape[0])
					bg.extend([pseg1!=s]*valid_seed.shape[0])
					obj_seed.append(valid_seed)
					im_id.extend([i]*valid_seed.shape[0])
				else:
					missed_seed += 1
			else:
				missed += 1
	n_seed,n_obj = len(im_id),len(obj_seed)
	obj_seed = np.hstack(obj_seed)
	print('[done]')
	print( "  Got %d seeds in %d objects. Missed %s (%d by seed)"%(n_seed,n_obj,missed+missed_seed,missed_seed) )
	
	# Get the features
	stdout.write( "Computing features...              " )
	stdout.flush()
	
	features = proposals.defaultUnaryFeatures()
	
	used_seeds = [[] for o in over_segs]
	for i,s in zip(im_id,obj_seed):
		used_seeds[i].append( s )
	
	obj_feature = list( Parallel(n_jobs=-1)( delayed(computeFeatures)( features, over_seg, seed ) for over_seg,seed in zip(over_segs,used_seeds) ) )
	from itertools import chain
	obj_feature = list(chain(*obj_feature))
	print('[done]')

	# Initialize the masks
	print( "Initializing masks..." )
	bs = [fg,bg]
	all_cls = [ [svm.LinearSVC(class_weight={1:0.1,0:10},fit_intercept=True,dual=True,tol=1e-6,loss='l1') for b in bs] for u in range(n_masks) ]

	all_scores = 1e10*np.ones( (n_masks, n_seed) )
	best_cls = np.zeros( n_seed, dtype=int )
	for i,cls in enumerate(all_cls):
		fitMask( obj_feature, cls, bs, np.random.choice(n_seed,N_INITIAL) )
		all_scores[i] = list(score( obj_feature, cls, bs, range(n_seed) ))
	print( )
	
	print( "Training masks..." )
	for it in range(N_ITER):
		# Compute the score for each classifier
		for i,cls in enumerate(all_cls):
			# Find all the training examples
			sel = np.nonzero(all_scores[i]==np.min(all_scores,0))[0]
			print("  Training classifier %d with %d samples"%(i,len(sel)) )
			fitMask( obj_feature, cls, bs, sel )
			all_scores[i] = list(score( obj_feature, cls, bs, range(n_seed) ))
	
	def makeBinaryLearnedUnary( features, cls ):
		return proposals.binaryLearnedUnary( features, -np.array(cls.coef_,dtype=np.float32).flatten(), -float(cls.intercept_) )
	return [( makeBinaryLearnedUnary(features, cls[0]), makeBinaryLearnedUnary(features, cls[1]) ) for cls in all_cls]


def main(argv):
	masks = train()
	if len(argv) > 1:
		for i,u in enumerate(masks):
			proposals.saveLearnedUnary( "%s_%d_fg.dat"%(argv[1],i), u[0] )
			proposals.saveLearnedUnary( "%s_%d_bg.dat"%(argv[1],i), u[1] )

if __name__ == "__main__":
	from sys import argv
	exit( main( argv ) )
