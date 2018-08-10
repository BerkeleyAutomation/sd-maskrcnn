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
from sys import stdout
import os
DIR_PATH = os.path.dirname(os.path.realpath(__file__))

class ProgressPrint:
	def __init__( self, message, start, end=None, show_mem=False ):
		self.m = message
		if end == None:
			self.start = 0
			self.range = start
		else:
			self.start = start
			self.range = end-start
		self.show_mem = show_mem
		self.update(0)
	def update( self, n ):
		mem = ""
		if self.show_mem:
			m = mem_usage()
			if m>0:
				mem = "  [ %0.1fMb used ]"%m
		try:
			stdout.write( (self.m+mem+"   \r")%(100*(n-self.start+1e-50)/(self.range+1e-50)) )
		except:
			stdout.write( ("%s%0.1f%%   "+mem+"\r")%(self.m,100*(n-self.start+1e-50)/(self.range+1e-50)) )
	def __del__( self ):
		stdout.write('\n')

def getPSUtil():
	try:
		import psutil
		return psutil
	except:
		return None

def getProcess():
	psutil = getPSUtil()
	if psutil:
		import os
		return psutil.Process( os.getpid() )
	return None
	
def mem_usage():
	p = getProcess()
	if p == None:
		return 0
	return p.get_memory_info()[0] / 1048576.

def printMemUsage():
	psutil = getPSUtil()
	if psutil:
		u = mem_usage()
		M = psutil.virtual_memory()
		print( "Process Mem: %0.1fMb     System: %0.1fMb / %0.1fMb"%(u, M.used/1048576., M.total/1048576.) )

def fastSampleWithoutRep( a, size, tile=True ):
	S = np.random.randint( 0, a-1, size=2*size )
	US = np.unique( S )
	np.random.shuffle( US )
	if US.shape[0] < size:
		if tile:
			return US[np.arange(size)%US.shape[0]]
		return US
	return US[:size]

def loadCOCOAndOverSeg( im_set="test", detector="sf", N_SPIX=1000, fold=0 ):
	from pickle import dumps,loads
	try:
		import lz4, pickle
		decompress = lambda s: pickle.loads( lz4.decompress( s ) )
		compress = lambda o: lz4.compressHC( pickle.dumps( o ) )
	except:
		compress = lambda x: x
		decompress = lambda x: x
	from .gop import contour,dataset,segmentation
	FILE_NAME = '/tmp/coco_%s_%s_%d_%d.dat'%(im_set,detector,N_SPIX,fold)
	try:
		with open(FILE_NAME,'rb') as f:
			over_segs,segmentations = loads( f.read() )
			f.close()
			over_seg = segmentation.ImageOverSegmentationVec()
			for i in over_segs:
				over_seg.append( decompress(i) )
			return over_seg,[decompress(i) for i in segmentations],[]
			#return over_segs,segmentations,[]
	except FileNotFoundError:
		pass
	
	# Load the dataset
	data = dataset.loadCOCO2014( im_set=="train",im_set=="valid", fold)
	
	# COCO has some pretty gray scale images (WTF!!!)
	images = [e['image'] if e['image'].C==3 else e['image'].tileC(3)  for e in data]
	try:
		segmentations = [e['segmentation'] for e in data]
	except:
		segmentations = []
	
	# Do the over-segmentation
	if detector=='sf':
		detector = contour.StructuredForest()
		detector.load( 'gop/data/sf.dat' )
	elif detector == "mssf":
		detector = contour.MultiScaleStructuredForest()
		detector.load( "gop/data/sf.dat" )
	elif detector=='st':
		detector = contour.SketchTokens()
		detector.load( 'data/st_full_c.dat' )
	else:
		detector = contour.DirectedSobel()
	
	if detector != None:
		over_segs = segmentation.generateGeodesicKMeans( detector, images, N_SPIX )
	with open(FILE_NAME,'wb') as f:
		#f.write( dumps( (over_segs,segmentations) ) )
		f.write( dumps( ([compress(i) for i in over_segs],[compress(i) for i in segmentations]) ) )
		f.close()
	
	return over_segs,segmentations,[]

def loadVOCAndOverSeg( im_set="test", detector="sf", N_SPIX=1000, EVAL_DIFFICULT=False, year="2012" ):
	from pickle import dumps,loads
	try:
		import lz4, pickle
		decompress = lambda s: pickle.loads( lz4.decompress( s ) )
		compress = lambda o: lz4.compressHC( pickle.dumps( o ) )
	except:
		compress = lambda x: x
		decompress = lambda x: x
	from .gop import contour,dataset,segmentation
	FILE_NAME = '/tmp/%s_%s_%d_%d_%s.dat'%(im_set,detector,N_SPIX,EVAL_DIFFICULT,year)
	try:
		with open(FILE_NAME,'rb') as f:
			over_segs,segmentations,boxes = loads( f.read() )
			f.close()
			over_seg = segmentation.ImageOverSegmentationVec()
			for i in over_segs:
				over_seg.append( decompress(i) )
			return over_seg,[decompress(i) for i in segmentations],[decompress(i) for i in boxes]
	except FileNotFoundError:
		pass
	
	# Load the dataset
	data = eval("dataset.loadVOC%s"%year)(im_set=="train",im_set=="valid",im_set=="test")
	
	images = [e['image'] for e in data]
	try:
		segmentations = [e['segmentation'] for e in data]
	except:
		segmentations = []
	boxes = [[a['bbox'] for a in e['annotation'] if not a['difficult'] or EVAL_DIFFICULT] for e in data]

	# Do the over-segmentation
	if detector=='sf':
		detector = contour.StructuredForest()
		detector.load( '../data/sf.dat' )
	elif detector == "mssf":
		detector = contour.MultiScaleStructuredForest()
		detector.load( "../data/sf.dat" )
	elif detector=='st':
		detector = contour.SketchTokens()
		detector.load( '../data/st_full_c.dat' )
	else:
		detector = contour.DirectedSobel()
	
	if detector != None:
		over_segs = segmentation.generateGeodesicKMeans( detector, images, N_SPIX )
	#try:
	with open(FILE_NAME,'wb') as f:
		f.write( dumps( ([compress(i) for i in over_segs],[compress(i) for i in segmentations],[compress(i) for i in boxes]) ) )
		f.close()
	#except FileNotFoundError:
		#pass
	
	return over_segs,segmentations,boxes

def fastSampleWithoutRep( a, size, tile=True ):
	import numpy as np
	if a<2:
		return np.zeros(size,dtype=int)
	if size==0:
		return np.empty(0,dtype=int)
	S = np.random.randint( 0, a-1, size=2*size )
	US = np.unique( S )
	np.random.shuffle( US )
	if US.shape[0] < size:
		if tile:
			return US[np.arange(size)%US.shape[0]]
		return US
	return US[:size]

def setupBaseline(N_S,N_T,max_iou=0.85,SEED_PROPOSAL=False):
	from .gop import proposals
	prop_settings = proposals.ProposalSettings()
	prop_settings.max_iou = max_iou
	del prop_settings.unaries[:]
	
	prop_settings.unaries.append( proposals.UnarySettings( N_S, N_T, proposals.seedUnary(), [0,15] ) )
	if SEED_PROPOSAL: # Seed proposals
		prop_settings.unaries.append( proposals.UnarySettings( N_S, 1, proposals.seedUnary(), [], 0, 0 ) )
	# Background only proposals
	prop_settings.unaries.append( proposals.UnarySettings( 0, N_T, proposals.seedUnary(), list(range(16)), 0.1, 1  ) )
	return prop_settings

def setupLearned(N_S,N_T,max_iou=0.85,N_MASKS=3,SEED_PROPOSAL=False):	
	from .gop import proposals
	prop_settings = proposals.ProposalSettings()
	prop_settings.max_iou = max_iou
	# Load the seeds
	seed = proposals.LearnedSeed()
	seed.load(os.path.join(DIR_PATH, '..', 'data', 'seed_final.dat'))
	prop_settings.foreground_seeds = seed
	
	# Load the masks
	del prop_settings.unaries[:]
	for i in range(N_MASKS):
		fg = proposals.binaryLearnedUnary(os.path.join(DIR_PATH, '..', 'data', 'masks_final_%d_fg.dat'%i))
		bg = proposals.binaryLearnedUnary(os.path.join(DIR_PATH, '..', 'data', 'masks_final_%d_bg.dat'%i))
		prop_settings.unaries.append( proposals.UnarySettings( N_S, N_T, fg, bg ) )
	if SEED_PROPOSAL: # Seed proposals
		prop_settings.unaries.append( proposals.UnarySettings( N_S, 1, proposals.seedUnary(), [], 0, 0 ) )
	# Background only proposals
	prop_settings.unaries.append( proposals.UnarySettings( 0, N_T, proposals.seedUnary(), list(range(16)), 0.1, 1  ) )
	return prop_settings
