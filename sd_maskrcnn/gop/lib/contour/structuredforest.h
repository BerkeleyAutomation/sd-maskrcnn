/*
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
*/
#pragma once
#include "boundarydetector.h"
#include "learning/features.h"
#include "learning/forest.h"

struct StructuredForestSettings{
	int stride, shrink, out_patch_size, feature_patch_size;
	int patch_smooth, sim_smooth;
	int sim_cells;
	StructuredForestSettings( int stride=2, int shrink=2, int out_patch_size=16, int feature_patch_size=32, int patch_smooth=2, int sim_smooth=8, int sim_cells=5 );
};

class StructuredForest: public BoundaryDetector
{
protected:
	friend class SPStructuredForest;
	int nms_, suppress_;
	StructuredForestSettings settings_;
	RangeForest forest_;
	VectorXus patch_ids_;
public:
	StructuredForest( int nms=1, int suppress=5, const StructuredForestSettings & s = StructuredForestSettings() );
	StructuredForest( const StructuredForestSettings & s );
	void load( const std::string & fn );
	void save( const std::string & fn ) const;
	virtual RMatrixXf detect( const Image8u & rgb_im ) const;
	virtual RMatrixXf filter( const RMatrixXf & detection, int suppress, int nms=-1 ) const;
	virtual RMatrixXf filter( const RMatrixXf & detection ) const;
	void setFromMatlab(const RMatrixXf &thrs, const RMatrixXi &child, const RMatrixXi &fid, const VectorXi &rng, const VectorXus & patches);
	// Training
	void fitAndAddTree( const Features & f, const RMatrixXf & lbl, const RMatrixXb & patch_data, const VectorXi & fid, TreeSettings settings = TreeSettings(), bool mt=false );
	void duplicateAndAddTree( const Features & f, const RMatrixXf & lbl, const RMatrixXb & patch_data, TreeSettings settings = TreeSettings() );
	void compress();
	// Debug
	RMatrixXb predictLastTree( const Features & f, const VectorXi & sid ) const{
		const int N = sid.size();
		RMatrixXb r = RMatrixXb::Zero( N, settings_.out_patch_size*settings_.out_patch_size );
		for( int i=0; i<N; i++ ) {
			RangeData d = forest_.tree(forest_.nTrees()-1).predictData( f, sid[i] );
			for( int j=d.begin; j<d.end; j++ )
				r(i,patch_ids_[j]) = 1;
		}
		return r;
	}
};
class MultiScaleStructuredForest: public StructuredForest
{
public:
	virtual RMatrixXf detect( const Image8u & rgb_im ) const;
};

class SFFeatures: public Features {
protected:
	friend class StructuredForest;
	friend class SPStructuredForest;
	RMatrixXf patch_features_, ssim_features_;
	std::vector<int> did_, did1_, did2_, id_;
	VectorXi x_, y_;
	int n_feature_, n_ssim_feature_, n_patch_feature_;
public:
	SFFeatures( const Image8u & rgb_im, const StructuredForestSettings & s = StructuredForestSettings() );
	const RMatrixXf& patchFeatures() const;
	const RMatrixXf& ssimFeatures() const;
	const VectorXi & x() const;
	const VectorXi & y() const;
	virtual int nSamples() const;
	virtual int featureSize() const;
	virtual float get( int s, int f ) const;
};
