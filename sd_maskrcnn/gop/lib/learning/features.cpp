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
#include "features.h"
#include "util/util.h"
#include "util/algorithm.h"
#include <iostream>

Features::~Features() {
}
RMatrixXf Features::getFeatures() const {
	RMatrixXf r( nSamples(), featureSize() );
	#pragma omp parallel for
	for( int i=0; i<nSamples(); i++ )
		for( int j=0; j<r.cols(); j++ )
			r(i,j) = get(i,j);
	return r;
}
RMatrixXf Features::getFeatures( const VectorXi & samples ) const {
	RMatrixXf r( samples.size(), featureSize() );
#pragma omp parallel for
	for( int i=0; i<samples.size(); i++ )
		for( int j=0; j<r.cols(); j++ )
			r(i,j) = get(samples[i],j);
	return r;
}
RMatrixXf Features::getFeatures( const VectorXi & samples, const VectorXi & fids ) const {
	RMatrixXf r( samples.size(), fids.size() );
#pragma omp parallel for
	for( int i=0; i<samples.size(); i++ )
		for( int j=0; j<fids.size(); j++ )
			r(i,j) = get(samples[i],fids[j]);
	return r;
}
FeaturesMatrix::FeaturesMatrix(const RMatrixXf &data) : data_(data){
}
int FeaturesMatrix::featureSize() const {
	return data_.cols();
}
int FeaturesMatrix::nSamples() const {
	return data_.rows();
}
float FeaturesMatrix::get(int s, int f) const {
	return data_(s,f);
}
FeaturesVector::FeaturesVector(const std::vector< std::shared_ptr< Features > > & f, const RMatrixXi & ids, const VectorXi & fids ) : features_(f), ids_(ids), fids_(fids) {
	eassert( ids.cols() == 2 );
	// Check all the indices
	for( int i=0; i<ids.rows(); i++ )
		if( ids(i,0) < 0 || ids(i,0) >= f.size() || ids(i,1) < 0 || ids(i,1) >= f[ids(i,0)]->nSamples() )
			throw std::invalid_argument( "Invalid index ("+std::to_string(ids(i,0))+", "+std::to_string(ids(i,1))+")!" );
	if (!fids_.size())
		fids_ = arange( features_[0]->featureSize() );
	eassert( fids_.minCoeff() >= 0 && fids_.maxCoeff() < features_[0]->featureSize() );
}
int FeaturesVector::featureSize() const {
	return fids_.size();
}
int FeaturesVector::nSamples() const {
	return ids_.rows();
}
float FeaturesVector::get(int s, int f) const {
	return features_[ ids_(s,0) ]->get( ids_(s,1), fids_[f] );
}
