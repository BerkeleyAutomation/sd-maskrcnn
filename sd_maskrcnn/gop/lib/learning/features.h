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
#include "util/eigen.h"
#include <memory>

class Features {
protected:
public:
	virtual ~Features();
	virtual int nSamples() const = 0;
	virtual int featureSize() const = 0;
	virtual float get( int s, int f ) const = 0;
	virtual RMatrixXf getFeatures() const;
	virtual RMatrixXf getFeatures( const VectorXi & samples ) const;
	virtual RMatrixXf getFeatures( const VectorXi & samples, const VectorXi & fids ) const;
};

class FeaturesMatrix: public Features {
protected:
	RMatrixXf data_;
public:
	FeaturesMatrix( const RMatrixXf & data );
	virtual int featureSize() const;
	virtual int nSamples() const;
	virtual float get(int s, int f) const;
};

class FeaturesVector: public Features {
protected:
	std::vector< std::shared_ptr< Features > > features_;
	RMatrixXi ids_;
	VectorXi fids_;
public:
	FeaturesVector( const std::vector< std::shared_ptr< Features > > & f, const RMatrixXi & ids, const VectorXi & fids=VectorXi() );
	virtual int featureSize() const;
	virtual int nSamples() const;
	virtual float get(int s, int f) const;
};