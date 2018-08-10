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
#include <vector>
#include <memory>

class ImageOverSegmentation;
class SeedFeature {
protected:
	friend class SeedFeatureVector;
	RMatrixXf f_;
public:
	SeedFeature(){}
	SeedFeature( const RMatrixXf & f ):f_(f){}
	virtual ~SeedFeature();
	operator const RMatrixXf&() const {
		return f_;
	}
	virtual bool update( int n ) {
		return false;
	}
	virtual int dim() const = 0;
};

class SeedFeatureVector: public SeedFeature {
	friend class SeedFeatureFactory;
protected:
	int dim_;
	std::vector< std::shared_ptr<SeedFeature> > features_;
	SeedFeatureVector( const std::vector< std::shared_ptr<SeedFeature> > & features );
public:
	virtual bool update( int n );
	virtual int dim() const;
};

class SeedFeatureFactory {
public:
	class Creator;
protected:
	std::vector< std::shared_ptr<Creator> > creator_;
public:
	// Create an instantiation
	SeedFeatureVector create( const ImageOverSegmentation & ios ) const;
	void clear();
	
	// Add different features
	void addPosition();
	void addColor();
	void addGeodesic( float edge_w=1.0, float edge_p=1.0, float const_w=0.0 );
	void addGeodesicBnd( float edge_w=1.0, float edge_p=1.0, float const_w=0.0 );
};
