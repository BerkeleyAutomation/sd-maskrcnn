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
#include "imgproc/image.h"
#include "util/graph.h"
#include "util/eigen.h"
#include <vector>
#include <memory>

///// Feature Type and Set /////
typedef uint32_t FeatureType;
class FeatureSet {
protected:
	FeatureType features_;
public:
	FeatureSet();
	explicit FeatureSet( FeatureType t );
	void add( const FeatureType & t );
	void add( const FeatureSet & t );
	void remove( const FeatureType & t );
	bool has( const FeatureType & t ) const;
	void load( std::istream & is );
	void save( std::ostream & os ) const;
};
FeatureSet defaultUnaryFeatures();
///// Unary Features /////
class ImageOverSegmentation;
class UnaryFeature {
public:
	UnaryFeature();
	virtual ~UnaryFeature();
	virtual void compute( Ref<RMatrixXf> f, int seed ) const = 0;
	virtual RMatrixXf compute( int seed ) const;
	virtual VectorXf multiply( int seed, const VectorXf & w ) const;
	virtual FeatureType featureId() const = 0;
	virtual int dim() const = 0;
	virtual int N() const = 0;
	static int D( const FeatureType & t );
	static bool isStatic( const FeatureType & t );
	static FeatureType Constant;
	static FeatureType Indicator;
	static FeatureType InverseIndicator;
	static FeatureType Position;
	static FeatureType RGB;
	static FeatureType Lab;
	static FeatureType RGBHistogram;
	static FeatureType LabHistogram;
	static FeatureType BoundaryIndicator;
	static FeatureType BoundaryID;
	static FeatureType BoundaryDistance;
};
struct BoundaryType {
	static const int TOP=1;
	static const int BOTTOM=2;
	static const int LEFT=4;
	static const int RIGHT=8;
	static const int ALL_BND=15;
	static const int TOP_LEFT=16;
	static const int TOP_RIGHT=32;
	static const int BOTTOM_LEFT=64;
	static const int BOTTOM_RIGHT=128;
};
class UnaryFeatures {
protected:
	std::vector< std::shared_ptr<UnaryFeature> > features_;
	UnaryFeatures();
public:
	UnaryFeatures( const ImageOverSegmentation & ios, const FeatureSet & features = defaultUnaryFeatures() );
	UnaryFeatures subset( const FeatureSet & t ) const;
	std::shared_ptr<UnaryFeature> get( FeatureType t ) const;
	const std::vector< std::shared_ptr<UnaryFeature> > & features() const{ return features_; }
	int dim() const;
	static int D( const FeatureSet & f );
	RMatrixXf compute( int seed ) const;
};


