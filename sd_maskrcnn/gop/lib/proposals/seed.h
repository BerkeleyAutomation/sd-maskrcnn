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
#include "util/graph.h"
#include "util/eigen.h"
#include "imgproc/image.h"
#include "seedfeature.h"

class OverSegmentation;
class ImageOverSegmentation;
class SeedFunction {
public:
	virtual ~SeedFunction();
	virtual VectorXi compute( const OverSegmentation & os, int M ) const = 0;
	virtual std::shared_ptr<SeedFunction> clone() const = 0;
	virtual SeedFunction* clonePtr() const = 0;
	
	virtual void save( std::ostream & s ) const;
	virtual void load( std::istream & s );
};
template<typename T>
class ClonableSeedFunction: public SeedFunction {
public:
	virtual std::shared_ptr<SeedFunction> clone() const {
		return std::make_shared<T>( *dynamic_cast<const T*>(this) );
	}
	virtual SeedFunction* clonePtr() const {
		return new T( *dynamic_cast<const T*>(this) );
	}
};

class ImageSeedFunction: public SeedFunction {
protected:
	virtual VectorXi computeImageOverSegmentation( const ImageOverSegmentation & ios, int M ) const = 0;
public:
	virtual VectorXi compute( const OverSegmentation & ios, int M ) const;
};
template<typename T>
class ClonableImageSeedFunction: public ImageSeedFunction {
public:
	virtual std::shared_ptr<SeedFunction> clone() const {
		return std::make_shared<T>( *dynamic_cast<const T*>(this) );
	}
	virtual SeedFunction* clonePtr() const {
		return new T( *dynamic_cast<const T*>(this) );
	}
};

class RegularSeed: public ClonableImageSeedFunction<RegularSeed> {
protected:
	virtual VectorXi computeImageOverSegmentation( const ImageOverSegmentation & ios, int M ) const;
};

class SaliencySeed: public ClonableImageSeedFunction<SaliencySeed> {
protected:
	virtual VectorXi computeImageOverSegmentation( const ImageOverSegmentation & ios, int M ) const;
};

class GeodesicSeed: public ClonableSeedFunction<GeodesicSeed> {
protected:
	float pow_, const_w_, min_d_;
	virtual VectorXi compute( const OverSegmentation & ios, int M ) const;
public:
	GeodesicSeed( float pow=3, float const_w=2e-3, float min_d=0 );
	
	virtual void save( std::ostream & s ) const;
	virtual void load( std::istream & s );
};

class RandomSeed: public ClonableSeedFunction<RandomSeed> {
public:
	virtual VectorXi compute( const OverSegmentation & ios, int M ) const;
};

class SegmentationSeed: public ClonableSeedFunction<SegmentationSeed> {
public:
	virtual VectorXi compute( const OverSegmentation & ios, int M ) const;
};

class LearnedSeed: public ClonableImageSeedFunction<LearnedSeed> {
protected:
	SeedFeatureFactory features_;
	std::vector<VectorXf> w_;
	void train( std::vector<SeedFeatureVector> & f, const std::vector<VectorXs> & lbl, int max_feature, int n_seed_per_obj=1 );
	virtual VectorXi computeImageOverSegmentation( const ImageOverSegmentation & ios, int M ) const;
	void initFeatures();
public:
	LearnedSeed();
	void addSmallObjectMap( const VectorXf & map );
	void train( const std::vector< std::shared_ptr<ImageOverSegmentation> > & gops, const std::vector<VectorXs> & lbl, int max_feature, int n_seed_per_obj=1 );
	
	virtual void save( std::ostream & s ) const;
	virtual void load( std::istream & s );
	void save( const std::string & s ) const;
	void load( const std::string & s );
};

