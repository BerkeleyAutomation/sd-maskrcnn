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
#include "unary.h"
#include "segmentation/segmentation.h"
#include "imgproc/color.h"
#include "util/algorithm.h"
#include <iostream>
#include <fstream>

///////////////////////////////////////////
//////////// Unary and factory ////////////
///////////////////////////////////////////
Unary::~Unary() {
}
UnaryFactory::~UnaryFactory() {
}
bool UnaryFactory::isStatic() const {
	return false;
}

///////////////////////////////////////////
////////////// Static Unary ///////////////
///////////////////////////////////////////
class StaticUnary: public Unary {
protected:
	RMatrixXf r_;
public:
	StaticUnary(const RMatrixXf & r ):r_(r) {
	}
	virtual RMatrixXf compute( int seed ) const {
		return r_;
	}
};
class ConstantUnaryFactory: public UnaryFactory{
protected:
	float v_;
public:
	ConstantUnaryFactory( float v ) : v_( v ) {
	}
	virtual std::shared_ptr<Unary> create( const UnaryFeatures & f ) const {
		eassert( UnaryFeature::D( UnaryFeature::Constant ) == 1 );
		VectorXf boundary = f.get( UnaryFeature::Constant )->compute( 0 );
		return std::make_shared<StaticUnary>( v_ * boundary.array() );
	}
	FeatureSet requiredFeatures() const {
		return FeatureSet(UnaryFeature::Constant);
	}
	virtual bool isStatic() const {
		return true;
	}
	virtual int dim() const { return 1; }
};

///////////////////////////////////////////
////////////// Vector Unary ///////////////
///////////////////////////////////////////
class VectorUnary: public Unary {
protected:
	std::shared_ptr<UnaryFeature> features_;
	VectorXf w_;
public:
	VectorUnary(std::shared_ptr<UnaryFeature> features, const VectorXf & w ):features_(features), w_(w) {
		eassert( features_->dim() == w_.size() );
	}
	virtual RMatrixXf compute( int seed ) const {
		return features_->multiply( seed, w_ );
	}
};
class VectorUnaryFactory: public UnaryFactory {
protected:
	FeatureType t_;
	VectorXf w_;
public:
	VectorUnaryFactory(FeatureType t, const VectorXf & w ):t_(t), w_(w) {
		eassert( UnaryFeature::D(t) == w_.size() );
	}
	VectorUnaryFactory(FeatureType t, float scale=1.0 ):t_(t), w_(VectorXf::Constant(UnaryFeature::D(t),scale)) {
	}
	virtual std::shared_ptr<Unary> create( const UnaryFeatures & f ) const {
		if( UnaryFeature::isStatic( t_ ) )
			return std::make_shared<StaticUnary>( f.get(t_)->multiply( 0, w_ ) );
		return std::make_shared<VectorUnary>( f.get(t_), w_ );
	}
	virtual FeatureSet requiredFeatures() const {
		return FeatureSet(t_);
	}
	virtual bool isStatic() const {
		return UnaryFeature::isStatic( t_ );
	}
	virtual int dim() const { return 1; }
};

///////////////////////////////////////////
////////////// Vector Unary ///////////////
///////////////////////////////////////////
template<bool BINARY>
class VectorSetUnary: public Unary {
protected:
	UnaryFeatures features_;
	VectorXf w_;
	float b_;
public:
	VectorSetUnary(const UnaryFeatures & features, const VectorXf & w, float b ):features_(features), w_(w), b_(b) {
		eassert( features_.dim() == w_.size() );
	}
	virtual RMatrixXf compute( int seed ) const {
		int p=0;
		VectorXf r;
		for( auto f: features_.features() ) {
			if (!p)
				r = f->multiply( seed, w_.segment( p, f->dim() ) );
			else
				r += f->multiply( seed, w_.segment( p, f->dim() ) );
			p += f->dim();
		}
		r.array() += b_;
		if (BINARY) {
			if ((r.array()>=0).all())
				return r.array()*0;
			return (r.array()>=0).cast<float>() * 1e10;
		}
		return r;
	}
};
template<bool BINARY=false>
class VectorSetUnaryFactory: public UnaryFactory {
protected:
	FeatureSet t_;
	VectorXf w_;
	float b_;
public:
	VectorSetUnaryFactory(const FeatureSet & t, const VectorXf & w, float b=0 ):t_(t), w_(w), b_(b) {
		eassert( UnaryFeatures::D(t) == w_.size() );
	}
	VectorSetUnaryFactory(const FeatureSet & t, float scale=1.0 ):t_(t), w_(VectorXf::Constant(UnaryFeatures::D(t),scale)), b_(0) {
	}
	VectorSetUnaryFactory( const std::string & filename ) {
		std::ifstream is( filename, std::ios::in | std::ios::binary );
		if(!is.is_open())
			throw std::invalid_argument( "Could not open file '"+filename+"'!" );
		t_.load( is );
		loadMatrixX( is, w_ );
		is.read( (char*)&b_, sizeof(b_) );
		is.close();
	}
	void save( const std::string & filename ) const {
		std::ofstream os( filename, std::ios::out | std::ios::binary );
		if(!os.is_open())
			throw std::invalid_argument( "Could not write file '"+filename+"'!" );
		t_.save( os );
		saveMatrixX( os, w_ );
		os.write( (const char*)&b_, sizeof(b_) );
		os.close();
	}
	virtual std::shared_ptr<Unary> create( const UnaryFeatures & f ) const {
		return std::make_shared< VectorSetUnary<BINARY> >( f.subset(t_), w_, b_ );
	}
	virtual FeatureSet requiredFeatures() const {
		return t_;
	}
	virtual bool isStatic() const {
		return false;
	}
	virtual int dim() const { return 1; }
};

///////////////////////////////////////////
////////////// Bounary Unary //////////////
///////////////////////////////////////////

static int bitcount(unsigned int u) {
	unsigned int uCount;
	uCount = u - ((u >> 1) & 033333333333) - ((u >> 2) & 011111111111);
	return ((uCount + (uCount >> 3)) & 030707070707) % 63;
}
static VectorXb getBoundary( const VectorXi & s, int type ) {
	int strict = (int)(type > (int)BoundaryType::ALL_BND);
	if( strict )
		type /= (int)BoundaryType::ALL_BND+1;
	VectorXb r( s.size() );
	for( int i=0; i<s.size(); i++ )
		r[i] = (bitcount(s[i] & type) > strict);
	return r;
}
class BoundaryUnaryFactory: public UnaryFactory{
protected:
	std::vector<int> t_;
public:
	BoundaryUnaryFactory( const std::vector<int> & t ) : t_( t ) {
	}
	virtual std::shared_ptr<Unary> create( const UnaryFeatures & f ) const {
		eassert( UnaryFeature::D( UnaryFeature::BoundaryID ) == 1 );
		VectorXi boundary = f.get( UnaryFeature::BoundaryID )->compute( 0 ).cast<int>();
        RMatrixXf r = RMatrixXf::Zero( boundary.size(), t_.size() );
        for( int i=0; i<t_.size(); i++ )
			if( t_[i] )
				r.col(i) = 1e10*(1-getBoundary( boundary, t_[i] ).cast<float>().array());
		return std::make_shared<StaticUnary>( r );
	}
	FeatureSet requiredFeatures() const {
		return FeatureSet(UnaryFeature::BoundaryID);
	}
	virtual bool isStatic() const {
		return true;
	}
	virtual int dim() const { return t_.size(); }
};

std::shared_ptr<UnaryFactory> seedUnary() {
	return std::make_shared<VectorUnaryFactory>(UnaryFeature::InverseIndicator, 1e10);
}
std::shared_ptr<UnaryFactory> zeroUnary() {
	return std::make_shared<ConstantUnaryFactory>(0.f);
}
std::shared_ptr<UnaryFactory> rgbUnary( float scale ) {
	return std::make_shared<VectorUnaryFactory>(UnaryFeature::RGB, scale);
}
std::shared_ptr<UnaryFactory> labUnary( float scale ) {
	return std::make_shared<VectorUnaryFactory>(UnaryFeature::Lab, scale);
}
std::shared_ptr<UnaryFactory> learnedUnary( const std::string & filename ) {
	return std::make_shared<VectorSetUnaryFactory<false> >( filename );
}
std::shared_ptr<UnaryFactory> learnedUnary( const FeatureSet & features, const VectorXf & w, float b ){
	return std::make_shared<VectorSetUnaryFactory<false> >(features, w, b);
}
std::shared_ptr<UnaryFactory> learnedUnary( const VectorXf & w, float b ) {
	return learnedUnary( defaultUnaryFeatures(), w, b );
}
std::shared_ptr<UnaryFactory> binaryLearnedUnary( const std::string & filename ) {
	return std::make_shared<VectorSetUnaryFactory<true> >( filename );
}
std::shared_ptr<UnaryFactory> binaryLearnedUnary( const FeatureSet & features, const VectorXf & w, float b ){
	return std::make_shared< VectorSetUnaryFactory<true> >(features, w, b);
}
std::shared_ptr<UnaryFactory> binaryLearnedUnary( const VectorXf & w, float b ) {
	return binaryLearnedUnary( defaultUnaryFeatures(), w, b );
}
void saveLearnedUnary( const std::string & filename, std::shared_ptr<UnaryFactory> unary ) {
	if( std::dynamic_pointer_cast< VectorSetUnaryFactory<true> >( unary ) )
		std::dynamic_pointer_cast< VectorSetUnaryFactory<true> >( unary )->save( filename );
	else if( std::dynamic_pointer_cast< VectorSetUnaryFactory<false> >( unary ) )
		std::dynamic_pointer_cast< VectorSetUnaryFactory<false> >( unary )->save( filename );
	else
		throw std::invalid_argument( std::string("Cannot save unary of type '")+typeid( *unary.get() ).name()+"'!" );
}
std::shared_ptr<UnaryFactory> backgroundUnary( const std::vector<int> & t ) {
	return std::make_shared<BoundaryUnaryFactory>( t );
}
