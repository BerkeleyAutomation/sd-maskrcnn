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
#include "unaryfeature.h"
#include "geodesics.h"
#include "unary.h"
#include "imgproc/color.h"
#include "util/util.h"
#include "util/algorithm.h"
#include "segmentation/segmentation.h"

////////////////////////////////////////////
/////////////// Feature Set ////////////////
////////////////////////////////////////////
FeatureSet::FeatureSet():features_(0){
}
FeatureSet::FeatureSet( FeatureType t ) : features_( 1<<t ) {
}
void FeatureSet::add( const FeatureType & t ) {
	features_ |= (1<<t);
}
void FeatureSet::add( const FeatureSet & s ) {
	features_ |= s.features_;
}
void FeatureSet::remove( const FeatureType & t ) {
	features_ &=~(1<<t);
}
bool FeatureSet::has( const FeatureType & t ) const {
	return features_ & (1<<t);
}
void FeatureSet::load( std::istream & is ) {
	is.read( (char*)&features_, sizeof(features_) );
}
void FeatureSet::save( std::ostream & os ) const {
	os.write( (const char*)&features_, sizeof(features_) );
}
////////////////////////////////////////////
////////// Unary Feature Factory ///////////
////////////////////////////////////////////
struct UnaryFeatureFactory {
	struct Creator {
		virtual std::shared_ptr<UnaryFeature> create( const ImageOverSegmentation & ios ) const = 0;
		virtual int dim() const = 0;
		virtual bool isStatic() const = 0;
	};
	std::vector< std::shared_ptr<Creator> > creators_;
	FeatureType addCreator( std::shared_ptr<Creator> c ) {
		creators_.push_back( c );
		return (creators_.size()-1);
	};
	std::shared_ptr<UnaryFeature> create( const ImageOverSegmentation & ios, const FeatureType & t ) const {
		for( FeatureType i=0; i<(FeatureType)creators_.size(); i++ )
			if (i == t)
				return creators_[i]->create( ios );
		throw std::invalid_argument("Invalid unary feature id!");
	}
	std::vector< std::shared_ptr<UnaryFeature> > create( const ImageOverSegmentation & ios, const FeatureSet & s ) const {
		std::vector< std::shared_ptr<UnaryFeature> > r;
		for( FeatureType i=0; i<(FeatureType)creators_.size(); i++ )
			if ( s.has(i) )
				r.push_back( creators_[i]->create( ios ) );
		return r;
	}
	bool isStatic( const FeatureType & t ) {
		if( t < creators_.size() )
			return creators_[t]->isStatic();
		throw std::invalid_argument("Invalid unary feature id!");
	}
	int dimension( const FeatureType & t ) {
		if( t < creators_.size() )
			return creators_[t]->dim();
		throw std::invalid_argument("Invalid unary feature id!");
	}
	int dimension( const FeatureSet & t ) {
		int r=0;
		for( int i=0; i<(int)creators_.size(); i++ )
			if( t.has( i ) )
				r += creators_[i]->dim();
		return r;
	}
};
static UnaryFeatureFactory unary_feature_factory;
template<typename T>
struct TypedUnaryCreator: public UnaryFeatureFactory::Creator {
	virtual std::shared_ptr<UnaryFeature> create( const ImageOverSegmentation & ios ) const {
		return std::make_shared<T>( ios );
	}
	virtual int dim() const{
		return T::D();
	}
	virtual bool isStatic() const{
		return T::isStatic();
	}
};
#define REGISTER_UNARY_FEATURE( T ) FeatureType T##UnaryFeature::featureId() const{ return UnaryFeature::T; } FeatureType UnaryFeature::T = unary_feature_factory.addCreator( std::make_shared< TypedUnaryCreator<T##UnaryFeature> >() )

////////////////////////////////////////////
////////////// Unary Features //////////////
////////////////////////////////////////////
UnaryFeatures::UnaryFeatures(){
}
UnaryFeatures::UnaryFeatures( const ImageOverSegmentation &ios, const FeatureSet &features ) {
	features_ = unary_feature_factory.create( ios, features );
}
UnaryFeatures UnaryFeatures::subset( const FeatureSet &t ) const {
	UnaryFeatures r;
	for( auto f : features_ )
		if( t.has( f->featureId() ) )
			r.features_.push_back( f );
	return r;
}
std::shared_ptr< UnaryFeature > UnaryFeatures::get( FeatureType t ) const {
	for( auto f : features_ )
		if( t == f->featureId() )
			return f;
	throw std::invalid_argument( "Feature type t="+std::to_string(t)+" not found" );
}
int UnaryFeatures::D( const FeatureSet &f ) {
	return unary_feature_factory.dimension( f );
}
int UnaryFeatures::dim() const {
	int r=0;
	for( auto f : features_ )
		r += f->dim();
	return r;
}
RMatrixXf UnaryFeatures::compute( int seed ) const {
	RMatrixXf r( features_[0]->N(), dim() );
	int p=0;
	for( auto f: features_ ) {
		r.middleCols( p, f->dim() ) = f->compute( seed );
		p += f->dim();
	}
	return r;
}

////////////////////////////////////////////
////////////// Unary Feature ///////////////
////////////////////////////////////////////
UnaryFeature::UnaryFeature() {
}
UnaryFeature::~UnaryFeature() {
}
RMatrixXf UnaryFeature::compute( int seed ) const {
	RMatrixXf r = RMatrixXf::Zero( N(), dim() );
	compute( r, seed );
	return r;
}
VectorXf UnaryFeature::multiply( int seed, const VectorXf & w ) const {
	return compute( seed ) * w;
}
bool UnaryFeature::isStatic( const FeatureType &t ) {
	return unary_feature_factory.isStatic( t );
}
int UnaryFeature::D( const FeatureType &t ) {
	return unary_feature_factory.dimension( t );
}

/////////////////////////////////////////
/////////// Constant Feature ////////////
/////////////////////////////////////////
class ConstantUnaryFeature:public UnaryFeature {
	const int N_;
public:
	ConstantUnaryFeature( const ImageOverSegmentation & ios ):N_(ios.Ns()){
	}
	virtual void compute( Ref<RMatrixXf> f, int seed) const {
		f.setOnes();
	}
	static int D() { return 1; }
	virtual int dim() const { return D(); }
	virtual int N() const { return N_; }
	static bool isStatic() { return true; }
	virtual FeatureType featureId() const;
};
REGISTER_UNARY_FEATURE( Constant );

/////////////////////////////////////////////
///////////// Indicator Feature /////////////
/////////////////////////////////////////////
class IndicatorUnaryFeature:public UnaryFeature {
	const int N_;
public:
	IndicatorUnaryFeature( const ImageOverSegmentation & ios ):N_(ios.Ns()){
	}
	void compute( Ref<RMatrixXf> f, int seed) const {
		f.setZero();
		f(seed,0) = 1;
	}
	static int D() { return 1; }
	virtual int dim() const { return D(); }
	virtual int N() const { return N_; }
	static bool isStatic() { return false; }
	virtual FeatureType featureId() const;
};
REGISTER_UNARY_FEATURE( Indicator );

class InverseIndicatorUnaryFeature:public UnaryFeature {
	const int N_;
public:
	InverseIndicatorUnaryFeature( const ImageOverSegmentation & ios ):N_(ios.Ns()){
	}
	void compute( Ref<RMatrixXf> f, int seed) const {
		f.setOnes();
		f(seed,0) = 0;
	}
	static int D() { return 1; }
	virtual int dim() const { return D(); }
	virtual int N() const { return N_; }
	static bool isStatic() { return false; }
	virtual FeatureType featureId() const;
};
REGISTER_UNARY_FEATURE( InverseIndicator );

////////////////////////////////////////////
///////////// Position Feature /////////////
////////////////////////////////////////////
class PositionUnaryFeature: public UnaryFeature {
protected:
	RMatrixXf pos_;
public:
	PositionUnaryFeature( const ImageOverSegmentation & ios ) {
		const RMatrixXs & s = ios.s();
		VectorXf cnt = 1e-3 * VectorXf::Ones( ios.Ns() );
		RMatrixXf cm = RMatrixXf::Zero( ios.Ns(), dim() );
		for( int j = 0; j < s.rows(); j++ )
			for( int i = 0; i < s.cols(); i++ ) {
				const float x = 1.0 * i / ( s.cols() - 1 ) - 0.5, y = 1.0 * j / ( s.rows() - 1 ) - 0.5;
				const int id = s( j, i );
				cnt[ id ]++;
				// X and Y position
				cm( id, 0 ) += x;
				cm( id, 1 ) += y;
				// Distance to bnd
				cm( id, 2 ) += fabs( x );
				cm( id, 3 ) += fabs( y );
			}
		pos_ = cnt.cwiseInverse().asDiagonal() * cm;
	}
	void compute( Ref < RMatrixXf > f, int seed ) const {
		f.col( 0 ) = pos_.col( 0 ).array() - pos_( seed, 0 );
		f.col( 1 ) = pos_.col( 1 ).array() - pos_( seed, 1 );

		f.col( 2 ) = f.col( 0 ).array().abs();
		f.col( 3 ) = f.col( 1 ).array().abs();

		f.col( 4 ) = pos_.col( 2 );
		f.col( 5 ) = pos_.col( 3 );

		f.col( 6 ) = pos_.col( 2 ).array() - pos_( seed, 2 );
		f.col( 7 ) = pos_.col( 3 ).array() - pos_( seed, 3 );
	}
	static int D() { return 8; }
	virtual int dim() const { return D(); }
	virtual int N() const { return pos_.rows(); }
	static bool isStatic() { return false; }
	virtual FeatureType featureId() const;
};
REGISTER_UNARY_FEATURE( Position );


/////////////////////////////////////////
///////////// Color Feature /////////////
/////////////////////////////////////////
class RGBUnaryFeature: public UnaryFeature {
protected:
	RMatrixXf mean_;
	RGBUnaryFeature(){}
public:
	RGBUnaryFeature( const ImageOverSegmentation & ios ) {
		Image fim = ios.image();
		mean_ = ios.project( fim, "mean" );
	}
	void compute( Ref < RMatrixXf > f, int seed ) const {
		f = ( mean_.rowwise() - mean_.row( seed ) ).array().square().matrix();
	}
	static int D() { return 3; }
	virtual int dim() const { return D(); }
	virtual int N() const { return mean_.rows(); }
	static bool isStatic() { return false; }
	virtual FeatureType featureId() const;
};
REGISTER_UNARY_FEATURE( RGB );

class LabUnaryFeature: public RGBUnaryFeature {
public:
	LabUnaryFeature( const ImageOverSegmentation & ios ) {
		Image fim;
		rgb2lab( fim, ios.image() );
		mean_ = ios.project( fim, "mean" );
	}
	virtual FeatureType featureId() const;
};
REGISTER_UNARY_FEATURE( Lab );

//////////////////////////////////////////////
///////////// Color Hist Feature /////////////
//////////////////////////////////////////////
const int N_COLOR_CLUSTER = 30;

class RGBHistogramUnaryFeature: public UnaryFeature {
protected:
	RMatrixXf chist_;
	RGBHistogramUnaryFeature(){}
	void init( const ImageOverSegmentation & ios, const Image & fim ) {
		std::mt19937 rand;
		const int N_SAMPLES = 1000;

		// Collect some color samples
		RMatrixXf samples( N_SAMPLES, 3 );
		for( int i = 0; i < N_SAMPLES; i++ )
			samples.row( i ) = fim.at<3>( rand() % fim.H(), rand() % fim.W() );

		// And run K-means
		RMatrixXf means = kmeans( samples, N_COLOR_CLUSTER, 1 );

		// Build a color histogram
		chist_ = RMatrixXf::Zero( ios.Ns(), N_COLOR_CLUSTER );
		const RMatrixXs & s = ios.s();
		for( int j = 0; j < fim.H(); j++ )
			for( int i = 0; i < fim.W(); i++ ) {
				VectorXf d = ( means.rowwise() - fim.at<3>( j, i ).transpose() ).rowwise().squaredNorm();
				// Hard binning (Maybe we should try soft binning too [normalized inverse distance or something])
				int h;
				d.minCoeff( &h );
				chist_( s( j, i ), h ) += 1;
			}
		chist_.array().colwise() /= chist_.array().rowwise().sum();
	}
public:
	RGBHistogramUnaryFeature( const ImageOverSegmentation & ios ) {
		init( ios, ios.image() );
	}
	void compute( Ref < RMatrixXf > f, int seed ) const {
		RowVectorXf c = chist_.row( seed );
		// Compute the chi2 distance
		f = (chist_.rowwise() - c).array().square() / (chist_.array().rowwise() + c.array()+1e-10);
	}
	static int D() { return N_COLOR_CLUSTER; }
	virtual int dim() const { return D(); }
	virtual int N() const { return chist_.rows(); }
	static bool isStatic() { return false; }
	virtual FeatureType featureId() const;
};
REGISTER_UNARY_FEATURE( RGBHistogram );

class LabHistogramUnaryFeature: public RGBHistogramUnaryFeature {
public:
	LabHistogramUnaryFeature( const ImageOverSegmentation & ios ) {
		Image fim;
		rgb2lab( fim, ios.image() );
		init( ios, fim );
	}
	virtual FeatureType featureId() const;
};
REGISTER_UNARY_FEATURE( LabHistogram );

///////////////////////////////////////////
//////// Bounary Indicator Feature ////////
///////////////////////////////////////////
static int bitcount(unsigned int u) {
	unsigned int uCount;
	uCount = u - ((u >> 1) & 033333333333) - ((u >> 2) & 011111111111);
	return ((uCount + (uCount >> 3)) & 030707070707) % 63;
}
VectorXb getBoundary( const VectorXi & s, int type ) {
	int strict = (int)(type > (int)BoundaryType::ALL_BND);
	if( strict )
		type /= (int)BoundaryType::ALL_BND+1;
	VectorXb r( s.size() );
	for( int i=0; i<s.size(); i++ )
		r[i] = (bitcount(s[i] & type) > strict);
	return r;
}
VectorXi boundarySegments( const RMatrixXs & s ) {
	int Ns = s.maxCoeff()+1;
	VectorXi r = VectorXi::Zero( Ns );
	for( int w=0; w<4; w++ ) {
		for( int i=0; i<s.cols(); i++ ) r[ s(w,i) ] |= BoundaryType::TOP;
		for( int i=0; i<s.cols(); i++ ) r[ s(s.rows()-1-w,i) ] |= BoundaryType::BOTTOM;
		for( int i=0; i<s.rows(); i++ ) r[ s(i,w) ] |= BoundaryType::LEFT;
		for( int i=0; i<s.rows(); i++ ) r[ s(i,s.cols()-1-w) ] |= BoundaryType::RIGHT;
	}
	return r;
}
class BoundaryIndicatorUnaryFeature: public UnaryFeature {
protected:
	VectorXf bnd_;
public:
	BoundaryIndicatorUnaryFeature( const ImageOverSegmentation & ios ) {
		bnd_ = (boundarySegments( ios.s() ).array()>0).cast<float>();
	}
	virtual void compute( Ref < RMatrixXf > f, int seed ) const {
		f = bnd_;
	}
	static int D() { return 1; }
	virtual int dim() const { return D(); }
	virtual int N() const { return bnd_.size(); }
	static bool isStatic() { return true; }
	virtual FeatureType featureId() const;
};
REGISTER_UNARY_FEATURE( BoundaryIndicator );

///////////////////////////////////////////
/////////// Bounary ID Feature ////////////
///////////////////////////////////////////
class BoundaryIDUnaryFeature: public UnaryFeature {
protected:
	VectorXf bnd_;
public:
	BoundaryIDUnaryFeature( const ImageOverSegmentation & ios ) {
		bnd_ = boundarySegments( ios.s() ).cast<float>();
	}
	virtual void compute( Ref < RMatrixXf > f, int seed ) const {
		f = bnd_;
	}
	static int D() { return 1; }
	virtual int dim() const { return D(); }
	virtual int N() const { return bnd_.size(); }
	static bool isStatic() { return true; }
	virtual FeatureType featureId() const;
};
REGISTER_UNARY_FEATURE( BoundaryID );

///////////////////////////////////////////
//////// Bounary Distance Feature /////////
///////////////////////////////////////////
const int N_BND_HIST = 4, HIST_T= 2;

class BoundaryDistanceUnaryFeature: public UnaryFeature {
protected:
	RMatrixXf f_;
public:
	BoundaryDistanceUnaryFeature( const ImageOverSegmentation & ios ) {
		ArrayXi bnd = boundarySegments( ios.s() );
		GeodesicDistance gdist( ios.edges(), VectorXf::Ones(ios.edges().size()) );
		
		f_ = RMatrixXf::Zero( ios.Ns(), 4*(N_BND_HIST+1) );
		for( int k=0; k<4; k++ )
			f_.col(k) = gdist.compute( (VectorXf)((1-getBoundary(bnd,1<<k).cast<float>().array())*1e10) );
		for( int i=0; i<N_BND_HIST; i++ )
			f_.middleCols( 4*(i+1), 4 ) = (f_.leftCols(4).array() <= i*HIST_T).cast<float>();
	}
	virtual void compute( Ref < RMatrixXf > f, int seed ) const {
		f = f_;
	}
	static int D() { return 4*(N_BND_HIST+1); }
	virtual int dim() const { return D(); }
	virtual int N() const { return f_.rows(); }
	static bool isStatic() { return true; }
	virtual FeatureType featureId() const;
};
REGISTER_UNARY_FEATURE( BoundaryDistance );


FeatureSet defaultUnaryFeatures() {
	FeatureSet r;
	r.add( UnaryFeature::Constant );
	r.add( UnaryFeature::Indicator );
	r.add( UnaryFeature::Position );
	r.add( UnaryFeature::RGB );
	r.add( UnaryFeature::Lab );
	r.add( UnaryFeature::LabHistogram );
	r.add( UnaryFeature::BoundaryIndicator );
	return r;
}
