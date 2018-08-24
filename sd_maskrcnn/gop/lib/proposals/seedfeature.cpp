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
#include "seedfeature.h"
#include "geodesics.h"
#include "util/util.h"
#include "segmentation/segmentation.h"
#include "imgproc/color.h"
#include <tuple>
#include <queue>
#include <iostream>

//////////////////////// Seed Feature ////////////////////////
SeedFeature::~SeedFeature() {
}

//////////////////////// Seed Feature Factory ////////////////////////
class SeedFeatureFactory::Creator {
public:
	virtual std::shared_ptr<SeedFeature> create( const ImageOverSegmentation & ios ) const = 0;
};
template<typename T, typename ...ARGS>
class TypedSeedCreator: public SeedFeatureFactory::Creator {
protected:
	std::tuple< ARGS... > params_;
public:
	TypedSeedCreator( ARGS... args ):params_(std::make_tuple( args...)) {}
	
	virtual std::shared_ptr<SeedFeature> create( const ImageOverSegmentation & ios ) const {
		return make_shared_from_tuple<T>( std::tuple_cat( std::make_tuple( ios ), params_ ) );
	}
};
template<typename T>
class TypedSeedCreator<T>: public SeedFeatureFactory::Creator {
public:
	virtual std::shared_ptr<SeedFeature> create( const ImageOverSegmentation & ios ) const {
		return std::make_shared<T>( ios );
	}
};
SeedFeatureVector SeedFeatureFactory::create( const ImageOverSegmentation & ios ) const {
	std::vector< std::shared_ptr<SeedFeature> > features;
	for( auto c: creator_ )
		features.push_back( c->create( ios ) );
	return SeedFeatureVector( features );
}
template<typename T, typename ...ARGS>
std::shared_ptr< TypedSeedCreator<T,ARGS...> > make_seed_creator( ARGS... args ) {
	return std::make_shared< TypedSeedCreator<T,ARGS...> >( args... );
}
template<typename T>
std::shared_ptr< TypedSeedCreator<T> > make_seed_creator( ) {
	return std::make_shared< TypedSeedCreator<T> >( );
}
void SeedFeatureFactory::clear() {
	creator_.clear();
}

//////////////////////// Seed Feature Vector ////////////////////////
SeedFeatureVector::SeedFeatureVector( const std::vector< std::shared_ptr<SeedFeature> > & features ):features_(features) {
	// Get the matrix dims
	dim_ = 0;
	int l = 0;
	for( auto f: features_ ) {
		dim_ += f->dim();
		l = f->f_.rows();
	}
	// Get the initial features
	f_ = RMatrixXf::Zero( l, dim_ );
	int s=0;
	for( auto f: features_ ) {
		f_.block(0, s, f_.rows(), f->dim()) = f->f_;
		s += f->dim();
	}
}
bool SeedFeatureVector::update(int n) {
	bool r = false;
	int s = 0;
	for (auto f : features_) {
		if (f->update(n)) {
			f_.block(0, s, f_.rows(), f->dim()) = f->f_;
			r = true;
		}
		s += f->dim();
	}
	return r;
}
int SeedFeatureVector::dim() const {
	return dim_;
}

//////////////////////// Helper functions ////////////////////////
static VectorXb findBoundary( const RMatrixXs & s ) {
	int Ns = s.maxCoeff()+1;
	VectorXb r = VectorXb::Zero( Ns );
	for( int i=0; i<s.cols(); i++ ) r[ s(0,i) ] = 1;
	for( int i=0; i<s.cols(); i++ ) r[ s(s.rows()-1,i) ] = 1;
	for( int i=0; i<s.rows(); i++ ) r[ s(i,0) ] = 1;
	for( int i=0; i<s.rows(); i++ ) r[ s(i,s.cols()-1) ] = 1;
	return r;
}

//////////////////////// Various Seed Features ////////////////////////
class PositionSeedFeature: public SeedFeature {
public:
	PositionSeedFeature( const ImageOverSegmentation & ios ){
		VectorXf cnt = 1e-3*VectorXf::Ones( ios.Ns() );
		RMatrixXf cm = RMatrixXf::Zero( ios.Ns(), dim() );
		const RMatrixXs & s = ios.s();
		for( int j=0; j<s.rows(); j++ )
			for( int i=0; i<s.cols(); i++ ) {
				const float x = 1.0*i/(s.cols()-1)-0.5, y = 1.0*j/(s.rows()-1)-0.5;
				const int id = s(j,i);
				cnt[ id ]++;
				// Quadratic function for image location
				cm( id, 0 ) += x;
				cm( id, 1 ) += y;
				cm( id, 2 ) += x*x;
				cm( id, 3 ) += y*y;
				// Distance to bnd
				cm( id, 4 ) += fabs(x);
				cm( id, 5 ) += fabs(y);
			}
		f_ = cnt.cwiseInverse().asDiagonal() * cm;
	}
	virtual int dim() const {
		return 6;
	}
};
void SeedFeatureFactory::addPosition() {
	creator_.push_back( make_seed_creator< PositionSeedFeature >( ) );
}

class ColorSeedFeature: public SeedFeature {
protected:
	std::vector<Vector2f> location_;
	std::vector<RowVectorXf> color_;
	RMatrixXf var_, min_dist_;
	float n_;
public:
	ColorSeedFeature( const ImageOverSegmentation & ios ) {
		Image rgb_im = ios.image(), lab_im;
		rgb2lab( lab_im, rgb_im );
		const RMatrixXs & s = ios.s();
		
		const int Ns = ios.Ns();
		VectorXf cnt = 1e-5*VectorXf::Ones( Ns );
		RMatrixXf sm = RMatrixXf::Zero( Ns, 2+3+3 );
		
		for( int j=0; j<s.rows(); j++ )
			for( int i=0; i<s.cols(); i++ ) {
				const int id = s(j,i);
				cnt[ id ]++;
				// Location
				sm( id, 0 ) += 1.0*i/(s.cols()-1)-0.5;
				sm( id, 1 ) += 1.0*j/(s.rows()-1)-0.5;
				// RGB color
				sm( id, 2 ) += rgb_im(j,i,0);
				sm( id, 3 ) += rgb_im(j,i,1);
				sm( id, 4 ) += rgb_im(j,i,2);
				// LAB color
				sm( id, 5 ) += lab_im(j,i,0);
				sm( id, 6 ) += lab_im(j,i,1);
				sm( id, 7 ) += lab_im(j,i,2);
			}
		sm = cnt.cwiseInverse().asDiagonal() * sm;
		for( int i=0; i<Ns; i++ ) {
			location_.push_back( Vector2f( sm(i,0), sm(i,1) ) );
			color_.push_back( sm.block(i,2,1,6) );
		}
		f_ = RMatrixXf::Zero( Ns, dim() );
		var_ = RMatrixXf::Zero( Ns, 6 );
		min_dist_ = 10*RMatrixXf::Ones( Ns, 5 );
		n_ = 0;
	}
	virtual bool update( int n ) {
		const float loc_w = 1.0;
		
		RowVectorXf col = color_[n];
		for( int i=0; i<color_.size(); i++ ) {
			RowVectorXf cd = (color_[i]-col).array().square().matrix();
			var_.row(i) += cd;
			float c1 = sqrt(cd.leftCols(3).sum()), c2 = sqrt(cd.rightCols(3).sum()), d = (location_[n]-location_[i]).norm();
			min_dist_(i,0) = std::min( min_dist_(i,0), c1 );
			min_dist_(i,1) = std::min( min_dist_(i,1), c2 );
			min_dist_(i,2) = std::min( min_dist_(i,2), c1+loc_w*d );
			min_dist_(i,3) = std::min( min_dist_(i,3), c2+loc_w*d );
			min_dist_(i,4) = std::min( min_dist_(i,4), d );
		}
		n_++;
		f_.leftCols(6) = var_ / n_;
		f_.rightCols(5) = min_dist_;
		return true;
	}
	virtual int dim() const {
		return 11;
	}
};
void SeedFeatureFactory::addColor() {
	creator_.push_back( make_seed_creator< ColorSeedFeature >( ) );
}

class GeodesicSeedFeature: public SeedFeature {
protected:
	int Ns_;
	GeodesicDistance gdist;
public:
	GeodesicSeedFeature( const ImageOverSegmentation & ios, float edge_w, float edge_p, float const_w ):Ns_(ios.Ns()),gdist(ios.edges(),edge_w*ios.edgeWeights().array().pow(edge_p)+const_w){
		// Initialize with 0s (since large values can mess up the gradient computation due to cancelation)
		f_ = 1.*gdist.d();
	}
	virtual bool update( int n ) {
		f_ = gdist.update(n).d();
		return true;
	}
	virtual int dim() const {
		return 1;
	}
};
void SeedFeatureFactory::addGeodesic(float edge_w, float edge_p, float const_w) {
	creator_.push_back( make_seed_creator< GeodesicSeedFeature >( edge_w, edge_p, const_w ) );
}

class GeodesicBndSeedFeature: public SeedFeature {
public:
	GeodesicBndSeedFeature( const ImageOverSegmentation & ios, float edge_w, float edge_p, float const_w ){
		GeodesicDistance gdist( ios.edges(), edge_w*ios.edgeWeights().array().pow(edge_p)+const_w );
		f_ = gdist.compute( (VectorXf)(1e10*(VectorXf::Ones( ios.Ns() )-findBoundary( ios.s() ).cast<float>())) );
	}
	virtual int dim() const {
		return 1;
	}
};
void SeedFeatureFactory::addGeodesicBnd(float edge_w, float edge_p, float const_w) {
	creator_.push_back( make_seed_creator< GeodesicBndSeedFeature >( edge_w, edge_p, const_w ) );
}
