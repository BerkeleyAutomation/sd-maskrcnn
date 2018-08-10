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
#include "edgefeature.h"
#include "util/util.h"
#include "imgproc/color.h"
#include "segmentation/segmentation.h"

//////////////////////// Edge Feature Feature ////////////////////////
class EdgeFeature::Feature {
public:
	virtual void compute( Ref<RMatrixXf> m, const ImageOverSegmentation & ios ) const = 0;
	virtual int dim() const = 0;
};
//////////////////////// Edge Feature ////////////////////////
EdgeFeature::EdgeFeature():dim_(0) {
}
RMatrixXf EdgeFeature::compute( const ImageOverSegmentation & ios ) const {
	const int N = ios.edges().size();
	RMatrixXf r( N, dim_ );
	int s=0;
	for( auto f: features_ ) {
		f->compute( r.block(0, s, N, f->dim()), ios );
		s += f->dim();
	}
	return r;
}
int EdgeFeature::dim() const {
	return dim_;
}
void EdgeFeature::addFeature( std::shared_ptr<EdgeFeature::Feature> f ) {
	dim_ += f->dim();
	features_.push_back( f );
}
//////////////////////// Various Edge Features ////////////////////////
class WeightedEdgeFeature: public EdgeFeature::Feature {
protected:
	float edge_w_, pow_, const_w_;
public:
	WeightedEdgeFeature( float edge_w, float pow, float const_w  ):edge_w_(edge_w),pow_(pow),const_w_(const_w){
	}
	void compute( Ref<RMatrixXf> r, const ImageOverSegmentation & ios ) const {
		r = edge_w_ * ios.edgeWeights().array().pow( pow_ ) + const_w_;
	}
	int dim() const {
		return 1;
	}
};
void EdgeFeature::addWeighted( float edge_w, float pow, float const_w ) {
	addFeature( std::make_shared< WeightedEdgeFeature >( edge_w, pow, const_w ) );
}

class ColorEdgeFeature: public EdgeFeature::Feature {
protected:
	bool lab_;
public:
	ColorEdgeFeature( bool lab  ):lab_(lab){
	}
	void compute( Ref<RMatrixXf> r, const ImageOverSegmentation & ios ) const {
		Image f_im;
		if( lab_ )
			rgb2lab( f_im, ios.image() );
		else
			f_im = ios.image();
		
		const int N = ios.Ns();
		std::vector<Vector3f> mean_color( N, Vector3f::Zero() );
		std::vector<float> cnt( N, 1e-8 );
		const RMatrixXs & s = ios.s();
		for( int j=0; j<s.rows(); j++ )
			for( int i=0; i<s.cols(); i++ ) 
				if( s(j,i) >= 0 ){
					mean_color[ s(j,i) ] += f_im.at<3>( j, i );
					cnt[ s(j,i) ] += 1;
				}
		for( int i=0; i<N; i++ )
			mean_color[i] /= cnt[i];
		
		for( int i=0; i<(int)ios.edges().size(); i++ )
			r.row(i) = (mean_color[ ios.edges()[i].a ]-mean_color[ ios.edges()[i].b ]).array().abs();
	}
	int dim() const {
		return 3;
	}
};
void EdgeFeature::addRGB() {
	addFeature( std::make_shared< ColorEdgeFeature >( false ) );
}
void EdgeFeature::addLAB() {
	addFeature( std::make_shared< ColorEdgeFeature >( true ) );
}

class EdgeLengthFeature: public EdgeFeature::Feature {
protected:
	int n_hist_, max_hist_;
public:
	EdgeLengthFeature( int n_hist=5, int max_hist=25  ):n_hist_(n_hist),max_hist_(max_hist){
	}
	void compute( Ref<RMatrixXf> r, const ImageOverSegmentation & ios ) const {
		std::unordered_map<Edge,int> elen;
		const RMatrixXs & s = ios.s();
		for( int j=0; j<s.rows(); j++ )
			for( int i=0; i<s.cols(); i++ ) {
				if( i && s(j,i) != s(j,i-1) )
					elen[ Edge( s(j,i), s(j,i-1) ) ]++;
				if( j && s(j,i) != s(j-1,i) )
					elen[ Edge( s(j,i), s(j-1,i) ) ]++;
			}
		r.setZero();
		for( int i=0; i<(int)ios.edges().size(); i++ ) {
			int l = elen[ ios.edges()[i] ];
			r(i,0) = std::max(0.01*l,1.0);
			r(i,1) = 1.0 / l;
			if( l < max_hist_ )
				r(i,2+l*n_hist_/max_hist_)=1;
		}
	}
	int dim() const {
		return 2+n_hist_;
	}
};
void EdgeFeature::addLength( int n_hist, int max_hist ) {
	addFeature( std::make_shared< EdgeLengthFeature >( n_hist, max_hist ) );
}


