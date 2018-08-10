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
#include "saliency.h"
#include "segmentation/segmentation.h"
#include "imgproc/color.h"

SaliencySettings::SaliencySettings() {
	// Saliency filter radii
	sigma_p_ = 0.25;
	sigma_c_ = 20.0;
	k_ = 3; // The paper states K=6 (but then I think we used standard deviation and not variance, was probably a typo)
	
	// Upsampling parameters
	min_saliency_ = 0.1;
	alpha_ = 1.0 / 30.0;
	beta_ = 1.0 / 30.0;
	
	// Various algorithm settings
	upsample_ = true;
	uniqueness_ = true;
	distribution_ = true;
	use_spix_color_ = false; // Disabled to get a slightly better performance
}

struct SuperpixelStatistic {
	Vector3f mean_color_;
	Vector3f mean_rgb_;
	Vector2f mean_position_;
	int size_;
	SuperpixelStatistic() {
		mean_color_ = Vector3f::Zero();
		mean_rgb_ = Vector3f::Zero();
		mean_position_ = Vector2f::Zero();
		size_ = 0;
	}
};

std::vector< SuperpixelStatistic > computeStat( const Image8u &im, const RMatrixXs &s ) {
	const int Ns = s.maxCoeff()+1;
	Image rgb_im = im, lab_im;
	rgb2lab( lab_im, rgb_im );
	
	std::vector< SuperpixelStatistic > stat( Ns );
	std::vector< double > cnt( Ns, 1e-10 );
	
	for( int j=0; j<rgb_im.H(); j++ )
		for( int i=0; i<rgb_im.W(); i++ ) {
			int l = s( j, i );
			if ( l >=0 ) {
				stat[ l ].mean_color_ += lab_im.at<3>(j,i);
				stat[ l ].mean_rgb_ += lab_im.at<3>(j,i);
				stat[ l ].mean_position_ += Vector2f( i, j );
				cnt[ l ] += 1;
			}
		}
	for( int i=0; i<Ns; i++ ) {
		stat[ i ].mean_color_ *= 1.0 / cnt[ i ];
		stat[ i ].mean_rgb_ *= 1.0 / cnt[ i ];
		stat[ i ].mean_position_ *= 1.0 / cnt[ i ];
		stat[ i ].size_ = cnt[ i ];
	}
	// Rescale the position parameter
	for( int i=0; i<Ns; i++ )
		stat[ i ].mean_position_ *= 1.0 / std::max( rgb_im.W(), rgb_im.H() );
	return stat;
}

Saliency::Saliency( SaliencySettings settings ): settings_(settings) {
}
VectorXf Saliency::saliency( const ImageOverSegmentation &s ) const {
	return saliency( s.image(), s.s() );
}
VectorXf Saliency::saliency( const Image8u &im, const RMatrixXs &s ) const {
	
	// Do the abstraction
	std::vector< SuperpixelStatistic > stat = computeStat( im, s );
	
	// Compute the uniqueness
	std::vector<float> unique( stat.size(), 1 );
	if (settings_.uniqueness_)
		unique = uniqueness( stat );
	
	// Compute the distribution
	std::vector<float> dist( stat.size(), 0 );
	if (settings_.distribution_)
		dist = distribution( stat );
	
	// Combine the two measures
	VectorXf r( stat.size() );
	for( int i=0; i<stat.size(); i++ )
		r[i] = unique[i] * exp( - settings_.k_ * dist[i] );
	
	// Rescale the saliency to [0..1]
	r = ( r.array() - r.minCoeff() ) / ( r.maxCoeff() - r.minCoeff() );
	
	// Increase the saliency value until we are below the minimal threshold
	double m_sal = settings_.min_saliency_ * r.size();
	for( float sm = r.array().sum(); sm < m_sal; sm = r.array().sum() )
		r =  (r.array()*(m_sal/sm)).min( 1.0f );
	return r;
}
// Normalize a vector of floats to the range [0..1]
void normVec( std::vector< float > &r ){
	const int N = r.size();
	float mn = r[0], mx = r[0];
	for( int i=1; i<N; i++ ) {
		if (mn > r[i])
			mn = r[i];
		if (mx < r[i])
			mx = r[i];
	}
	for( int i=0; i<N; i++ )
		r[i] = (r[i] - mn) / (mx - mn);
}
std::vector< float > Saliency::uniqueness( const std::vector< SuperpixelStatistic >& stat ) const {
	const int N = stat.size();
	std::vector< float > r( N );
	const float sp = 0.5 / (settings_.sigma_p_ * settings_.sigma_p_);
	for( int i=0; i<N; i++ ) {
		float u = 0, norm = 1e-10;
		Vector3f c = stat[i].mean_color_;
		Vector2f p = stat[i].mean_position_;
		
		// Evaluate the score, for now without filtering
		for( int j=0; j<N; j++ ) {
			Vector3f dc = stat[j].mean_color_ - c;
			Vector2f dp = stat[j].mean_position_ - p;
			
			float w = exp( - sp * dp.dot(dp) );
			u += w*dc.dot(dc);
			norm += w;
		}
		// Let's not normalize here, must have been a typo in the paper
// 		r[i] = u / norm;
		r[i] = u;
	}
	normVec( r );
	return r;
}
std::vector< float > Saliency::distribution( const std::vector< SuperpixelStatistic >& stat ) const {
	const int N = stat.size();
	std::vector< float > r( N );
	const float sc =  0.5 / (settings_.sigma_c_*settings_.sigma_c_);
	for( int i=0; i<N; i++ ) {
		float u = 0, norm = 1e-10;
		Vector3f c = stat[i].mean_color_;
		Vector2f p(0.f, 0.f);
		
		// Find the mean position
		for( int j=0; j<N; j++ ) {
			Vector3f dc = stat[j].mean_color_ - c;
			float w = exp( - sc * dc.dot(dc) );
			p += w*stat[j].mean_position_;
			norm += w;
		}
		p *= 1.0 / norm;
		
		// Compute the variance
		for( int j=0; j<N; j++ ) {
			Vector3f dc = stat[j].mean_color_ - c;
			Vector2f dp = stat[j].mean_position_ - p;
			float w = exp( - sc * dc.dot(dc) );
			u += w*dp.dot(dp);
		}
		r[i] = u / norm;
	}
	normVec( r );
	return r;
}
