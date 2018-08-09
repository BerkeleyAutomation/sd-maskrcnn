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
#include "sketchtokens.h"
#include "imgproc/color.h"
#include "imgproc/filter.h"
#include "imgproc/nms.h"
#include "imgproc/gradient.h"
#include "imgproc/resample.h"
#include <fstream>
#include <cstdio>

SketchTokens::SketchTokens(int stride, int suppress, int nms):stride_(stride),suppress_(suppress),nms_(nms) {
}
RMatrixXf SketchTokens::detect( const Image8u & im ) const {
	STFeatures f( im, stride_ );
	VectorXf p = forest_.predictProb( f );
	Map<RMatrixXf> pp( p.data(), (im.H()+stride_-1)/stride_, (im.W()+stride_-1)/stride_ );
	return upsampleLinear( pp, im.W(), im.H(), stride_ );
}
RMatrixXf SketchTokens::filter(const RMatrixXf &detection, int suppress, int nms) const {
	if( suppress==-1 ) suppress = suppress_;
	if( nms==-1 ) nms = nms_;
	
	RMatrixXf r = detection;
	if( nms > 0 )
		r = ::nms( r, nms );
	if(suppress>0)
		suppressBnd( r, suppress );
	return r.array().min(1-1e-10).max(1e-10);
}
RMatrixXf SketchTokens::filter(const RMatrixXf &detection) const {
	return filter( detection, -1, -1 );
}

void SketchTokens::load(const std::string &fn) {
	forest_.load( fn );
}
STFeatures::STFeatures(const Image8u & im, int stride)
{
	const int R = 17, ssim_offset = 6;
	// Define some magic constants
#define N_FEATURE_SCALES 3
	const float feature_scales[N_FEATURE_SCALES] = {0,1.5,5};
	const int n_orientations[N_FEATURE_SCALES]   = {4,4,0};
	const int chns_smooth = 2;
	const int norm_rad = 5;
	const float norm_const = 0.01;
	const int sim_rad = 4;
	
	/***** Compute the Patch Features *****/
	const int W = im.W(), H = im.H();
	const int pW = W+2*R, pH = H+2*R;
	Image luv, bluv( pW, pH, 3 ), og;
	rgb2luv( luv, im );
	
	// Pad the image
	Image pluv = padIm( luv, R );
	
	// Luv color space
	typedef Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic> DynamicStride;
	patch_features_ = MatrixXf( pW*pH, 14 );
	int o=0;
	for( int i=0; i<3; i++ )
		patch_features_.col(o++) = Map<VectorXf, 0, DynamicStride>( pluv.data()+i, pW*pH, 1, DynamicStride(3,3) );
	
	Image hist;
	RMatrixXf gm, go;
	// Gradient features
	for( int i=0; i<N_FEATURE_SCALES; i++ ) {
		float s = feature_scales[i];
		int no = n_orientations[i];
		if( s > 0 )
			exactGaussianFilter( bluv.data(), pluv.data(), pW, pH, 3, s, 9 );
		else
			std::copy( pluv.begin(), pluv.end(), bluv.begin() );
		
		gradientMagAndOri( gm, go, bluv, norm_rad, norm_const );
		patch_features_.col(o++) = Map<VectorXf>( gm.data(), pW*pH );
		
		gradientHist( hist, gm, go, no );
		for( int j=0; j<no; j++ )
			patch_features_.col(o++) = Map<VectorXf, 0, DynamicStride>( hist.data()+j, pW*pH, 1, DynamicStride(no,no) );
	}
	// Blur all features
	ssim_features_ = 1*patch_features_;
	for( int i=0; i<patch_features_.cols(); i++ )
		tentFilter( patch_features_.col(i).data(), ssim_features_.col(i).data(), pW, pH, 1, chns_smooth );
	
	/***** Compute the ssim features *****/
	for( int i=0; i<patch_features_.cols(); i++ )
		boxFilter( ssim_features_.col(i).data(), patch_features_.col(i).data(), pW, pH, 1, sim_rad );
#undef N_FEATURE_SCALES

	for( int j=-R; j<=R; j++ )
		for( int i=-R; i<=R; i++ )
			did_.push_back( i+j*pW );
	
	int SR = (R/ssim_offset)*ssim_offset;
	for( int j1=-SR,k1=0; j1<= SR; j1+=ssim_offset )
		for( int i1=-SR; i1<= SR; i1+=ssim_offset,k1++ )
			for( int j2=-SR,k2=0; j2<= SR; j2+=ssim_offset )
				for( int i2=-SR; i2<= SR; i2+=ssim_offset,k2++ ) 
					if( k2 < k1 ){
						did1_.push_back( i1+j1*pW );
						did2_.push_back( i2+j2*pW );
					}
					else
						break;
	n_patch_feature_ = did_.size()*patch_features_.cols();
	n_ssim_feature_ = did1_.size()*ssim_features_.cols();
	n_feature_ = n_patch_feature_ + n_ssim_feature_;
	
	id_.reserve(((pW-2*R+1)/stride)*((pH-2*R+1)/stride));
	for (int j=R; j+R<pH; j+=stride)
		for (int i=R; i+R<pW; i+=stride)
			id_.push_back( j*pW+i );
}
int STFeatures::nSamples() const {
	return id_.size();
}
int STFeatures::featureSize() const {
	return n_feature_;
}
float STFeatures::get(int s, int f) const {
	int id = id_[s];
	if( f < n_patch_feature_ ) {
		int fid = f % patch_features_.cols();
		f /= patch_features_.cols();
		return patch_features_( id+did_[f], fid );
	}
	else {
		f -= n_patch_feature_;
		int fid = f % ssim_features_.cols();
		f /= ssim_features_.cols();
		return ssim_features_( id + did1_[f], fid ) - ssim_features_( id + did2_[f], fid );
	}
}
