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
#include "resample.h"
#include <vector>

RMatrixXf upsample( const RMatrixXf & im, int W, int H, int stride ) {
	// THERE IS A BUG IN HERE THAT SHIFTS THE IMAGE
	RMatrixXf r( H, W );
	const int sW = im.cols(), sH = im.rows();
	
	std::vector<float> inter_weight(stride);
	for( int i=0; i<stride; i++ )
		inter_weight[i] = 1.0 - 1.0*i / stride;
	
	for( int j=0; j<H; j++ ) {
		int j0 = j/stride, j1 = (j+stride-1)/stride;
		if( j0 >= sH ) j0 = sH-1;
		if( j1 >= sH ) j1 = sH-1;
		float wy = inter_weight[j-j0*stride];
		for( int i=0; i<W; i++ ) {
			int i0 = i/stride, i1 = (i+stride-1)/stride;
			if( i0 >= sW ) i0 = sW-1;
			if( i1 >= sW ) i1 = sW-1;
			float wx = inter_weight[i-i0*stride];
			
			r(j,i) = wx   *(wy*im(j0,i0)+(1-wy)*im(j1,i0))+
			        (1-wx)*(wy*im(j0,i1)+(1-wy)*im(j1,i1));
		}
	}
	return r;
}

RMatrixXf upsampleLinear( const RMatrixXf & im, int W, int H, int stride ) {
	RMatrixXf r( H, W );
	const int sW = im.cols(), sH = im.rows();
	
	for( int j=0; j<H; j++ ) {
		float fj = (j+0.5)/stride-0.5;
		int j0 = fj, j1 = fj +1;
		float wy = 1-(fj-j0);
		if( j0 >= sH ) j0 = sH-1;
		if( j1 >= sH ) j1 = sH-1;
		if( j0 < 0 ) j0 = 0;
		for( int i=0; i<W; i++ ) {
			float fi = (i+0.5)/stride-0.5;
			int i0 = fi, i1 = fi +1;
			float wx = 1-(fi-i0);
			if( i0 >= sW ) i0 = sW-1;
			if( i1 >= sW ) i1 = sW-1;
			if( i0 < 0 ) i0 = 0;
			
			r(j,i) = wx   *(wy*im(j0,i0)+(1-wy)*im(j1,i0))+
			        (1-wx)*(wy*im(j0,i1)+(1-wy)*im(j1,i1));
		}
	}
	return r;
}

static void downsample( float *res, const float *im, int W, int H, int NW, int NH, int C ){
	const int h_nbrs = H/NH, w_nbrs = W/NW;
	memset( res, 0, NW*NH*C*sizeof(float));
	for(int j = 0; j < H; j++){
		for(int i = 0; i < W; i++){
			const int ni = i*NW/W, nj = j*NH/H;
			for(int c = 0; c < C; c++)
				res[nj*NW*C + ni*C + c] += im[j*W*C + i*C + c] / (h_nbrs*w_nbrs);
		}
	}
}
RMatrixXf downsample( const RMatrixXf & image, int NW, int NH ) {
	RMatrixXf res( NH, NW );
	downsample( res.data(), image.data(), image.cols(), image.rows(), NW, NH, 1 );
	return res;
}
Image downsample( const Image & image, int NW, int NH ) {
	Image res( NW, NH, image.C() );
	downsample( res.data(), image.data(), image.W(), image.H(), NW, NH, image.C() );
	return res;
}
template<typename T>
static void resize( T *res, const T *im, int W, int H, int NW, int NH, int C ){
	const float dy = 1.0*(H-1)/(NH-1), dx = 1.0*(W-1)/(NW-1);
	memset( res, 0, NW*NH*C*sizeof(T));
	for(int j = 0; j < NH; j++){
		for(int i = 0; i < NW; i++){
			const int i0 = i*dx       , j0 = j*dy;
			const int i1 = i0+(i0<W-1), j1 = j0+(j0<H-1);
			const float wi = i*dx-i0  , wj = j*dy-j0;
			for(int c = 0; c < C; c++)
				res[j*NW*C + i*C + c] = (1-wj)*( (1-wi)*im[j0*W*C+i0*C+c] + wi*im[j0*W*C+i1*C+c] ) +
				                        (wj  )*( (1-wi)*im[j1*W*C+i0*C+c] + wi*im[j1*W*C+i1*C+c] );
		}
	}
}
RMatrixXf resize( const RMatrixXf & image, int NW, int NH ) {
	RMatrixXf res( NH, NW );
	resize( res.data(), image.data(), image.cols(), image.rows(), NW, NH, 1 );
	return res;
}
Image resize( const Image & image, int NW, int NH ) {
	Image res( NW, NH, image.C() );
	resize( res.data(), image.data(), image.W(), image.H(), NW, NH, image.C() );
	return res;
}
Image8u resize( const Image8u & image, int NW, int NH ) {
	Image8u res( NW, NH, image.C() );
	resize( res.data(), image.data(), image.W(), image.H(), NW, NH, image.C() );
	return res;
}
Image padIm( const Image & im, int R ) {
	const int W = im.W(), H = im.H(), C = im.C();
	const int pW = W+2*R, pH=H+2*R;
	Image res( pW, pH, C );
	// Pad the image
	for( int j=0; j<H; j++ ) {
		float * pad_p = res.data()+(j+R)*pW*C;
		const float * im_p = im.data()+j*W*C;
		memcpy( pad_p+R*C, im_p, W*C*sizeof(float) );
		// Pad in x
		for( int i=0; i<R; i++ )
			memcpy( pad_p+i*C, im_p+(R-1-i)*C, C*sizeof(float) );
		for( int i=0; i<R; i++ )
			memcpy( pad_p+(R+W+i)*C, im_p+(W-1-i)*C, C*sizeof(float) );
	}
	// Pad in y
	for( int i=0; i<R; i++ )
		memcpy( res.data()+i*C*pW, res.data()+(2*R-1-i)*C*pW, C*pW*sizeof(float) );
	for( int i=0; i<R; i++ )
		memcpy( res.data()+(R+H+i)*C*pW, res.data()+(R+H-1-i)*C*pW, C*pW*sizeof(float) );
	return res;
}
