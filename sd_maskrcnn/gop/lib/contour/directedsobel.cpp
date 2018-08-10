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
#include "directedsobel.h"
#include "imgproc/color.h"
#include "imgproc/filter.h"
#include "imgproc/gradient.h"
#include <vector>

static void addDiffSqrSum( float * r, const float * a, const float * b, int N, int C, float w=1.0 ) {
	ArrayXXf d = Map<ArrayXXf>( (float*)a, C, N ) - Map<ArrayXXf>( (float*)b, C, N );
	Map<ArrayXf>( r, N ) += w*(d*d).colwise().sum().sqrt();
}
static void addDxDy( float * gx, float * gy, const float * im, int W, int H, int C, float w=1.0 ) {
	for( int j=0; j<H; j++ ) {
		float * pgx = gx+j*(W-1);
		const float * pim = im+j*W*C;
		// Compute the x gradient
		addDiffSqrSum( pgx, pim, pim+C, W-1, C, w );
	}
	// Compute the y gradient
	addDiffSqrSum( gy, im, im+C*W, W*(H-1), C, w );
}
template<int DX, int DY>
RMatrixXf nms( const RMatrixXf & d ) {
	RMatrixXf r = 1*d;
	const float T = 1.0;
	const int W = d.cols(), H = d.rows();
	for( int j=0; j<H; j++ )
		for( int i=0; i<W; i++ ){
			if( DY && j     && T*d(j,i) < d(j-1,i) )
				r(j,i) = 0.1*d(j,i);
			if( DY && j<H-1 && T*d(j,i) < d(j+1,i) )
				r(j,i) = 0.1*d(j,i);
			if( DX && i     && T*d(j,i) < d(j,i-1) )
				r(j,i) = 0.1*d(j,i);
			if( DX && i<W-1 && T*d(j,i) < d(j,i+1) )
				r(j,i) = 0.1*d(j,i);
		}
	return r;
}
// #define NORM_INDIVIDUAL
std::tuple<RMatrixXf,RMatrixXf> DirectedSobel::detectXY(const Image8u & im, int do_nms, int fast) const {
	// Some magic parameters
	const int norm_rad = 5;
	const float norm_const = 0.01;
	const int b[] = {2,4,8};
	const float w[] = {0.25,0.5,1};
	const int NB = fast ? 1 : (sizeof(b)/sizeof(b[0]));
	if( do_nms == -1 ) do_nms = do_nms_;
	
	// Convert the image to luv
	const int W = im.W(), H = im.H();
	const int C = 3, N = W*H;
	Image luv, bluv;
	rgb2luv( luv, im );
	bluv = luv;

	RMatrixXf dx = RMatrixXf::Zero(H,W-1), dy = RMatrixXf::Zero(H-1,W), tdx, tdy;
	
	for (int i=0; i<NB; i++ ) {
		if( b[i] > 0 )
			tentFilter( bluv.data(), luv.data(), W, H, 3, 2*b[i] );
		else
			std::copy( luv.data(), luv.data()+N*C, bluv.data() );
		addDxDy( dx.data(), dy.data(), bluv.data(), W, H, C, w[i] );
	}
	if( do_nms ) {
		dx = nms<1,0>( dx );
		dy = nms<0,1>( dy );
	}
	if( norm_rad > 0 ){
#ifdef NORM_INDIVIDUAL
		{
			RMatrixXf n(H,W-1), nn(H,W-1);
			n = dx.array()*dx.array();
			tentFilter( nn.data(), n.data(), W-1, H, 1, norm_rad );
			dx.array() /= nn.array().sqrt()+norm_const;
		}
		{
			RMatrixXf n(H-1,W), nn(H-1,W);
			n = dy.array()*dy.array();
			tentFilter( nn.data(), n.data(), W, H-1, 1, norm_rad );
			dy.array() /= nn.array().sqrt()+norm_const;
		}
#else
		RMatrixXf n = RMatrixXf::Zero(H,W), nn(H,W);
		n.leftCols(W-1)   = dx.array().max( n.leftCols(W-1).array() );
		n.rightCols(W-1)  = dx.array().max( n.rightCols(W-1).array() );
		n.topRows(H-1)    = dy.array().max( n.topRows(H-1).array() );
		n.bottomRows(H-1) = dy.array().max( n.bottomRows(H-1).array() );
		n = n.array()*n.array();
		tentFilter( nn.data(), n.data(), W, H, 1, norm_rad );
		n = nn.array().sqrt()+norm_const;
		
		dx.array() /= n.rightCols(W-1).array().max( n.leftCols(W-1).array() );
		dy.array() /= n.bottomRows(H-1).array().max( n.topRows(H-1).array() );
#endif
	}
	float mx = std::max( dx.maxCoeff(), dy.maxCoeff() );
	dx.array() /= mx;
	dy.array() /= mx;
	return std::tie(dx,dy);
}
RMatrixXf DirectedSobel::detect(const Image8u & im) const {
	RMatrixXf dx, dy, d = RMatrixXf::Zero(im.H(),im.W());
	std::tie( dx, dy ) = detectXY( im );
	
// 	d.leftCols(im.W()-1)  = dx.array().max( d.leftCols(im.W()-1).array() );
	d.rightCols(im.W()-1) = dx.array().max( d.rightCols(im.W()-1).array() );
	
// 	d.topRows(im.H()-1)  = dy.array().max( d.topRows(im.H()-1).array() );
	d.bottomRows(im.H()-1) = dy.array().max( d.bottomRows(im.H()-1).array() );
	
	return d;
}
DirectedSobel::DirectedSobel(bool do_nms) : do_nms_(do_nms)
{
}
