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
#include "util/win_util.h"
#include "gradient.h"
#include "filter.h"
#include <stdexcept>
#include <iostream>
#include <Eigen/Core>
using namespace Eigen;

static const float * acosTable(){
	static float table[2001];
	float * r = table+1000;
	for( int i=-1000; i<1001; i++ )
		r[i] = acos( i / 1000. );
	return r;
}
static void computeGradientOriAndMag( float * g, float * o, const float * gx, const float * gy, int N, int C ) {
	const float * acost = acosTable();
	Map<const ArrayXXf> mgx( gx, C, N ), mgy( gy, C, N );
	ArrayXXf mag = (mgx*mgx+mgy*mgy);
	for( int i=0; i<N; i++ ) {
		int j;
		float m = mag.col(i).maxCoeff(&j);
		float gm = std::max((float)sqrt(m),1e-10f);
		float cm = mgx(j,i) / gm;
		if( mgy(j,i) <= -0 ) cm = -cm;
		if( cm > 1 )  cm = 1;
		if( cm < -1 ) cm = -1;
		
		o[i] = acost[(int)(cm*1000)];
		g[i] = gm;
	}
}
static void computeGradientMag( float * g, const float * gx, const float * gy, int N, int C ) {
	Map<const ArrayXXf> mgx( gx, C, N ), mgy( gy, C, N );
	Map<VectorXf>(g,N) = (mgx*mgx+mgy*mgy).colwise().maxCoeff().sqrt();
}
template <int BINS>
static void computeGradHist( Image & hist, const RMatrixXf & gm, const RMatrixXf & go, int nori ) {
	const int W = gm.cols(), H = gm.rows();
	const int Wb = W/BINS, Hb = H/BINS;
	const int W0 = Wb*BINS, H0 = Hb*BINS;
	hist.create( Wb, Hb, nori );
	hist = 0;
	for(int j = 0; j < H0; j++){
		float * phist = hist.data() + (j/BINS)*Wb*nori;
		for(int i = 0; i < W0; phist+=nori){
			for(int k = 0; k < BINS && i<W; i++, k++){
				float o = go(j,i) / M_PI;
				unsigned int o0 = o*nori;
				float w = o*nori - o0;
				if( o0 >= nori )
					o0 = 0;
				unsigned int o1 = o0+1;
				if( o1 >= nori )
					o1 = 0;
				phist[o0] += (1-w)*gm(j,i) / (BINS*BINS);
				phist[o1] += w*gm(j,i) / (BINS*BINS);
			}
		}
	}
}
void gradientHist( Image & hist, const RMatrixXf & gm, const RMatrixXf & go, int nori, int nbins) {
	switch(nbins){
		case 1: return computeGradHist<1>(hist, gm, go, nori);
		case 2: return computeGradHist<2>(hist, gm, go, nori);
		case 3: return computeGradHist<3>(hist, gm, go, nori);
		case 4: return computeGradHist<4>(hist, gm, go, nori);
		case 5: return computeGradHist<5>(hist, gm, go, nori);
		case 6: return computeGradHist<6>(hist, gm, go, nori);
		case 7: return computeGradHist<7>(hist, gm, go, nori);
		case 8: return computeGradHist<8>(hist, gm, go, nori);
		default: throw std::invalid_argument("Bin size too large!");
	}
}
static void diff( float * r, const float * a, const float * b, int N, float w=1.0 ) {
	Map<VectorXf>( r, N ) = w*(Map<VectorXf>( (float*)a, N ) - Map<VectorXf>( (float*)b, N ));
}
void gradient( Image & gx, Image & gy, const Image & im ) {
	const int W = im.W(), H = im.H(), C = im.C();
	gx.create( W, H, C );
	gy.create( W, H, C );
	for( int j=0; j<H; j++ ) {
		float * pgx = gx.data()+j*W*C;
		float * pgy = gy.data()+j*W*C;
		const float * pim = im.data()+j*W*C;
		// Compute the x gradient
		for( int c=0; c<C; c++ ) {
			pgx[c        ] = (float)pim[C      +c]-(float)pim[c];
			pgx[(W-1)*C+c] = (float)pim[(W-1)*C+c]-(float)pim[(W-2)*C+c];
		}
		diff( pgx+C, pim+2*C, pim, (W-2)*C, 0.5 );
		// Compute the y gradient
		if(j==0)
			diff( pgy, pim+W*C, pim, W*C );
		else if( j==H-1 )
			diff( pgy, pim, pim-W*C, W*C );
		else
			diff( pgy, pim+W*C, pim-W*C, W*C, 0.5 );
	}
}
void gradientMagAndOri( RMatrixXf & gm, RMatrixXf & go, const Image & im, int norm_rad, float norm_const ) {
	Image gx, gy;
	gradient( gx, gy, im );
	const int W = im.W(), H = im.H(), C = im.C();
	go = RMatrixXf::Zero(H, W);
	gm = RMatrixXf::Zero(H, W);
	computeGradientOriAndMag( gm.data(), go.data(), gx.data(), gy.data(), W*H, C );

	if( norm_rad>0 ) {
		float * tmp = gx.data();
		tentFilter( tmp, gm.data(), W, H, 1, norm_rad );
		for( int i=0; i<W*H; i++ )
			gm.data()[i] /= tmp[i] + norm_const;
	}
}
RMatrixXf gradientMag( const Image & im, int norm_rad, float norm_const ) {
	Image gx, gy;
	gradient( gx, gy, im );
	const int W = im.W(), H = im.H(), C = im.C();
	RMatrixXf gm(H, W);
	computeGradientMag( gm.data(), gx.data(), gy.data(), W*H, C );
	
	if( norm_rad>0 ) {
		float * tmp = gx.data();
		tentFilter( tmp, gm.data(), W, H, 1, norm_rad );
		for( int i=0; i<W*H; i++ )
			gm.data()[i] /= tmp[i] + norm_const;
	}
	return gm;
}
