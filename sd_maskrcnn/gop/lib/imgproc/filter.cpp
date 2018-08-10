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
#include "filter.h"
#include "util/util.h"
#include "util/algorithm.h"
#include "util/sse_defs.h"
#include "util/eigen.h"
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <queue>
#include <Eigen/Core>
using namespace Eigen;

static void transposeBlock( float * r, const float * im, int x0, int y0, int x1, int y1, int W, int H, int C ) {
	for( int i=x0; i<x1; i++ )
		for( int j=y0; j<y1; j++ )
			for( int c=0; c<C; c++ )
				r[(j+i*H)*C+c] = im[(i+j*W)*C+c];
}
static void transpose( float * r, const float * im, int W, int H, int C ) {
	int BS = 64 / (int)sqrt(C);
	for( int i=0; i<W; i+=BS )
		for( int j=0; j<H; j+=BS )
			transposeBlock( r, im, i, j, std::min(i+BS,W), std::min(j+BS,H), W, H, C );
}

template<typename SSE_F,typename F>
void strideOp( int N, SSE_F sf, F f ) {
	int i=0;
	for( ; i+3<N; i+=4 ) sf( i );
	for( ; i<N; i++ ) f( i );
}
static void strideAddSSE( float * r, const float * a, int N ) {
	strideOp( N, [r,a](int i){ *(__m128*)(r+i) += _mm_loadu_ps( a+i ); }, [r,a](int i){ r[i] += a[i]; } );
}
static void strideSubSSE( float * r, const float * a, int N ) {
	strideOp( N, [r,a](int i){ *(__m128*)(r+i) -= _mm_loadu_ps( a+i ); }, [r,a](int i){ r[i] -= a[i]; } );
}
// static void strideAddSubMulSSE( float * r, const float * a, const float * b, float f, int N ) {
// 	__m128 sf = _mm_set1_ps( f );
// 	strideOp( N, [r,a,b,sf](int i){ *(__m128*)(r+i) += sf*(_mm_loadu_ps( a+i ) - _mm_loadu_ps( b+i )); }, [r,a,b,f](int i){ r[i] += f*(a[i] - b[i]); } );
// }
static void strideAddSubSSE( float * r, const float * a, const float * b, int N ) {
	strideOp( N, [r,a,b](int i){ *(__m128*)(r+i) += (_mm_loadu_ps( a+i ) - _mm_loadu_ps( b+i )); }, [r,a,b](int i){ r[i] += a[i] - b[i]; } );
}
static void strideMulSSE( float * r, const float * a, float f, int N ) {
	__m128 sf = _mm_set1_ps( f );
	strideOp( N, [r,a,sf](int i){ *(__m128*)(r+i) = *(__m128*)(a+i) * sf; }, [r,a,f](int i){ r[i] = a[i]*f; } );
}
static void strideAddAddSSE( float * r1, float * r2, const float * a, int N ) {
	strideOp( N, [r1,r2,a](int i){ *(__m128*)(r1+i) += *(__m128*)(r2+i) += _mm_loadu_ps(a+i); }, [r1,r2,a](int i){ r1[i] += r2[i] += a[i]; } );
}
static void strideAddAddDiffSSE( float * r1, float * r2, const float * a, const float * b, const float * c, int N ) {
	__m128 two = _mm_set1_ps( 2.0 );
	strideOp( N, [two,r1,r2,a,b,c](int i){ *(__m128*)(r1+i) += *(__m128*)(r2+i) += _mm_loadu_ps(a+i) + _mm_loadu_ps(b+i) - two*_mm_loadu_ps(c+i); }, [r1,r2,a,b,c](int i){ r1[i] += r2[i] += a[i] + b[i] - 2*c[i]; } );
}
template<int C>
void bx( float *r, const float *a, int W, int r2, float f ) {
	// TODO: SSE THIS!
	float sm[C] = {0.f};
	int o=C*r2, p=C*(r2+1), w0=r2+1, w1=W-r2;
	for(int j=0,k=0; j<=r2; j++)
		for(int c=0; c<C; c++,k++)
			sm[c] += a[k];
	for(int c=0; c<C; c++)
		sm[c] += sm[c]-a[C*r2+c];
	int j=0,k=0;
	for(; j<w0; j++)
		for(int c=0; c<C; c++,k++)
			r[k]=f*( sm[c] -= a[(r2-j)*C+c]-a[k+o] );
	for(; j<w1; j++)
		for(int c=0; c<C; c++,k++)
			r[k]=f*( sm[c] -= a[k-p]-a[k+o] );
	for(; j<W; j++)
		for(int c=0; c<C; c++,k++)
			r[k]=f*( sm[c] -= a[k-p]-a[(2*W-(r2+1)-j)*C+c] );
}
typedef void (*BxFuntion)( float*,const float*,int,int,float );
static BxFuntion bx_array[] = { bx<1>, bx<2>, bx<3>, bx<4>, bx<5>, bx<6>, bx<7>, bx<8>, bx<9>, bx<10>, bx<11>, bx<12>, bx<13>, bx<14>, bx<15>, bx<16>, bx<17>, bx<18>, bx<19>, bx<20> };
void boxFilter( float * r, const float * im, int W, int H, int C, int r2 ) {
	if( C<1 || C>20 )
		throw std::invalid_argument( "Unsupported number of channels! C not in 1..20." );
	float nm = 1.0 / ((2*r2+1)*(2*r2+1));
	BxFuntion bx = bx_array[C-1];
	float * sm = (float*)_mm_malloc( (W*C+16)*sizeof(float), 16 );
	memset( sm, 0, (W*C+16)*sizeof(float) );
	
	for( int j=0; j<H && j<=r2; j++ )
		strideAddSSE( sm, im+j*W*C, W*C );
	strideMulSSE( sm, sm, 2.0, W*C );
	strideSubSSE( sm, im+r2*W*C, W*C );
	
	for( int j=0; j<H; j++ ) {
		bx( r+j*W*C, sm, W, r2, nm );
		strideAddSubSSE( sm, im+(2*H-1-abs(2*H-2*(j+r2+1)-1))/2*W*C, im+abs(2*(j-r2)+1)/2*W*C, W*C );
	}
	_mm_free( sm );
}
template<int C>
void tx( float *r, const float *a, int W, int r2, float f ) {
	// TODO: SSE THIS!
	float sm[C] = {0.f}, sm2[C] = {0.f};
	int o=C*(r2+1), w0=r2+1, w1=W-r2-1;
	for(int j=0,k=0; j<=r2 && j<W; j++)
		for(int c=0; c<C; c++,k++)
			sm2[c] += sm[c] += a[k];
	for(int c=0; c<C; c++)
		sm2[c] = 2*sm2[c] - sm[c];
	for(int c=0; c<C; c++)
		sm[c] = 0;
	int j=0,k=0;
	for(; j<w0; j++)
		for(int c=0; c<C; c++,k++) {
			r[k] = f*sm2[c];
			sm2[c] += sm[c] += a[(r2-j)*C+c]+a[k+o]-2*a[k];
		}
	for(; j<w1; j++)
		for(int c=0; c<C; c++,k++) {
			r[k] = f*sm2[c];
			sm2[c] += sm[c] += a[k-o]+a[k+o]-2*a[k];
		}
	for(; j<W-1; j++)
		for(int c=0; c<C; c++,k++){
			r[k] = f*sm2[c];
			sm2[c] += sm[c] += a[k-o]+a[(2*W-r2-j-2)*C+c]-2*a[k];
		}
	for(int c=0; c<C; c++,k++)
		r[k] = f*sm2[c];
}
typedef void (*TxFuntion)( float*,const float*,int,int,float );
static TxFuntion tx_array[] = { tx<1>, tx<2>, tx<3>, tx<4>, tx<5>, tx<6>, tx<7>, tx<8>, tx<9>, tx<10>, tx<11>, tx<12>, tx<13>, tx<14>, tx<15>, tx<16>, tx<17>, tx<18>, tx<19>, tx<20> };
void tentFilter( float * r, const float * im, int W, int H, int C, int r2 ) {
	if( C<1 || C>20 )
		throw std::invalid_argument( "Unsupported number of channels! C not in 1..20." );
	float nm = 1.0 / ((r2+1)*(r2+1)*(r2+1)*(r2+1));
	TxFuntion tx = tx_array[C-1];
	float * sm = (float*)_mm_malloc( (W*C+16)*sizeof(float), 16 ), *sm2 = (float*)_mm_malloc( (W*C+16)*sizeof(float), 16 );
	memset( sm, 0, (W*C+16)*sizeof(float) );
	memset( sm2, 0, (W*C+16)*sizeof(float) );
	
	for( int j=0; j<H && j<=r2; j++ )
		strideAddAddSSE( sm2, sm, im+j*W*C, W*C );
	strideMulSSE( sm2, sm2, 2.0, W*C );
	strideSubSSE( sm2, sm, W*C );
	memset( sm, 0, (W*C+16)*sizeof(float) );
	
	for( int j=0; j<H; j++ ) {
		tx( r+j*W*C, sm2, W, r2, nm );
		strideAddAddDiffSSE( sm2, sm, im+(2*H-1-abs(2*H-2*(j+r2+1)-1))/2*W*C, im+abs(2*(j-r2-1)+1)/2*W*C, im+j*W*C, W*C );
	}
	_mm_free( sm );
	_mm_free( sm2 );
}

Vector4f YvVCoef(float sigma)
{
    /* the recipe in the Young-van Vliet paper:
     * I.T. Young, L.J. van Vliet, M. van Ginkel, Recursive Gabor filtering.
     * IEEE Trans. Sig. Proc., vol. 50, pp. 2799-2805, 2002.
     *
     * (this is an improvement over Young-Van Vliet, Sig. Proc. 44, 1995)
     */

	/* initial values */
	float m0 = 1.16680, m1 = 1.10783, m2 = 1.40586;
	float m1sq = m1*m1, m2sq = m2*m2;

	/* calculate q */
	float q;
	if(sigma < 3.556)
		q = -0.2568 + 0.5784 * sigma + 0.0561 * sigma * sigma;
	else
		q = 2.5091 + 0.9804 * (sigma - 3.556);
	float qsq = q*q;

	/* calculate scale, and b[0,1,2,3] */
	float scale = (m0 + q) * (m1sq + m2sq + 2*m1*q + qsq);
	
	Vector4f r;
	/* calculate B */
	r[0] = (m0 * (m1sq + m2sq))/scale;
	/* calculate the filter parameters */
	r[1] = q * (2*m0*m1 + m1sq + m2sq + (2*m0 + 4*m1) * q + 3*qsq) / scale;
	r[2] = - qsq * (m0 + 2*m1 + 3*q) / scale;
	r[3] = qsq * q / scale;
	
    return r;
}

Matrix3f TriggsM(const Vector4f & b)
{
	float a1 = b[1];
	float a2 = b[2];
	float a3 = b[3];
	
	float scale = 1.0/((1.0+a1-a2+a3)*(1.0-a1-a2-a3)*(1.0+a2+(a1-a3)*a3));
	Matrix3f M;
	M(0,0) = scale*(-a3*a1+1.0-a3*a3-a2);
	M(0,1) = scale*(a3+a1)*(a2+a3*a1);
	M(0,2) = scale*a3*(a1+a3*a2);
	M(1,0) = scale*(a1+a3*a2);
	M(1,1) = -scale*(a2-1.0)*(a2+a3*a1);
	M(1,2) = -scale*a3*(a3*a1+a3*a3+a2-1.0);
	M(2,0) = scale*(a3*a1+a2+a1*a1-a2*a2);
	M(2,1) = scale*(a1*a2+a3*a2*a2-a1*a3*a3-a3*a3*a3-a3*a2+a3);
	M(2,2) = scale*a3*(a1+a3*a2);
	return M;
}
static void rgy( float * r, const float * im, int W, int H, int C, const VectorXf & b ) {
	Matrix3f M = TriggsM( b );
	float smsq = b[0];
	float sm = b[0]*b[0];
	
	Matrix<float,Dynamic,4> p( W*C, 4 );
	Map<const MatrixXf> pim(im,W*C,H);
	Map<MatrixXf> pr(r,W*C,H);
	p.col(0) = p.col(1) = p.col(2) = p.col(3) = pim.col(0) / smsq;
	
	VectorXf iplus = pim.col(H-1);
	for( int i=0; i<H; i++ )
		pr.col(i) = p.col(i&3) = pim.col(i) + b[1]*p.col((i+3)&3) + b[2]*p.col((i+2)&3) + b[3]*p.col((i+1)&3);
		
	VectorXf uplus = iplus.array() / (1.-b[1]-b[2]-b[3]);
	VectorXf vplus = uplus.array() / (1.-b[1]-b[2]-b[3]);
		
	Matrix<float,Dynamic,3> pp( W*C, 3 );
	pp.col(0) = p.col((H-1)&3) - uplus;
	pp.col(1) = p.col((H-2)&3) - uplus;
	pp.col(2) = p.col((H-3)&3) - uplus;
	pp = pp*M.transpose();
	p.col((H-1)&3) = sm*(pp.col(0) + vplus);
	p.col((H  )&3) = sm*(pp.col(1) + vplus);
	p.col((H+1)&3) = sm*(pp.col(2) + vplus);
		
	pr.col(H-1) = p.col((H-1)&3);
	for( int i=H-2; i>=0; i-- )
		pr.col(i) = p.col(i&3) = sm*pr.col(i) + b[1]*p.col((i+1)&3) + b[2]*p.col((i+2)&3) + b[3]*p.col((i+3)&3);
}
void gaussianFilter( float * r, const float * im, int W, int H, int C, float sigma ) {
	Vector4f b = YvVCoef( sigma );
	float * tmp = new float[W*H*C];
	transpose( tmp, im, W, H, C );
	rgy( tmp, tmp, H, W, C, b );
	transpose( r, tmp, H, W, C );
	delete[] tmp;
	rgy( r, r, W, H, C, b );
}
void egx( Matrix4Xf & r, const Matrix4Xf & im, int W, int H, const VectorXf & f ) {
	__m128 zero = _mm_set1_ps( 0.f );
	int R = f.size();
	std::vector<__m128> sf(R);
	for( int i=0; i<R; i++ )
		sf[i] = _mm_set1_ps( f[i] );
	
	for( int j=0; j<H; j++ ) {
		const __m128 * pim = ((const __m128 *) im.data())+j*W;
		__m128 * pr = ((__m128 *) r.data())+j*W;
		for( int i=0; i<W; i++ ) {
			__m128 s = zero;
			int k=-R+1;
			for( ; k<-i; k++ )
				s += sf[-k]*pim[-k-i];
			for( ; k<=0; k++ )
				s += sf[-k]*pim[i+k];
			for( ; k<R && i+k<W; k++ )
				s += sf[k]*pim[i+k];
			for( ; k<R; k++ )
				s += sf[k]*pim[2*(W-1)-(i+k)];
			pr[i] = s;
		}
	}
}
void egy( Matrix4Xf & r, const Matrix4Xf & im, int W, int H, const VectorXf & f ) {
	const __m128 * pim = (const __m128*)im.data();
	__m128 * pr  = (__m128*)r.data();
	
	__m128 zero = _mm_set1_ps( 0.f );
	int R = f.size();
	std::vector<__m128> sf(R);
	for( int i=0; i<R; i++ )
		sf[i] = _mm_set1_ps( f[i] );
	for( int i=0; i<H; i++ ) {
		for( int j=0; j<W; j++ )
			pr[i*W+j] = zero;
		int k=-R+1;
		for( ; k<-i; k++ )
			for( int j=0; j<W; j++ )
				pr[i*W+j] += sf[-k]*pim[(-k-i)*W+j];
		for( ; k<=0; k++ )
			for( int j=0; j<W; j++ )
				pr[i*W+j] += sf[-k]*pim[(k+i)*W+j];
		for( ; k<R && i+k<H; k++ )
			for( int j=0; j<W; j++ )
				pr[i*W+j] += sf[k]*pim[(k+i)*W+j];
		for( ; k<R; k++ )
			for( int j=0; j<W; j++ )
				pr[i*W+j] += sf[k]*pim[(2*(H-1)-(i+k))*W+j];
	}
}
VectorXf gaussFactor( float sigma, int R ) {
	VectorXf r(R);
	for( int i=0; i<R; i++ )
		r[i] = exp( -0.5*i*i/(sigma*sigma) );
	return r / (2*r.array().sum()-1);
}
void exactGaussianFilter( float * r, const float * im, int W, int H, int C, float sigma, int R ) {
	if( C>4 )
		throw std::invalid_argument( "At most 4 channels supported" );
	// Load the image
	Matrix4Xf mim = MatrixXf::Zero( 4, W*H ), tmp = MatrixXf::Zero( 4, W*H );
	mim.topRows(C) = Map<const MatrixXf>( im, C, W*H );
	
	// Compute the filter
	VectorXf f = gaussFactor( sigma, R );
	
	egx( tmp, mim, W, H, f );
	egy( mim, tmp, W, H, f );
	
	Map<MatrixXf>( r, C, W*H ) = mim.topRows(C);
}
template<typename F, int C>
void prefixFilter( float * r, const float * im, int W, int H, F f, int rad, float init=0 ) {
#define CP( a, b, N ) {for( int k=0; k<(N); k++ ){ (a)[k] = (b)[k]; }}
#define CF( a, b, N ) {for( int k=0; k<(N); k++ ){ (a)[k] = f( (a)[k], (b)[k] ); }}
	assert( W>2*rad && H > 2*rad );
	std::vector<float> vtmp(W*H*C,init);
	float * tmp = vtmp.data();
	std::copy(im,im+W*H*C,tmp);
	const int d = 2*rad+1;
	// Filter in X
	for( int j=0; j<H; j++ ) {
		// Prefix
		for( int i=0; i<W; i+=d-1 ) {
			float sm[C];
			CP( sm, im+(j*W+i)*C, C );
			if( i>=rad )
				CF( tmp+(j*W+i-rad)*C, sm, C );
			for( int ii=i+1; ii<i+d-1; ii++ ) {
				if( ii < W )
					CF( sm, im+(j*W+ii)*C, C );
				if( ii>=rad && ii-rad<W )
					CF( tmp+(j*W+ii-rad)*C, sm, C );
			}
		}
		// Postfix
		for( int i=W/(d-1)*(d-1); i>=0; i-=d-1 ) {
			float sm[C];
			int imax = std::min(i+d-2,W-1);
			CP( sm, im+(j*W+imax)*C, C );
			if( imax+rad<W )
				CF( tmp+(j*W+imax+rad)*C, sm, C );
			for( int ii=imax-1; ii>=0 && ii>=i; ii-- ) {
				CF( sm, im+(j*W+ii)*C, C );
				if( ii+rad<W )
					CF( tmp+(j*W+ii+rad)*C, sm, C );
			}
		}
	}
	std::fill( r, r+W*H*C, init );
	std::vector<float> vsm( W*C );
	float * sm = vsm.data();
	// Prefix
	for( int j=0; j<H; j+=d-1 ) {
		CP( sm, tmp+j*C*W, C*W );
		if( j>=rad )
			CF( r+(j-rad)*C*W, sm, C*W );
		for( int jj=j+1; jj<j+d-1; jj++ ) {
			if( jj < H )
				CF( sm, tmp+jj*W*C, W*C );
			if( jj>=rad && jj-rad < H )
				CF( r+(jj-rad)*W*C, sm, W*C );
		}
	}
	// Postfix
	for( int j=H/(d-1)*(d-1); j>=0; j-=d-1 ) {
		int jmax = std::min(j+d-2,H-1);
		CP( sm, tmp+jmax*W*C, W*C );
		if( jmax+rad<H )
			CF( r+(jmax+rad)*W*C, sm, W*C );
		for( int jj=jmax-1; jj>=0 && jj>=j; jj-- ) {
			CF( sm, tmp+jj*W*C, W*C );
			if( jj+rad<H )
				CF( r+(jj+rad)*W*C, sm, W*C );
		}
	}
#undef CP
#undef CF
}
template<typename F>
void prefixFilter( float * r, const float * im, int W, int H, int C, F f, int rad, float init=0 ) {
	switch(C){
	case 1: return prefixFilter<F,1>( r, im, W, H, f, rad, init );
	case 2: return prefixFilter<F,2>( r, im, W, H, f, rad, init );
	case 3: return prefixFilter<F,3>( r, im, W, H, f, rad, init );
	case 4: return prefixFilter<F,4>( r, im, W, H, f, rad, init );
	case 5: return prefixFilter<F,5>( r, im, W, H, f, rad, init );
	case 6: return prefixFilter<F,6>( r, im, W, H, f, rad, init );
	case 7: return prefixFilter<F,7>( r, im, W, H, f, rad, init );
	case 8: return prefixFilter<F,8>( r, im, W, H, f, rad, init );
	case 9: return prefixFilter<F,9>( r, im, W, H, f, rad, init );
	case 10: return prefixFilter<F,10>( r, im, W, H, f, rad, init );
	case 11: return prefixFilter<F,11>( r, im, W, H, f, rad, init );
	case 12: return prefixFilter<F,12>( r, im, W, H, f, rad, init );
	case 13: return prefixFilter<F,13>( r, im, W, H, f, rad, init );
	case 14: return prefixFilter<F,14>( r, im, W, H, f, rad, init );
	case 15: return prefixFilter<F,15>( r, im, W, H, f, rad, init );
	default: throw std::invalid_argument("Only C<16 supported");
	}
}

void minFilter( float * r, const float * im, int W, int H, int C, int rad ) {
	prefixFilter( r, im, W, H, C, [](float a, float b){ return a<b?a:b; }, rad, 1e10 );
}
void maxFilter( float * r, const float * im, int W, int H, int C, int rad ) {
	prefixFilter( r, im, W, H, C, [](float a, float b){ return a>b?a:b; }, rad );
}
void percentileFilter( float *r, const float *im, int W, int H, int C, int rad, float p ) {
	if( p < 1e-5 ) return minFilter( r, im, W, H, C, rad );
	if( p > 1-1e-5 ) return maxFilter( r, im, W, H, C, rad );
	std::vector<float> v;
	for( int j=0; j<H; j++ )
		for( int i=0; i<W; i++ )
			for( int c=0; c<C; c++ ) {
				v.clear();
				for( int jj=std::max(j-rad,0); jj<=j+rad && jj<H; jj++ )
					for( int ii=std::max(i-rad,0); ii<=i+rad && ii<W; ii++ )
						v.push_back( im[(jj*W+ii)*C+c] );
				r[(j*W+i)*C+c] = quickSelect( v, p*(v.size()-1) );
			}
}

