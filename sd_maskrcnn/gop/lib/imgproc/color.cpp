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
#include "color.h"
#include "util/util.h"
#include "util/sse_defs.h"
#include <vector>
#include <Eigen/Core>
using namespace Eigen;
#include <emmintrin.h>

template<typename F>
std::vector<float> initTable( F f, int N, float s ) {
	std::vector<float> r( N );
	for( int i=0; i<N; i++ )
		r[i] = f(s*i);
	return r;
}
static const std::vector<float> ltable = initTable( [](float y){ return (y > ((6.0/29)*(6.0/29)*(6.0/29)) ? 116*pow((double)y,1.0/3.0)-16 : y*((29.0/3)*(29.0/3)*(29.0/3)))/270.; }, 1025, 1.0 / 1024. );
static const std::vector<float> stable = initTable( [](float v){ return (v < 0.4045f ? v / 12.92f : pow( (v+0.055f)/1.055f, 2.4f ) ); }, 1025, 1.0 / 1024. );

inline float mapTable( float v, const float * t, int N ) {
	if( v < 0 )  v = 0;
	if( v > N-1 ) v = N-1;
	return t[(int)v];
}
template<typename T>
T min( T a, T b ) {
	return std::min(a,b);
}
template<> __m128 min<__m128>( __m128 a, __m128 b ) {
	return _mm_min_ps(a,b);
}
template<typename T>
T max( T a, T b ) {
	return std::max(a,b);
}
template<> __m128 max<__m128>( __m128 a, __m128 b ) {
	return _mm_max_ps(a,b);
}
template<typename T>
static T c( float a ) {
	return a;
}
template<> __m128 c<__m128>( float a ) {
	return _mm_set1_ps( a );
}
inline __m128 mapTable( __m128 v, const float * t, int N ) {
	float * vv = (float*)&v;
	return _mm_set_ps( mapTable( vv[3], t, N ), mapTable( vv[2], t, N ), mapTable( vv[1], t, N ), mapTable( vv[0], t, N ) );
}
template<typename T>
void rgb2xyz( T & x, T & y, T & z, const T & r, const T & g, const T & b ) {
	x = c<T>(0.430574f)*r + c<T>(0.341550f)*g + c<T>(0.178325f)*b;
	y = c<T>(0.222015f)*r + c<T>(0.706655f)*g + c<T>(0.071330f)*b;
	z = c<T>(0.020183f)*r + c<T>(0.129553f)*g + c<T>(0.939180f)*b;
}
template<typename T>
T gcorr( const T & v ) {
	return mapTable( c<T>(1024)*v, stable.data(), stable.size() );
}
template<typename T>
void srgb2xyz( T & x, T & y, T & z, T r, T g, T b ) {
	r = gcorr( r );
	g = gcorr( g );
	b = gcorr( b );
	x = c<T>(0.4124564f)*r + c<T>(0.3575761f)*g + c<T>(0.1804375f)*b;
	y = c<T>(0.2126729f)*r + c<T>(0.7151522f)*g + c<T>(0.0721750f)*b;
	z = c<T>(0.0193339f)*r + c<T>(0.1191920f)*g + c<T>(0.9503041f)*b;
}
template<typename T> void xyz2luv( T& l, T& u, T& v, T x, T y, T z ) {
	const T un = c<T>(0.197833f), vn = c<T>(0.468331f);
	l = mapTable( c<T>(1024.f)*y, ltable.data(), ltable.size() );
	z = c<T>(1.f) / (x + c<T>(15.f)*y + c<T>(3.f)*z + c<T>(1e-35f) );
	u = l * (c<T>(13.f)*(c<T>(4.f)*x*z - un)) + c<T>(88.f/270.f);
	v = l * (c<T>(13.f)*(c<T>(9.f)*y*z - vn)) + c<T>(134.f/270.f);
}
template<typename T> void xyz2lab( T& l, T& a, T& b, T x, T y, T z ) {
	x *= c<T>(1.f/0.950456f);
	z *= c<T>(1.f/1.088754f);
	x = mapTable( c<T>(1024)*x, ltable.data(), ltable.size() );
	y = mapTable( c<T>(1024)*y, ltable.data(), ltable.size() );
	z = mapTable( c<T>(1024)*z, ltable.data(), ltable.size() );
	l = y;
	a = c<T>(500.f/116.f)*(x-y);
	b = c<T>(200.f/116.f)*(y-z);
}
enum ColorType{
	LUV,
	LAB
};
template<bool SRGB,ColorType type,typename T>
static void convertRGB( T& c1, T& c2, T& c3, const T & R, const T & G, const T & B ) {
	T x,y,z;
	if (SRGB)
		srgb2xyz( x,y,z, R,G,B );
	else
		rgb2xyz( x,y,z, R,G,B );
	if (type==LUV)
		xyz2luv( c1,c2,c3, x,y,z );
	else
		xyz2lab( c1,c2,c3, x,y,z );
}
template<bool SRGB,ColorType type>
static void convertRGB( Image & luv, const Image & rgb ) {
	if( rgb.C()!= 3 )
		throw std::invalid_argument( "RGB image required!" );
	const int W = rgb.W(), H = rgb.H();
	luv.create( W, H, 3 );
	int i;
	for( i=0; i+3<W*H; i+=4 ) {
		__m128 r = _mm_set_ps( rgb[3*i+0], rgb[3*(i+1)+0], rgb[3*(i+2)+0], rgb[3*(i+3)+0] );
		__m128 g = _mm_set_ps( rgb[3*i+1], rgb[3*(i+1)+1], rgb[3*(i+2)+1], rgb[3*(i+3)+1] );
		__m128 b = _mm_set_ps( rgb[3*i+2], rgb[3*(i+1)+2], rgb[3*(i+2)+2], rgb[3*(i+3)+2] );
		__m128 l,u,v;
		convertRGB<SRGB,type>( l, u, v, r, g, b );
		float * ll = (float*)&l, * uu = (float*)&u, * vv = (float*)&v;
		for( int k=0; k<4; k++ ){
			luv[3*(i+k)+0] = ll[k];
			luv[3*(i+k)+1] = uu[k];
			luv[3*(i+k)+2] = vv[k];
		}
	}
	for( i=0; i<W*H; i++ )
		convertRGB<SRGB,type>( luv[3*i+0],luv[3*i+1],luv[3*i+2], rgb[3*i+0],rgb[3*i+1],rgb[3*i+2] ); 
}
void rgb2luv( Image & luv, const Image & rgb ) {
	convertRGB<false,LUV>( luv, rgb );
}
void srgb2luv( Image & luv, const Image & rgb ) {
	convertRGB<true,LUV>( luv, rgb );
}
void rgb2lab( Image & lab, const Image & rgb ) {
	convertRGB<false,LAB>( lab, rgb );
}
void srgb2lab( Image & lab, const Image & rgb ) {
	convertRGB<true,LAB>( lab, rgb );
}
void rgb2hsv( Image & hsv, const Image & rgb ) {
	if( rgb.C()!= 3 )
		throw std::invalid_argument( "RGB image required!" );
	const int W = rgb.W(), H = rgb.H();
	hsv.create( W, H, 3 );
	const int N = W*H;
	for( int i=0; i<N; i++ ) {
		float s,v;
		hsv[3*i+2] = v = std::max(rgb[3*i+0],std::max(rgb[3*i+1],rgb[3*i+2]));
		hsv[3*i+1] = s = v-std::min(rgb[3*i+0],std::min(rgb[3*i+1],rgb[3*i+2]));
		if( v == rgb[3*i+0] )
			hsv[3*i+0] = (rgb[3*i+1] - rgb[3*i+2]) / (6.*s+1e-10);
		else if( v == rgb[3*i+1] )
			hsv[3*i+0] = (2+rgb[3*i+2] - rgb[3*i+0]) / (6.*s+1e-10);
		else if( v == rgb[3*i+2] )
			hsv[3*i+0] = (4+rgb[3*i+0] - rgb[3*i+1]) / (6.*s+1e-10);
		if( hsv[3*i+0] < 0 )
			hsv[3*i+0] += 1;
	}
}
