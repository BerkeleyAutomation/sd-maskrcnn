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
#ifdef _MSC_VER
#define _USE_MATH_DEFINES
#endif
#include <cmath>
#include "nms.h"
#include "filter.h"


RMatrixXf ori( const RMatrixXf & im, int rad=4 ) {
	RMatrixXf bim = im, r = RMatrixXf::Zero( im.rows(), im.cols() );
	tentFilter( bim.data(), im.data(), im.cols(), im.rows(), 1, rad );
	for( int j=0; j<im.rows(); j++ ) {
		int j0 = j?j-1:j, j1 = j+1<im.rows()?j+1:j;
		for( int i=0; i<im.cols(); i++ ) {
			int i0 = i?i-1:i, i1 = i+1<im.cols()?i+1:i;
			float dx = 2*bim(j,i)-bim(j,i1)-bim(j,i0);
			float dy = 2*bim(j,i)-bim(j1,i)-bim(j0,i);
			if( bim(j0,i0)+bim(j1,i1)-bim(j1,i0)-bim(j0,i1) > 0 )
				dy = -dy;
			
			float a = atan2(dy,dx);
			if( a < 0 )
				a += M_PI;
			r(j,i) = a;
		}
	}
	return r;
}
const float lerp( const RMatrixXf & im, float x, float y ) {
	int x0 = x, y0 = y, x1 = x+1, y1 = y+1;
	float wx = x1-x, wy = y1-y;
	if( x0 < 0 ) x0 = 0;
	if( x1 < 0 ) x1 = 0;
	if( y0 < 0 ) y0 = 0;
	if( y1 < 0 ) y1 = 0;
	if( x0 >= im.cols() ) x0 = im.cols()-1;
	if( x1 >= im.cols() ) x1 = im.cols()-1;
	if( y0 >= im.rows() ) y0 = im.rows()-1;
	if( y1 >= im.rows() ) y1 = im.rows()-1;
	return    wx  * ( wy * im(y0,x0) + (1-wy) * im(y1,x0) ) +
	       (1-wx) * ( wy * im(y0,x1) + (1-wy) * im(y1,x1) );
}
RMatrixXf nms( const RMatrixXf & im, int R ) {
	RMatrixXf o = ori( im );
	RMatrixXf dx = o.array().cos(), dy = o.array().sin();
	RMatrixXf res = im;
	for( int j=0; j<im.rows(); j++ )
		for( int i=0; i<im.cols(); i++ )
			for( int r=-R; r<=R; r++ ) 
				if(r!=0) {
					float x = i+r*dx(j,i), y = j+r*dy(j,i);
					if( 1.01*im(j,i) < lerp(im,x,y) )
						res(j,i) = 0;
				}
	return res;
}
void suppressBnd( RMatrixXf &im, int b ) {
	for( int j=0; j<im.rows(); j++ )
		for( int i=0; i<im.cols(); i++ ) {
			int d = std::min( std::min(j,i), std::min((int)im.rows()-1-j, (int)im.cols()-1-i) );
			if (d < b) {
				float w = 1.0 * d / b;
				im(j,i) *= w;
			}
		}
}
