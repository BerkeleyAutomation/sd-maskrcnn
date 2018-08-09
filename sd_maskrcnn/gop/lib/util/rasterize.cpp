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
#include "rasterize.h"
#include "util.h"

void rasterize( Ref<RMatrixXf> target, const RMatrixXf & l ) {
	rasterize( [&target](int x, int y, RasterType t)->void{ if( 0<=x && x<target.cols() && 0<=y && y<target.rows() ) target(y,x) = t; }, l );
// 	rasterize( [&target](int x, int y, float t)->void{ target(y,x) = t; }, l );
}
// Positive if CCW, negative otherwise
static float signedArea( const RMatrixXf & polygon ) {
	VectorXf x = polygon.col(0), y = polygon.col(1);
	VectorXf dy = 0*y;
	dy.head( y.size()-1 ) += y.tail( y.size()-1 );
	dy( y.size()-1 )      += y( 0 );
	dy.tail( y.size()-1 ) -= y.head( y.size()-1 );
	dy( 0 )               -= y( y.size()-1 );
	
	return 0.5*( x.dot( dy ) );
}
static float nextT( const Vector2f & a, const Vector2f & b, float t ) {
	Vector2f d = b-a;
	Vector2f p = a+t*d;
	float tx = d[0] == 0 ? 1e10f : ( d[0] > 0 ? (ceil(p[0])-p[0])/d[0] : (p[0]-floor(p[0]))/-d[0] );
	float ty = d[1] == 0 ? 1e10f : ( d[1] > 0 ? (ceil(p[1])-p[1])/d[1] : (p[1]-floor(p[1]))/-d[1] );
	return t + std::min(tx,ty);
}

void rasterize( RasterFunction f, const RMatrixXf & polygon ) {
	const float EPS=1e-3, DELTA_EPS=1e-5;
	eassert( polygon.cols() == 2 );
	// Make sure the poylgon is CCW (Counter Clock Wise)
	RMatrixXf l = polygon;
	if( signedArea( polygon ) < 0 )
		l = polygon.colwise().reverse();
	
	const int x0 = l.col(0).minCoeff()-1, x1 = l.col(0).maxCoeff()+2;
	const int y0 = l.col(1).minCoeff()-1, y1 = l.col(1).maxCoeff()+2;
	RMatrixXf dist = 10*RMatrixXf::Ones( y1-y0, x1-x0 );
	RMatrixXi8 label = RMatrixXi8::Zero( y1-y0, x1-x0 );
	
#define ADD( x, y, p, l ) { const float d=(p[0]-(x))*(p[0]-(x))+(p[1]-(y))*(p[1]-(y)); if( dist((y)-y0,(x)-x0)>d ) { dist((y)-y0,(x)-x0)=d; label((y)-y0,(x)-x0)=l; } }
	// This algorithm will produce odd results in concave regions with very small angle (smaller than a pixel)
	for( int k=0; k<l.rows(); k++ ) {
		Vector2f a = l.row(k), b = l.row((k+1)%l.rows());
		const float len = (b-a).norm();
		// Decide which point are inside and which outside
		const RasterType c_x0 = a[1] < b[1] ? INSIDE : OUTSIDE_BOUNDARY;
		const RasterType c_y0 = a[0] > b[0] ? INSIDE : OUTSIDE_BOUNDARY;
		const RasterType c_x1 = (RasterType)(INSIDE + OUTSIDE_BOUNDARY - c_x0), c_y1 = (RasterType)(INSIDE + OUTSIDE_BOUNDARY - c_y0);
		// Go through all the grid lines we intersect
		for( float t=nextT(a,b,0.f); t<1+DELTA_EPS; t=nextT(a,b,t+DELTA_EPS)) {
			// Avoid tie braking issues at points
			const float tt = std::min( std::max(t, 0.5f*EPS/len), 1-0.5f*EPS/len );
			VectorXf p = (1-tt)*a + tt*b;
			VectorXi ip = (p.array()+0.5).cast<int>();
			// Are we at an integer point? Then draw it solid 1
			if( fabs(p[0]-ip[0]) <= EPS && fabs(p[1]-ip[1]) <= EPS ) {
				ADD(ip[0],ip[1],p,INSIDE);
				// Color the adjacent points
				if( a[1] < b[1] ) ADD(ip[0]+1,ip[1],p,OUTSIDE_BOUNDARY);
				if( a[1] > b[1] ) ADD(ip[0]-1,ip[1],p,OUTSIDE_BOUNDARY);
				if( a[0] > b[0] ) ADD(ip[0],ip[1]+1,p,OUTSIDE_BOUNDARY);
				if( a[0] < b[0] ) ADD(ip[0],ip[1]-1,p,OUTSIDE_BOUNDARY);
			}
			else if( fabs(p[0]-ip[0]) <= EPS ) {
				ADD(ip[0],(int)p[1]  , p, c_y0);
				ADD(ip[0],(int)p[1]+1, p, c_y1);
			}
			else if( fabs(p[1]-ip[1]) <= EPS ) {
				ADD((int)p[0]  ,ip[1], p, c_x0);
				ADD((int)p[0]+1,ip[1], p, c_x1);
			}
			else
				eassert("Rasterization failed");
		}
	}
	// Rasterize interior
	for( int j=0; j<y1-y0; j++ )
		for( int i=0; i<x1-x0; i++ ) {
			if( i && label(j,i) == OUTSIDE && label(j,i-1) == INSIDE )
				label(j,i) = INSIDE;
			if( label(j,i) != OUTSIDE )
				f( i+x0, j+y0, (RasterType)label(j,i) );
		}
}
RMatrixXf rasterize( const RMatrixXf &l ) {
	int W = ceil(l.col( 0 ).maxCoeff()) + 1, H = ceil(l.col( 1 ).maxCoeff()) + 1;
	RMatrixXf r = RMatrixXf::Zero( H, W );
	rasterize( (Ref<RMatrixXf>)r, l );
	return r;
}
void rasterize( Ref<RMatrixXf> target, const Polygons & l ) {
	for (auto p: l)
		rasterize( target, p );
}
void rasterize( RasterFunction f, const Polygons & l ) {
	for (auto p: l)
		rasterize( f, p );
}
RMatrixXf rasterize( const Polygons & l ) {
	int W = 0, H = 0;
	for (auto p: l) {
		W = std::max( W, (int)ceil(p.col( 0 ).maxCoeff()) + 1 );
		H = std::max( H, (int)ceil(p.col( 1 ).maxCoeff()) + 1 );
	}
	RMatrixXf r = RMatrixXf::Zero( H, W );
	rasterize( (Ref<RMatrixXf>)r, l );
	return r;
}


