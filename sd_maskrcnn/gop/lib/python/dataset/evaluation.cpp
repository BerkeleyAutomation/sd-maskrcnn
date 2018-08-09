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
#include "evaluation.h"
#include "util/util.h"
#include "util/eigen.h"
#include <imgproc/morph.h>
#include <tuple>
#include <queue>
#include <stack>
#include <cmath>
#include "python/util.h"

namespace EvalPrivate {
	struct BipartiteMatch{
	private:
		bool bfs( int na, const std::vector< std::vector<int> > & nbr, std::vector<float> & dist ) {
			const float inf = std::numeric_limits<float>::infinity();
			std::queue<int> q;
			dist[0] = inf;
			for( int v=0; v <na; v++ ){
				if( match_a[v]==-1 ) {
					dist[v+1] = 0;
					q.push( v );
				}
				else
					dist[v+1] = inf;
			}
			while( !q.empty() ) {
				int v = q.front();
				q.pop();
				if (dist[v+1] < dist[0])
					for( int u: nbr[v] )
						if (dist[ match_b[u]+1 ] == inf) {
							dist[ match_b[u]+1 ] = dist[v+1] + 1;
							q.push( match_b[u] );
						}
			}
			return dist[0] < inf;
		}
		bool dfs( int v, const std::vector< std::vector<int> > & nbr, std::vector<float> & dist ) {
			const float inf = std::numeric_limits<float>::infinity();
			if( v>=0 ) {
				if( dist[v+1] == inf )
					return false;
				
				for( int u: nbr[v] ) {
					if( dist[ match_b[u]+1 ] == dist[v+1] + 1 && dfs( match_b[u], nbr, dist ) ) {
						match_b[u] = v;
						match_a[v] = u;
						return true;
					}
				}
				dist[v+1] = inf;
				return false;
			}
			return true;
		}
	public:
		struct Edge{
			int a,b;
			Edge( int a=0, int b=0 ):a(a),b(b){}
		};
		std::vector< int > match_a, match_b;
		BipartiteMatch( int na, int nb, const std::vector< Edge > & edges ):match_a( na, -1 ), match_b( nb, -1 ){
	// 		const float inf = std::numeric_limits<float>::infinity();
			// Build the graph
			std::vector< std::vector<int> > nbr( na );
			for( Edge e: edges )
				nbr[e.a].push_back( e.b );
			
			// Run Hopcroft-Karp
			std::vector<float> dist( na+1, 0 );
			while(1) {
				// Run BFS
				if( !bfs( na, nbr, dist ) )
					break;
				
				// Start matching
				for( int i=0; i<na; i++ )
					if( match_a[i] == -1 ) {
						// Run a DFS
						dfs( i, nbr, dist );
					}
			}
		}
	};
}
void matchAny( bool * pr, const bool * pa, const bool * pb, int W, int H, double max_r ) {
	memset( pr, 0, W*H*2*sizeof(bool) );
	float r2 = max_r*max_r*(W*W+H*H);
    const int rd = (int) ceil(sqrt(r2));	
	for( int j=0; j<H; j++ )
		for( int i=0; i<W; i++ )
			if( pa[j*W+i] )
				for( int jj=std::max(j-rd,0); jj<=std::min(j+rd,H-1); jj++ )
					for( int ii=std::max(i-rd,0); ii<=std::min(i+rd,W-1); ii++ )
						if( (i-ii)*(i-ii)+(j-jj)*(j-jj) <= r2 )
							if( pb[jj*W+ii] )
								pr[j*W+i+0*W*H] = pr[jj*W+ii+1*W*H] = 1;
}

np::ndarray matchAny(const np::ndarray & a, const np::ndarray & b, double max_r ) {
	checkArray( a, bool, 2, 2, true );
	checkArray( b, bool, 2, 2, true );
	int H = a.shape(0), W = a.shape(1);
	if( H != b.shape(0) || W != b.shape(1) )
		throw std::invalid_argument( "a and b need to have the same shape!\n" );
	np::ndarray r = np::zeros( make_tuple(2,H,W), a.get_dtype() );
	matchAny( (bool *)r.get_data(), (const bool *)a.get_data(), (const bool *)b.get_data(), W, H, max_r );
	return r;
}
void matchBp( bool * pr, const bool * pa, const bool * pb, int W, int H, double max_r ) {
	using namespace EvalPrivate;
	matchAny( pr, pa, pb, W, H, max_r );
	float r2 = max_r*max_r*(W*W+H*H);
    const int rd = (int) ceil(sqrt(r2));	
	// Compute the bipartite graph size
	std::vector<int> ia(W*H,-1), ib(W*H,-1);
	int ca=0,cb=0;
	for( int j=0; j<H; j++ )
		for( int i=0; i<W; i++ ) {
			if( pr[j*W+i+0] )
				ia[j*W+i] = ca++;
			if( pr[j*W+i+W*H] )
				ib[j*W+i] = cb++;
		}
	
	std::vector< BipartiteMatch::Edge > edges;
	for( int j=0; j<H; j++ )
		for( int i=0; i<W; i++ )
			if( ia[j*W+i]>=0 )
				for( int jj=std::max(j-rd,0); jj<=std::min(j+rd,H-1); jj++ )
					for( int ii=std::max(i-rd,0); ii<=std::min(i+rd,W-1); ii++ )
						if( (i-ii)*(i-ii)+(j-jj)*(j-jj) <= r2 )
							if( ib[jj*W+ii]>=0 )
								edges.push_back( BipartiteMatch::Edge(ia[j*W+i],ib[jj*W+ii]) );
	
	BipartiteMatch match( ca, cb, edges );
	for( int j=0; j<H; j++ )
		for( int i=0; i<W; i++ ) {
			if( ia[j*W+i]>=0 )
				pr[j*W+i+0]   = (match.match_a[ ia[j*W+i] ]>=0);
			if( ib[j*W+i]>=0 )
				pr[j*W+i+W*H] = (match.match_b[ ib[j*W+i] ]>=0);
		}
}
np::ndarray matchBp(const np::ndarray & a, const np::ndarray & b, double max_r ) {
	checkArray( a, bool, 2, 2, true );
	checkArray( b, bool, 2, 2, true );
	int H = a.shape(0), W = a.shape(1);
	if( H != b.shape(0) || W != b.shape(1) )
		throw std::invalid_argument( "a and b need to have the same shape!\n" );
	np::ndarray r = np::zeros( make_tuple(2,H,W), a.get_dtype() );
	matchBp( (bool*)r.get_data(), (const bool*)a.get_data(), (const bool*)b.get_data(), W, H, max_r );
	return r;
}
std::tuple<int,int,int,int> evalBoundaryBinary(const bool * d, const bool * bnd, int W, int H, int D, double max_r, const bool * mask=NULL ) {
	int sum_r=0,cnt_r=0;
	std::vector<char> acc(W*H,0), m(W*H*2,0);
	bool * pacc = (bool*)acc.data(), *pm = (bool*)m.data();
	for( int k=0; k<D; k++ ) {
		matchBp( pm, d, bnd+W*H*k, W, H, max_r );
		for( int i=0; i<W*H; i++ )
			if (mask==NULL || mask[i]) {
				pacc[i] |= pm[i];
				sum_r += bnd[W*H*k+i];
				cnt_r += pm[W*H+i];
			}
	}
	int sum_p=0,cnt_p=0;
	for( int i=0; i<W*H; i++ )
		if (mask==NULL || mask[i]) {
			// Ignore everything outside the mask
			sum_p += d[i];
			cnt_p += pacc[i];
		}
	return std::tie(cnt_r,sum_r,cnt_p,sum_p);
}
tuple evalBoundaryBinary(const np::ndarray & d, const np::ndarray & bnd, double max_r ) {
	checkArray( d, bool, 2, 2, true );
	checkArray( bnd, bool, 3, 3, true );
	int H = d.shape(0), W = d.shape(1);
	if( H != bnd.shape(1) || W != bnd.shape(2) )
		throw std::invalid_argument( "Ground truth size and boundary size need to have the same shape!\n" );
	int cnt_r,sum_r,cnt_p,sum_p;
	std::tie(cnt_r,sum_r,cnt_p,sum_p) = evalBoundaryBinary( (const bool *)d.get_data(), (const bool *)bnd.get_data(), W, H, bnd.shape(0), max_r );
	return make_tuple(cnt_r,sum_r,cnt_p,sum_p);
}
void evalBoundary( float * r, const float * d, const bool * bnd, int W, int H, int N, int nthres, double max_r, const bool * mask=NULL ){
	for( int k=0; k <nthres; k++ ) {
		float t = 1.0*k / nthres;
		RMatrixXb tmp(H,W);
		bool * pp = tmp.data();
		for( int i=0; i<W*H; i++ )
			pp[i] = d[i] > t;
		if(t>0)
			thinningGuoHall( tmp );
		std::tie(r[5*k+1],r[5*k+2],r[5*k+3],r[5*k+4]) = evalBoundaryBinary( pp, bnd, W, H, N, max_r, mask );
		r[5*k+0] = t;
	}
}
np::ndarray evalBoundary(const np::ndarray & d, const np::ndarray & bnd, int nthres, double max_r ){
	checkArray( d, float, 2, 2, true );
	checkArray( bnd, bool, 3, 3, true );
	int H = d.shape(0), W = d.shape(1);
	if( H != bnd.shape(1) || W != bnd.shape(2) )
		throw std::invalid_argument( "Ground truth size and boundary size need to have the same shape!\n" );
	np::ndarray r = np::zeros( make_tuple(nthres,5), np::dtype::get_builtin<float>() );
	evalBoundary( (float*)r.get_data(), (const float *)d.get_data(), (const bool *)bnd.get_data(), W, H, bnd.shape(0), nthres, max_r );
	return r;
}
list evalBoundaryAll( const list &ds, const list &bnds, int nthres, double max_r ) {
	if( len(ds) != len(bnds) )
		throw std::invalid_argument("Detection and Boundaries of different length!");
	
	std::vector<float*> pr(len(ds)), pd(len(ds));
	std::vector<bool*> pbnd(len(ds));
	std::vector<int> W(len(ds)), H(len(ds)), N(len(ds));
	
	list res;
	for( int i=0; i<len(ds); i++ ) {
		np::ndarray d = extract<np::ndarray>(ds[i]);
		np::ndarray bnd = extract<np::ndarray>(bnds[i]);
		checkArray( d, float, 2, 2, true );
		checkArray( bnd, bool, 3, 3, true );
		H[i] = d.shape(0);
		W[i] = d.shape(1);
		N[i] = bnd.shape(0);
		if( H[i] != bnd.shape(1) || W[i] != bnd.shape(2) )
			throw std::invalid_argument( "Ground truth size and boundary size need to have the same shape!\n" );
		np::ndarray r = np::zeros( make_tuple(nthres,5), np::dtype::get_builtin<float>() );
		pd[i] = (float*)d.get_data();
		pbnd[i] = (bool*)bnd.get_data();
		pr[i] = (float*)r.get_data();
		res.append( r );
	}
#pragma omp parallel for ordered schedule(dynamic)
	for( int i=0; i<len(ds); i++ )
		evalBoundary( pr[i], pd[i], pbnd[i], W[i], H[i], N[i], nthres, max_r );
	return res;
}

list evalSegmentBoundaryAll( const list &ds, const list &segs, int nthres, double max_r ) {
	if( len(ds) != len(segs) )
		throw std::invalid_argument("Detection and Boundaries of different length!");
	
	std::vector<float*> pr(len(ds)), pd(len(ds));
	std::vector<int*> pseg(len(ds));
	std::vector<int> W(len(ds)), H(len(ds)), N(len(ds));
	
	list res;
	for( int i=0; i<len(ds); i++ ) {
		np::ndarray d = extract<np::ndarray>(ds[i]);
		np::ndarray seg = extract<np::ndarray>(segs[i]);
		checkArray( d, float, 2, 2, true );
		checkArray( seg, int, 3, 3, true );
		H[i] = d.shape(0);
		W[i] = d.shape(1);
		N[i] = seg.shape(0);
		if( H[i] != seg.shape(1) || W[i] != seg.shape(2) )
			throw std::invalid_argument( "Ground truth size and boundary size need to have the same shape!\n" );
		np::ndarray r = np::zeros( make_tuple(nthres,5), np::dtype::get_builtin<float>() );
		pd[i] = (float*)d.get_data();
		pseg[i] = (int*)seg.get_data();
		pr[i] = (float*)r.get_data();
		res.append( r );
	}
#pragma omp parallel for ordered schedule(dynamic)
	for( int i=0; i<len(ds); i++ ) {
		RMatrixXb mask = RMatrixXb::Zero(H[i],W[i]), bnd = RMatrixXb::Zero(H[i],W[i]);
		for( int r=0; r<H[i]; r++ )
			for( int c=0; c<W[i]; c++ ) {
				mask(r,c) = pseg[i][r*W[i]+c] >= 0;
				if( c && pseg[i][r*W[i]+c] != pseg[i][r*W[i]+c-1] )
					bnd(r,c) = bnd(r,c-1) = 1;
				if( r && pseg[i][r*W[i]+c] != pseg[i][(r-1)*W[i]+c] )
					bnd(r,c) = bnd(r-1,c) = 1;
			}
		const float r2 = max_r*max_r*(W[i]*W[i]+H[i]*H[i]);
		const int sr = sqrt(r2);
		for( int r=0; r<H[i]; r++ )
			for( int c=0; c<W[i]; c++ )
				if( bnd(r,c) ) {
					for( int rr=std::max(-r,-sr); rr<=sr && r+rr < H[i]; rr++ )
						for( int cc=std::max(-c,-sr); cc<=sr && c+cc < W[i]; cc++ )
							if( rr*rr+cc*cc <= r2 )
								mask(r+rr,c+cc) = 1;
				}
		
		thinningGuoHall( bnd );
		
		evalBoundary( pr[i], pd[i], bnd.data(), W[i], H[i], N[i], nthres, max_r, mask.data() );
	}
	return res;
}

