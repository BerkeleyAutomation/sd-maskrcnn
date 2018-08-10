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
#include "algorithm.h"
#include <cstdlib>
#include <tuple>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <random>
#include <chrono>

template<typename itf>
std::tuple<itf,itf> split( itf a, itf b, float p ) {
	for(itf i=a; i<b; i++ ) {
		if(*i == p)
			std::swap(*i,*(--b));
		if( *i < p )
			std::swap(*i,*(a++));
	}
	return std::tie(a,b);
}
template<typename itf>
float runQuickSelect( itf a, itf b, int i ) {
	static std::mt19937 gen;
	if( a+1==b )
		return *a;
	float p = *(a+gen()%(b-a));
	itf s, e;
	std::tie(s,e) = split<itf>(a,b,p);
	int n1 = s-a, np = b-e;
	if( i < n1 )
		return runQuickSelect(a,s,i);
	if( i < n1+np )
		return p;
	return runQuickSelect(s,e,i-n1-np);
}
float quickSelect( std::vector< float > &elements, int i ) {
	return runQuickSelect( elements.begin(), elements.end(), i );
}
float quickSelect( const std::vector< float > &elements, int i ) {
	std::vector<float> o = elements;
	return quickSelect( o, i );
}
float quickSelect( VectorXf &elements, int i ) {
	return runQuickSelect( elements.data(), elements.data()+elements.size(), i );
}
float quickSelect( const VectorXf &elements, int i ) {
	VectorXf o = elements;
	return quickSelect( o, i );
}
unsigned int AlgomorativeSum::round_pow2( unsigned int v ) {
	v--;
	v |= v >> 1;
	v |= v >> 2;
	v |= v >> 4;
	v |= v >> 8;
	v |= v >> 16;
	v++;
	return v;
}
AlgomorativeSum::AlgomorativeSum( int N ):sum_left_(round_pow2( N )) {

}
void AlgomorativeSum::resize( int N ) {
	if( N > sum_left_.size() ) {
		int old_N = sum_left_.size();
		N = round_pow2( N );
		sum_left_.resize( N, 0 );
		sum_left_[N-1] = sum_left_[old_N-1];
	}
}
int AlgomorativeSum::size() const {
	return sum_left_.size();
}
void AlgomorativeSum::add( int i, float v ) {
	if( i>= sum_left_.size() )
		resize( i+1 );
	i++;
	int k = 1;
	while( i <= sum_left_.size() ) {
		sum_left_[i-1] += v;
		while( ( k & i ) == 0 && k <= sum_left_.size() )
			k <<= 1;
		i += k;
	}
}
float AlgomorativeSum::get( int i ) const {
	i++;
	if( i>sum_left_.size() )
		i = sum_left_.size();
	int k = 1;
	float r = 0;
	while( i > 0 ) {
		r += sum_left_[i-1];
		while( ( k & i ) == 0 && k <= sum_left_.size() )
			k <<= 1;
		i ^= k;
	}
	return r;
}
UnionFindSet::UnionFindSet( int N ) : id_( N ) {
	for( int i = 0; i < N; i++ )
		id_[i] = i;
}
int UnionFindSet::find( int p ) {
	int root = p;
	for( root = p; root != id_[root]; root = id_[root] )
		;
	while( p != root ) {
		int pp = id_[p];
		id_[p] = root;
		p = pp;
	}
	return root;
}
int UnionFindSet::merge( int x, int y )    {
	int i = find( x );
	int j = find( y );
	if( i == j )
		return i;

	if( i > j )
		std::swap( i, j );
	id_[j] = i;
	return i;
}
VectorXi arange( int n ) {
	VectorXi r( n );
	for( int i = 0; i < n; i++ )
		r[i] = i;
	return r;
}
VectorXi randomChoose( int M, int N ) {
	static std::mt19937 gen;
	if (N>M) N = M;
	VectorXi r( N );
	if( 2 * N < M ) {
		std::uniform_int_distribution<> dis( 0, M - 1 );
		// It's probably faster to randomly sample and check for repetitions
		std::unordered_set<int> sampled;
		for( int i = 0; i < N; i++ ) {
			int r = dis( gen );
			while( sampled.count( r ) > 0 )
				r = dis( gen );
			sampled.insert( r );
		}
		int k = 0;
		for( int i : sampled )
			r[k++] = i;
	}
	else {
		VectorXi vv = arange( M );
		for( int i = 0; i < N; i++ ) {
			int j = i + ( gen() % ( N - i ) );
			std::swap( vv[i], vv[j] );
			r[i] = vv[i];
		}
	}
	return r;

}
RMatrixXf kmeans(const RMatrixXf &v, int K, int nit, int seed ) {
	std::mt19937 rand( seed );
	std::uniform_real_distribution<float> udist(0.0, 1.0);
	const int N = v.rows(), D = v.cols();
	RMatrixXf r = RMatrixXf::Zero( K, D );
	float best_e = 1e10;
	
	for( int it=0; it<nit; it++ ) {
		ArrayXf d = 1e10*ArrayXf::Ones( N );
		ArrayXi a = -ArrayXi::Ones( N );
		
		RMatrixXf means( K, D );
		
		// Find the initial seeds (K-means++)
		for( int i=0; i<K; i++ ) {
			ArrayXf p = d / d.maxCoeff();
			float s = p.sum()*udist(rand);
			int j;
			for( j=0; s > 0 && j+1<N; j++ )
				s -= p[j];
			means.row(i) = v.row(j);
			ArrayXf nd = (v.rowwise() - means.row(i)).rowwise().squaredNorm();
			a = (nd < d).select( i, a );
			d = d.min( nd );
		}
		// Run K-means
		float e = d.sum();
		while(1) {
			// Estimate the centers
			for( int i=0; i<K; i++ )
				means.row(i) = ((a==i).cast<float>().matrix().transpose()*v).array() / ((a==i).cast<float>().array().sum()+1e-10);
			
			// Estimate the assignment
			d.setConstant( 1e10 );
			for( int i=0; i<K; i++ ) {
				ArrayXf nd = (v.rowwise() - means.row(i)).rowwise().squaredNorm();
				a = (nd < d).select( i, a );
				d = d.min( nd );
			}
			const float new_e = d.sum();
			if( new_e >= e )
				break;
			e = new_e;
		}
		if( d.sum() < best_e ) {
			best_e = d.sum();
			r = means;
		}
	}
	return r;
}

