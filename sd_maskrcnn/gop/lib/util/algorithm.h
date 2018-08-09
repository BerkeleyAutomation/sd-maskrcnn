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
#pragma once
#include <vector>
#include <random>
#include <unordered_set>
#include "eigen.h"
#include "threading.h"

float quickSelect( const std::vector<float> & elements, int i );
float quickSelect( std::vector<float> & elements, int i );
float quickSelect( const VectorXf & elements, int i );
float quickSelect( VectorXf & elements, int i );

class UnionFindSet {
	std::vector< int > id_;
public:
	UnionFindSet( int N=0 );
	int find( int p );
	int merge(int x, int y);
};


class AlgomorativeSum {
protected:
	std::vector<float> sum_left_;
	unsigned int round_pow2( unsigned int v );
public:
	AlgomorativeSum( int N=0 );
	void resize( int N );
	int size() const;
	void add( int i, float v );
	float get( int i ) const;
};

template<typename T>
class ProbabilisticQueue {
protected:
	std::vector<T> v_;
	std::vector<float> p_;
	AlgomorativeSum asm_;
	double sum_p_;
	static std::default_random_engine rand_;
public:
	ProbabilisticQueue() : sum_p_( 0 ) {
	}
	bool empty() const {
		return sum_p_ < 1e-3;
	}
	void push( const T & v, float p ) {
		v_.push_back( v );
		p_.push_back( p );
		asm_.add( p_.size() - 1, p );
		sum_p_ += p;
	}
	int pick() const {
		std::uniform_real_distribution<float> distribution( 0.0, sum_p_ );
		float v = distribution( rand_ );
		// Do a binary search
		int i0 = 0, i1 = p_.size() - 1;
		while( i0 < i1 ) {
			int i = ( i0 + i1 ) / 2;
			if( asm_.get( i ) < v )
				i0 = i + 1;
			else
				i1 = i;
		}
		return i0;
	}
	void pop( int i ){
		asm_.add( i, -p_[i] );
		sum_p_ -= p_[i];
		p_[i] = 0;
	}
	T pop() {
		int i = pick();
		pop( i );
		return get( i );
	}
	const T & get( int i ) const {
		return v_[i];
	}
};
template<typename T>
std::default_random_engine ProbabilisticQueue<T>::rand_;

VectorXi randomChoose( int M, int N );

template<typename T>
std::vector<T> randomChoose( const std::vector<T> & v, int N ) {
	VectorXi c = randomChoose( v.size(), N );
	std::vector<T> r( N );
	for( int i=0; i<N; i++ )
		r[i] = v[c[i]];
	return r;
}
template<typename T, int R>
Matrix<T,R,1> randomChoose( const Matrix<T,R,1> & v, int N ) {
	VectorXi c = randomChoose( v.size(), N );
	Matrix<T,-1,1> r( N );
	for( int i=0; i<N; i++ )
		r[i] = v[c[i]];
	return r;
}

template<typename T>
T max( const std::vector<T> & v ) {
	T r = v[0];
	for( T i: v )
		if( r<i )
			r = i;
	return r;
}

VectorXi arange( int n );

RMatrixXf kmeans( const RMatrixXf & v, int K, int nit=3, int seed=0 );
