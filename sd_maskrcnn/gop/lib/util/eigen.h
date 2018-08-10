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
#include "win_util.h"
#include <cstdint>
#include <Eigen/Core>
#include <Eigen/SparseCore>
using namespace Eigen;

#define DEFINE_MAT( N )\
typedef N<double>( N ## d );\
typedef N<float>( N ## f );\
typedef N<int32_t>( N ## i );\
typedef N<uint32_t>( N ## u );\
typedef N<int16_t>( N ## s );\
typedef N<uint16_t>( N ## us );\
typedef N<int8_t>( N ## i8 );\
typedef N<uint8_t>( N ## u8 );\
typedef N<bool>( N ## b )

#define DEFINE_MAT2( N )\
typedef N<uint32_t>( N ## u );\
typedef N<int16_t>( N ## s );\
typedef N<uint16_t>( N ## us );\
typedef N<int8_t>( N ## i8 );\
typedef N<uint8_t>( N ## u8 );\
typedef N<bool>( N ## b )

template<typename T> using RowVectorX = Matrix<T,1,Dynamic>;
template<typename T> using RMatrixX = Matrix<T,Dynamic,Dynamic,RowMajor>;
template<typename T> using SMatrixX = SparseMatrix<T>;
template<typename T> using SRMatrixX = SparseMatrix<T,RowMajor>;
template<typename T> using RArrayXX = Array<T,Dynamic,Dynamic,RowMajor>;
template<typename T> using MatrixX = Matrix<T,Dynamic,Dynamic>;
template<typename T> using ArrayXX = Array<T,Dynamic,Dynamic>;
template<typename T> using VectorX = Matrix<T,Dynamic,1>;
template<typename T> using ArrayX = Array<T,Dynamic,1>;

DEFINE_MAT( RMatrixX );
DEFINE_MAT( SMatrixX );
DEFINE_MAT( SRMatrixX );
DEFINE_MAT( RArrayXX );
DEFINE_MAT2( MatrixX );
DEFINE_MAT2( ArrayXX );
DEFINE_MAT2( VectorX );
DEFINE_MAT2( ArrayX );

namespace std{
	template< typename T, int R, int C, int O, int RR, int CC > const T * begin( const Matrix<T,R,C,O,RR,CC> & m ){
		return m.data();
	}
	template< typename T, int R, int C, int O, int RR, int CC > const T * end( const Matrix<T,R,C,O,RR,CC> & m ){
		return m.data()+m.size();
	}
	template< typename T, int R, int C, int O, int RR, int CC > T * begin( Matrix<T,R,C,O,RR,CC> & m ){
		return m.data();
	}
	template< typename T, int R, int C, int O, int RR, int CC > T * end( Matrix<T,R,C,O,RR,CC> & m ){
		return m.data()+m.size();
	}
}
VectorXi range( int end );
VectorXi range( int start, int end );

template<typename T,int C, int R, int O>
void saveMatrixX( std::ostream & s, const Matrix<T,C,R,O> & m ) {
	int rc[2] = {(int)m.rows(),(int)m.cols()};
	s.write( (char*)rc, sizeof(rc) );
	s.write( (char*)m.data(), m.size()*sizeof(T) );
}
template<typename T,int C, int R, int O>
void loadMatrixX( std::istream & s, Matrix<T,C,R,O> & m ) {
	int rc[2];
	s.read( (char*)rc, sizeof(rc) );
	m = Matrix<T,C,R>(rc[0],rc[1]);
	s.read( (char*)m.data(), m.size()*sizeof(T) );
}
