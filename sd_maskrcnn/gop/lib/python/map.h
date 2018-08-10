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
#include "util/eigen.h"
#include "util.h"

template< typename T >
Map< RMatrixX<T> > mapMatrixX(const np::ndarray & f) {
	checkArray( f, T, 2, 2, true );
	return Map< RMatrixX<T> >( (T*)f.get_data(), f.shape(0), f.shape(1) );
}
template< typename T >
Map< Matrix<T,Dynamic,1> > mapVectorX(const np::ndarray & f) {
	checkArray( f, T, 1, 1, true );
	return Map< Matrix<T,Dynamic,1> >( (T*)f.get_data(), f.shape(0) );
}
template< typename T, int N, int M >
np::ndarray toNumpy( const Matrix<T,N,M> & m ) {
	np::ndarray r = np::empty( make_tuple( m.rows(), m.cols() ), np::dtype::get_builtin<T>() );
	mapMatrixX<T>(r) = m;
	return r;
}
template< typename T, int N, int M >
np::ndarray toNumpy( const Matrix<T,N,M,RowMajor> & m ) {
	np::ndarray r = np::empty( make_tuple( m.rows(), m.cols() ), np::dtype::get_builtin<T>() );
	mapMatrixX<T>(r) = m;
	return r;
}
template< typename T, int N >
np::ndarray toNumpy( const Matrix<T,N,1> & m ) {
	np::ndarray r = np::empty( make_tuple( m.rows() ), np::dtype::get_builtin<T>() );
	mapVectorX<T>(r) = m;
	return r;
}
template< typename T, int N >
np::ndarray toNumpy( const Matrix<T,1,N> & m ) {
	np::ndarray r = np::empty( make_tuple( m.rows() ), np::dtype::get_builtin<T>() );
	mapVectorX<T>(r) = m;
	return r;
}

#include "imgproc/image.h"
Image mapImage( np::ndarray & f );
Image mapImage( const np::ndarray & f );
np::ndarray toNumpy( const Image & m );

template<typename T>
list to_list( const std::vector<T> & l ) {
	list r;
	for( const T& i: l )
		r.append( i );
	return r;
}
template<typename T>
std::vector<T> to_vector( const list & l ) {
	std::vector<T> r;
	for( int i=0; i<len(l); i++ )
		r.push_back( extract<T>(l[i]) );
	return r;
}


