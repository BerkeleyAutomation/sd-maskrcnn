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
#include <memory>
#include <stdexcept>
#include "util/eigen.h"

namespace ImagePrivate {
	template<typename T> struct ImageHelper {
		static const int range = 1;
	};
	template<> struct ImageHelper<uint8_t> {
		static const int range = 255;
	};
	template<> struct ImageHelper<uint16_t> {
		static const int range = 65535;
	};
}


template<typename T>
class TypedImage {
public:
	typedef T value_type;
protected:
	template<typename TO> friend class TypedImage;
	
	T * data_;
	bool external_data_;
	int W_, H_, C_;
public:
	TypedImage( T * data, int W=0, int H=0, int C=1 ):data_(data),external_data_(true),W_(W),H_(H),C_(C){
	}
	TypedImage( int W=0, int H=0, int C=1 ):data_(NULL),external_data_(false),W_(0),H_(0),C_(0){
		create( W, H, C );
	}
	TypedImage( const TypedImage<T> & o ):data_(NULL),external_data_(false),W_(0),H_(0),C_(0) {
		if( o.external_data_ ) {
			external_data_ = o.external_data_;
			data_ = o.data_;
		}
		else {
			create( o.W_, o.H_, o.C_ );
			memcpy( data_, o.data_, W_*H_*C_*sizeof(T) );
		}
	}
template<typename TO>
	TypedImage( const TypedImage<TO>& o ):data_(NULL),external_data_(false),W_(0),H_(0),C_(0) {
		using namespace ImagePrivate;
		const float f = (float)ImageHelper<T>::range / (float)ImageHelper<TO>::range;
		create( o.W_, o.H_, o.C_ );
		for( int i=0; i<W_*H_*C_; i++ )
			data_[i] = f*o.data_[i];
	}
	TypedImage & operator=( const TypedImage<T> & o ) {
		create( o.W_, o.H_, o.C_ );
		memcpy( data_, o.data_, W_*H_*C_*sizeof(T) );
		return *this;
	}
	~TypedImage() {
		if( data_ && !external_data_ )
			delete [] data_;
	}
	TypedImage<T> copy() const {
		TypedImage<T> r(W_, H_, C_);
		memcpy( r.data_, data_, W_*H_*C_*sizeof(T) );
		return r;
	}
template<typename TO>
	TypedImage<T> & operator=( const TypedImage<TO>& o ) {
		using namespace ImagePrivate;
		const float f = (float)ImageHelper<T>::range / (float)ImageHelper<TO>::range;
		create( o.W_, o.H_, o.C_ );
		for( int i=0; i<W_*H_*C_; i++ )
			data_[i] = f*o.data_[i];
		return *this;
	}
	void create( int W, int H, int C ) {
		if( external_data_ && ( W_!=W || H_!=H || C_!=C ) )
			throw std::invalid_argument( "Cannot resize external image" );
		if( W_*H_*C_ != W*H*C || W_*H_*C_ == 0) {
			if( data_!=NULL )
				delete[] data_;
			data_ = new T[W*H*C+1];
		}
		W_ = W;
		H_ = H;
		C_ = C;
	}
	
	const T * data() const {
		return data_;
	}
	T * data() {
		return data_;
	}
	const T & operator[]( int i ) const {
		return data_[i];
	}
	T & operator[]( int i ) {
		return data_[i];
	}
	const T & operator()( int y, int x, int c ) const {
		return data_[(y*W_+x)*C_+c];
	}
	T & operator()( int y, int x, int c ) {
		return data_[(y*W_+x)*C_+c];
	}
	
template<int N>
	Matrix<T,N,1> at( int y, int x ) const {
		return Matrix<T,N,1>::Map( data_+(y*W_+x)*C_ );
	}
	
	operator const RMatrixX<T>() const {
		if( 1!=C_ )
			throw std::invalid_argument( "Matrix and Image channels do not match!" );
		return Map< const RMatrixXf >( data(), H_, W_ );
	}
	TypedImage<T> & operator=( const RMatrixX<T> & o ) {
		if( 1!=C_ )
			throw std::invalid_argument( "Matrix and Image channels do not match!" );
		Map< RMatrixX<T> >( data(), H_, W_ ) = o;
		return *this;
	}
template<int N>
	const RMatrixX< Matrix<T,N,1> > toMatrix() const {
		if( N!=C_ )
			throw std::invalid_argument( "Matrix and Image channels do not match!" );
		return Map< const RMatrixX< Matrix<T,N,1> > >( (Matrix<T,N,1>*)data(), H_, W_ );
	}
template<int N>
	TypedImage<T> & operator=( const RMatrixX< Matrix<T,N,1> > & o ) {
		if( N!=C_ )
			throw std::invalid_argument( "Matrix and Image channels do not match!" );
		Map< RMatrixX< Matrix<T,N,1> > >( (Matrix<T,N,1>*)data(), H_, W_ ) = o;
		return *this;
	}
	TypedImage<T> & operator=( T v ){
		std::fill( begin(), end(), v );
		return *this;
	}
	int W() const { return W_; }
	int H() const { return H_; }
	int C() const { return C_; }
	bool empty() const { return W_*H_*C_==0; }
	T * begin() { return data_; }
	T * end() { return data_+W_*H_*C_; }
	const T * cbegin() const { return data_; }
	const T * cend() const { return data_+W_*H_*C_; }
	
	TypedImage<T> tileC( int C ) const {
		TypedImage<T> r( W_, H_, C );
		for( int i=0; i<W_*H_; i++ )
			for( int c=0; c<C; c++ )
				r.data_[i*C+c] = data_[i*C_+(c%C_)];
		return r;
	}
	
	virtual void save( std::ostream & s ) const {
		int sz[3] = {W_,H_,C_};
		s.write((const char*)sz,sizeof(sz));
		s.write((const char*)data_, W_*H_*C_*sizeof(T));
	}
	virtual void load( std::istream & s ) {
		int sz[3];
		s.read((char*)sz,sizeof(sz));
		create( sz[0], sz[1], sz[2] );
		s.read((char*)data_, W_*H_*C_*sizeof(T));
	}
};

typedef TypedImage<float> Image;
typedef TypedImage<uint8_t> Image8u;

#ifndef NO_IMREAD
Image8u imread( const std::string &file_name );
std::shared_ptr<Image8u> imreadShared( const std::string &file_name );
void imwrite( const std::string & file_name, const Image8u & im );
#endif