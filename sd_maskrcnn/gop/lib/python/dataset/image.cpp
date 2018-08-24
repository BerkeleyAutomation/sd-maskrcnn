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
//#include "image.h"
//#define cimg_display 0
//#define cimg_use_jpeg
//#define cimg_use_png
//#define PNG_SKIP_SETJMP_CHECK
//#include "CImg.h"
//#include <cstdio>
//using namespace cimg_library;
//
//static bool exists( const std::string &file_name ) {
//	FILE* fp = fopen( file_name.c_str(), "r" );
//	if( !fp )
//		return false;
//	fclose( fp );
//	return true;
//}
//np::ndarray imread( const std::string &file_name ) {
//	if(!exists( file_name ))
//		return np::ndarray(np::empty(make_tuple(0),np::dtype::get_builtin<unsigned char>()));
//	CImg<unsigned char> im( file_name.c_str() );
//	int W = im.width(), H = im.height(), C = im.spectrum();
//	if(C>1) {
//		np::ndarray r = np::zeros( make_tuple(H,W,C), np::dtype::get_builtin<unsigned char>() );
//		for( int j=0; j<H; j++ )
//			for( int i=0; i<W; i++ )
//				for( int k=0; k<C; k++ )
//					r.get_data()[(j*W+i)*C+k] = im(i,j,0,k);
//		return r;
//	}
//	else {
//		np::ndarray r = np::zeros( make_tuple(H,W), np::dtype::get_builtin<unsigned char>() );
//		std::copy( im.data(), im.data()+W*H, r.get_data() );
//		return r;
//	}
//}
//
//void imwrite( const std::string & file_name, const np::ndarray & im ) {
//	int W, H, C=1;
//	if( im.get_dtype() != np::dtype::get_builtin<unsigned char>() )
//		throw std::invalid_argument( "Currently only unsigned char image supported!" );
//	if( im.get_nd() < 2 )
//		throw std::invalid_argument( "Image needs at least 2 dimensions" );
//	if( im.get_nd() > 3 )
//		throw std::invalid_argument( "Image has too many dimensions (>3)" );
//	W = im.shape(1);
//	H = im.shape(0);
//	if( im.get_nd() > 2)
//		C = im.shape(2);
//	CImg<unsigned char> r( W, H, 1, C );
//	for( int j=0; j<H; j++ )
//		for( int i=0; i<W; i++ )
//			for( int k=0; k<C; k++ )
//				r(i,j,0,k) = im.get_data()[(j*W+i)*C+k];
//	r.save( file_name.c_str() );
//}
//bool empty(const np::ndarray &a)
//{
//	if (!a.get_nd())
//		return true;
//	for (int i = 0; i < a.get_nd(); i++)
//		if (a.shape(i) <= 0)
//			return true;
//	return false;
//}
