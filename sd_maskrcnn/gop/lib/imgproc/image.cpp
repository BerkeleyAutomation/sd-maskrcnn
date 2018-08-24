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
#include "image.h"
#ifndef NO_IMREAD
#define cimg_display 0
#define cimg_use_jpeg
#define cimg_use_png
#define PNG_SKIP_SETJMP_CHECK
#include "CImg.h"
using namespace cimg_library;
#endif
#include <cstdio>

static bool exists( const std::string &file_name ) {
	FILE* fp = fopen( file_name.c_str(), "r" );
	if( !fp )
		return false;
	fclose( fp );
	return true;
}
#ifndef NO_IMREAD
Image8u imread( const std::string &file_name ) {
	if(!exists( file_name ))
		return Image8u();
	CImg<unsigned char> im( file_name.c_str() );
	int W = im.width(), H = im.height(), C = im.spectrum();
	Image8u r(W,H,C);
	for( int j=0,l=0; j<H; j++ )
		for( int i=0; i<W; i++ )
			for( int k=0; k<C; k++,l++ )
				r[l] = im(i,j,0,k);
	return r;
}
std::shared_ptr<Image8u> imreadShared( const std::string &file_name ) {
	if(!exists( file_name ))
		return std::shared_ptr<Image8u>();
	CImg<unsigned char> im( file_name.c_str() );
	int W = im.width(), H = im.height(), C = im.spectrum();
	std::shared_ptr<Image8u> r = std::make_shared<Image8u>(W,H,C);
	for( int j=0,l=0; j<H; j++ )
		for( int i=0; i<W; i++ )
			for( int k=0; k<C; k++,l++ )
				(*r)[l] = im(i,j,0,k);
	return r;
}

void imwrite( const std::string & file_name, const Image8u & im ) {
	CImg<unsigned char> r( im.W(), im.H(), 1, im.C() );
	for( int j=0,l=0; j<im.H(); j++ )
		for( int i=0; i<im.W(); i++ )
			for( int k=0; k<im.C(); k++,l++ )
				r(i,j,0,k) = im[l];
	r.save( file_name.c_str() );
}
#endif
