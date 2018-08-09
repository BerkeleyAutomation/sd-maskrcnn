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
#include "weizmann.h"
#include "imgproc/image.h"
#include "imgproc/resample.h"
#include <dirent.h>
#include <fstream>
#include <sys/stat.h>

static std::vector< std::string > listDir(const std::string &dirname) {
	std::vector< std::string > r;
	DIR* dp = opendir( dirname.c_str() );
	if(dp)
		for( dirent * d = readdir( dp ); d != NULL; d = readdir( dp ) )
			if( d->d_name[0] != '.' )
				r.push_back( d->d_name );
	return r;
}

#define XSTR( x ) STR( x )
#define STR( x ) std::string( #x )
#ifdef WEIZMANN_DIR
std::string weizmann_dir = XSTR(WEIZMANN_DIR);
#else
std::string weizmann_dir = "data/weizmann_horse_db";
#endif

static dict loadEntry( std::string name ) {
	std::string im_dir = weizmann_dir+"/rgb/";
	std::string gt_dir = weizmann_dir+"/figure_ground/";

	dict r;
	if( name.length()<=4 || name.substr(name.length()-4) != ".jpg" )
		return r;
	
	std::shared_ptr<Image8u> im = imreadShared( im_dir + "/" + name );
	std::shared_ptr<Image8u> gt = imreadShared( gt_dir + "/" + name );
	if( im && gt && !im->empty() && !gt->empty() ) {
		std::string s_name = name.substr(0,name.length()-4);
		
		// some images are too large, so we should resize them
		if( im->W() != gt->W() || im->H() != im->W() )
			im = std::make_shared<Image8u>( resize( *im, gt->W(), gt->H() ) );
		
		RMatrixXs seg( gt->H(), gt->W() );
		for( int j=0; j<im->H(); j++ )
			for( int i=0; i<im->W(); i++ )
				seg(j,i) = ( (float)(*gt)(j,i,0)+(*gt)(j,i,1)+(*gt)(j,i,2) > 127*3 );
		
		r["image"] = im;
		r["segmentation"] = seg;
		r["name"] = s_name;
	}
	return r;
}
list loadWeizmann( bool train, bool test, int n_train ) {
	std::vector< std::string > file_names = listDir( weizmann_dir+"/rgb/" );
	std::sort( file_names.begin(), file_names.end() );
	
	std::mt19937 rand;
	for( int i=1; i<(int)file_names.size(); i++ )
		std::swap( file_names[i], file_names[rand()%(i+1)] );
	
	list r;
	for( int i=train?0:n_train; (test || i<n_train) && i<file_names.size(); i++ ) {
		dict d = loadEntry( file_names[i] );
		if( len(d) )
			r.append( d );
	}
	return r;
}

