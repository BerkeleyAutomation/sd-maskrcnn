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
#include "berkeley.h"
#ifndef IGNORE_BERKELEY
#include "apng.h"
#include "imgproc/image.h"
#include "matio.h"
#include <dirent.h>
#include <fstream>
#include <sys/stat.h>
#endif

#define XSTR( x ) STR( x )
#define STR( x ) std::string( #x )
#ifdef BERKELEY_DIR
std::string berkeley_dir = XSTR(BERKELEY_DIR);
#else
std::string berkeley_dir = "data/berkeley";
#endif

#ifndef IGNORE_BERKELEY
static std::vector< std::string > listDir(const std::string &dirname) {
	std::vector< std::string > r;
	DIR* dp = opendir( dirname.c_str() );
	if(dp)
		for( dirent * d = readdir( dp ); d != NULL; d = readdir( dp ) )
			r.push_back( d->d_name );
	return r;
}
static int mkdir( const std::string & pathname ){ 
	return mkdir( pathname.c_str(), 0777 );
}

static np::ndarray sgread( const std::string & name ) {
	np::ndarray r(np::empty(make_tuple(0), np::dtype::get_builtin<unsigned char>()));
	mat_t * mat = Mat_Open(name.c_str(),MAT_ACC_RDONLY);
	if(mat)
	{
		matvar_t * gt = Mat_VarRead( mat, (char*)"groundTruth" );
		if(!gt)
			return r;
		int nSeg = gt->dims[1];
		if(!nSeg)
			return r;
		for( int s = 0; s < nSeg; s++ ) {
			matvar_t * cl = Mat_VarGetCell(gt, s);
			matvar_t * seg = Mat_VarGetStructField( cl, (void*)"Segmentation", MAT_BY_NAME, 0 );
			int W = seg->dims[1], H = seg->dims[0];
			if( seg->data_type != MAT_T_UINT16 )
				printf("Unexpected segmentation type! Continuing in denial and assuming uint16_t\n");
			
			if( r.shape(0) <= 0 )
				r = np::empty(make_tuple(nSeg,H,W), np::dtype::get_builtin<unsigned char>());
			
			assert( r.shape(1) == H && r.shape(2) == W );
			const short * pdata = static_cast<const short*>(seg->data);
			unsigned char * pr = reinterpret_cast<unsigned char*>(r.get_data()+s*W*H);
			for( int j=0; j<H; j++ )
				for( int i=0; i<W; i++ )
					pr[j*W+i] = pdata[i*H+j]-1;
		}
		
		Mat_VarFree( gt );
		Mat_Close(mat);
	}
	return r;
}
static np::ndarray bnread( const std::string & name ) {
	np::ndarray r(np::empty(make_tuple(0), np::dtype::get_builtin<unsigned char>()));
	mat_t * mat = Mat_Open(name.c_str(),MAT_ACC_RDONLY);
	if(mat)
	{
		matvar_t * gt = Mat_VarRead( mat, (char*)"groundTruth" );
		if(!gt)
			return r;
		int nSeg = gt->dims[1];
		if(!nSeg)
			return r;
		for( int s = 0; s < nSeg; s++ ) {
			matvar_t * cl = Mat_VarGetCell(gt, s);
			matvar_t * bnd = Mat_VarGetStructField( cl, (void*)"Boundaries", MAT_BY_NAME, 0 );
			
			int W = bnd->dims[1], H = bnd->dims[0];
			if( bnd->data_type != MAT_T_UINT8 )
				printf("Unexpected boundary type! Continuing in denial and assuming uint8_t\n");
			
			if( r.shape(0) <= 0 )
				r = np::empty(make_tuple(nSeg,H,W), np::dtype::get_builtin<unsigned char>());
			
			assert( r.shape(1) == H && r.shape(2) == W );
			const unsigned char * pdata = static_cast<const unsigned char*>(bnd->data);
			unsigned char * pr = reinterpret_cast<unsigned char*>(r.get_data()+s*W*H);
			for( int j=0; j<H; j++ )
				for( int i=0; i<W; i++ )
					pr[j*W+i] = pdata[i*H+j];
		}
		
		Mat_VarFree( gt );
		Mat_Close(mat);
	}
	return r;
}

static dict loadEntry( std::string name ) {
	std::string sets[] = {"train","val","test"};
	std::string im_dir = berkeley_dir+"/images/";
	std::string gt_dir = berkeley_dir+"/groundTruth/";
	std::string cs_dir = berkeley_dir+"/cache/";
	mkdir( cs_dir );
	for (std::string s: sets)
		mkdir( cs_dir+s );
	
	dict r;
	for (std::string s: sets) {
		std::string im_name = s+"/"+name+".jpg";
		std::shared_ptr<Image8u> im = imreadShared( im_dir + "/" + im_name );
		if( im && !im->empty() ) {
			std::string sname = s+"/"+name+".png", bname = s+"/"+name+"_bnd.png", mname = s+"/"+name+".mat";
			
			np::ndarray seg = readAPNG( cs_dir + "/" + sname );
			np::ndarray bnd = readAPNG( cs_dir + "/" + bname );
			
			if( !seg.get_nd() ) {
				seg = sgread( gt_dir + "/" + mname );
				writeAPNG( cs_dir + "/" + sname, seg );
			}
			if( !bnd.get_nd() ) {
				bnd = bnread( gt_dir + "/" + mname );
				writeAPNG( cs_dir + "/" + bname, bnd );
			}
			if( bnd.get_nd() && seg.get_nd() ) {
				r["image"] = im;
				r["segmentation"] = seg;
				r["boundary"] = bnd;
				r["name"] = name;
				return r;
			}
		}
	}
	return r;
}
static void loadBSD300( list & r, const std::string & type ) {
	std::string sets[] = {"train","val","test"};
	std::string cs_dir = berkeley_dir+"/cache/";
	std::ifstream is(berkeley_dir+"/bsd300_iids_"+type+".txt");
	mkdir( cs_dir );
	for (std::string s: sets)
		mkdir( cs_dir+s );
	
	while(is.is_open() && !is.eof()) {
		std::string l;
		std::getline(is,l);
		if( !l.empty() ) {
			dict d = loadEntry( l );
			if( len( d ) )
				r.append( d );
		}
	}
}
#endif
list loadBSD300( bool train, bool valid, bool test ) {
	list r;
#ifndef IGNORE_BERKELEY
	if( train )
		loadBSD300( r, "train");
	if( test )
		loadBSD300( r, "test");
#else
	throw std::invalid_argument("matio needed to load the BSD");
#endif
	return r;
}
#ifndef IGNORE_BERKELEY
static void loadBSD500( list & r, const std::string & type, int max_entry=1<<30 ) {
	std::string sets[] = {"train","val","test"};
	std::string cs_dir = berkeley_dir+"/cache/";
	std::string im_dir = berkeley_dir+"/images/"+type+"/";
	
	mkdir( cs_dir );
	for (std::string s: sets)
		mkdir( cs_dir+s );
	
	std::vector<std::string> files = listDir( im_dir );
	std::sort( files.begin(), files.end() );
	int n = 0;
	for( std::string fn: files )
		if( fn.size() > 4 && fn.substr(fn.size()-4)==".jpg" ) {
			dict d = loadEntry( fn.substr(0,fn.size()-4) );
			if( len( d ) ) {
				r.append( d );
				n++;
				if( n >= max_entry )
					break;
			}
		}
}
#endif
list loadBSD500( bool train, bool valid, bool test ) {
	list r;
#ifndef IGNORE_BERKELEY
	if( train )
		loadBSD500( r, "train");
	if( valid )
		loadBSD500( r, "val");
	if( test )
		loadBSD500( r, "test");
#else
	throw std::invalid_argument("matio needed to load the BSD");
#endif
	return r;
}
list loadBSD50( bool train, bool valid, bool test ) {
	list r;
#ifndef IGNORE_BERKELEY
	if( train )
		loadBSD500( r, (std::string)"train", 20);
	if( valid )
		loadBSD500( r, (std::string)"val", 10);
	if( test )
		loadBSD500( r, (std::string)"test", 20);
#else
	throw std::invalid_argument("matio needed to load the BSD");
#endif
	return r;
}

