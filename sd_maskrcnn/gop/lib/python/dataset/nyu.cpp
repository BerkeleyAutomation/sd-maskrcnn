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
#include "nyu.h"
#include "imgproc/image.h"
#include "apng.h"
#include "python/map.h"
#include <string>
#include <fstream>
#include <unordered_map>

#define XSTR( x ) STR( x )
#define STR( x ) std::string( #x )
#ifdef NYU_DIR
std::string nyu_dir = std::string(XSTR(NYU_DIR))+"/";
#else
std::string nyu_dir = "./";
#endif
const float DEPTH_SCALE = 6400.f;

static void remapInst( np::ndarray& olbl, const np::ndarray& clbl ) {
	const short * pc = (const short*)clbl.get_data();
	short * po = (short*)olbl.get_data();
	const int N = olbl.shape(0)*olbl.shape(1);
	// Compress the initial ranges
	short rc[1<<16], ro[1<<16], nc=0, no=0;
	memset( rc, -1, sizeof(rc) );
	memset( ro, -1, sizeof(ro) );
	for( int i=0; i<N; i++ ) {
		if( ro[po[i]]<0 ) ro[po[i]] = no++;
		if( rc[pc[i]]<0 ) rc[pc[i]] = nc++;
	}
	short r[1<<16];
	memset( r, -1, sizeof(r) );
	for( int i=0,n=0; i<N; i++ ) {
		int id = ro[po[i]]*nc+rc[pc[i]];
		if( r[id]<0 ) r[id] = n++;
		po[i] = r[id];
	}
}
static void remapClass( np::ndarray & clbl, const std::vector<short> & class_set ) {
	short * pc = (short*)clbl.get_data();
	const int N = clbl.shape(0)*clbl.shape(1);
	for( int i=0; i<N; i++ )
		if( 0 < pc[i] && pc[i] < class_set.size() )
			pc[i] = class_set[ pc[i]-1 ];
		else
			pc[i] = pc[i]-1;
}
static Image8u crop( const Image8u & a, int x, int y, int W, int H ) {
	if( x>=0 && y>=0 && W>0 && H>0 ) {
		const int C = a.C();
		Image8u r(W,H,C);
		
		for( int i=0; i<H; i++ )
			memcpy( r.data() + i*W*C, a.data()+(i+y)*a.W()*C+x*C, W*C*sizeof(uint8_t) );
		return r;
	}
	return a;
}
static np::ndarray crop( const np::ndarray & a, int x, int y, int W, int H ) {
	if( x>=0 && y>=0 && W>0 && H>0 ) {
		std::vector<Py_intptr_t> d(a.get_nd());
		for( int i=0; i<(int)d.size(); i++ )
			d[i] = a.shape(i);
		d[0] = H;
		d[1] = W;
		np::ndarray r = np::zeros( d.size(), d.data(), a.get_dtype() );
		char * pr = r.get_data();
		const char * pa = a.get_data();
		const size_t es = r.strides(1);
		const size_t Ws = r.strides(0), Wsa = a.strides(0);
		for( int i=0; i<H; i++ )
			memcpy( pr+i*Ws, pa+(i+y)*Wsa+x*es, Ws );
		return r;
	}
	return a;
}
static dict loadEntry( const std::string & name, const std::vector<short> & class_set, int x=-1, int y=-1, int W=-1, int H=-1 ) {
	const std::string NYU_IMAGES = nyu_dir + "/%s_rgb.png";
	const std::string NYU_DEPTHS = nyu_dir + "/%s_d.png";
	const std::string NYU_LABELS = nyu_dir + "/%s_lbl.png";
	const std::string NYU_INSTANCES = nyu_dir + "/%s_ins.png";
	const std::string NYU_INFO   = nyu_dir + "/%s_info.png";
	
	dict r;
	char buf[1024];
	sprintf( buf, NYU_LABELS.c_str(), name.c_str() );
	np::ndarray clbl = readIPNG( buf, 16 );
	if( !clbl.get_nd() )
		return dict();
	
	sprintf( buf, NYU_INSTANCES.c_str(), name.c_str() );
	np::ndarray olbl = readIPNG( buf, 16 );
	if( !olbl.get_nd() )
		return dict();
	clbl = crop( clbl, x, y, W, H );
	olbl = crop( olbl, x, y, W, H );
	remapInst(olbl,clbl);
	remapClass( clbl, class_set );
	r["class"] = clbl;
	r["segmentation"] = olbl;
	
	sprintf( buf, NYU_IMAGES.c_str(), name.c_str() );
	std::shared_ptr<Image8u> im = imreadShared( buf );
	if( !im || im->empty() )
		return dict();
	r["image"] = crop( *im, x, y, W, H );
	
	sprintf( buf, NYU_DEPTHS.c_str(), name.c_str() );
	np::ndarray d = toNumpy( RMatrixXf(readPNG16( buf ).cast<float>().array() / DEPTH_SCALE) );
	if( !d.get_nd() )
		return dict();
	r["depth"] = crop( d, x, y, W, H );
	
	sprintf( buf, NYU_INFO.c_str(), name.c_str() );
	std::ifstream is( buf );
	std::string tmp;
	std::getline( is, tmp ); r["type"] = tmp;
	std::getline( is, tmp ); r["scene"] = tmp;
	std::getline( is, tmp ); r["accel"] = tmp;
	
	r["name"] = name;
	return r;
}
static list loadDataset( bool train, bool valid, bool test, const std::vector<short> & class_set, int x=-1, int y=-1, int W=-1, int H=-1 ) {
	bool read[3]={train,valid,test};
	std::string fn[3]={"/train.txt","","/test.txt"};
	list r;
	for( int i=0; i<3; i++ ) 
		if( read[i] ){
			std::ifstream is(nyu_dir+"/"+fn[i]);
			while(is.is_open() && !is.eof()) {
				std::string l;
				std::getline(is,l);
				if( !l.empty() ) {
					dict d = loadEntry( l, class_set, x, y, W, H );
					if( len( d ) )
						r.append( d );
				}
			}
		}
	return r;
}
static void readCrop( const std::string & fn, int & x, int & y, int & W, int & H ) {
	x=y=W=H=-1;
	std::ifstream is(fn);
	while(is.is_open() && !is.eof()) {
		std::string l;
		std::getline(is,l);
		if( l.size()>0 && l[0]!='#' ) {
			std::stringstream ss(l);
			ss >> x >> y >> W >> H;
		}
	}
}
static std::vector<short> readRemap( const std::string & fn ) {
	std::vector<short> r;
	std::ifstream is(fn);
	while(is.is_open() && !is.eof()) {
		std::string l;
		std::getline(is,l);
		if( l.size()>0 )
			r.push_back( std::stoi(l)-1 );
	}
	return r;
}
list loadNYU( bool train, bool valid, bool test ) {
	int x, y, W, H;
	readCrop( nyu_dir+"/crop.txt", x, y, W, H );
	return loadDataset( train, valid, test, std::vector<short>(), x, y, W, H );
}
list loadNYU04( bool train, bool valid, bool test ) {
	int x, y, W, H;
	readCrop( nyu_dir+"/crop.txt", x, y, W, H );
	return loadDataset( train, valid, test, readRemap( nyu_dir+"/id_map_40.txt" ), x, y, W, H );
}
list loadNYU40(bool train, bool valid, bool test) {
	int x, y, W, H;
	readCrop( nyu_dir+"/crop.txt", x, y, W, H );
	return loadDataset( train, valid, test, readRemap( nyu_dir+"/id_map_04.txt" ), x, y, W, H );
}
list loadNYU_nocrop( bool train, bool valid, bool test ) {
	return loadDataset( train, valid, test, std::vector<short>() );
}
list loadNYU04_nocrop( bool train, bool valid, bool test ) {
	return loadDataset( train, valid, test, readRemap( nyu_dir+"/id_map_40.txt" ) );
}
list loadNYU40_nocrop(bool train, bool valid, bool test) {
	return loadDataset( train, valid, test, readRemap( nyu_dir+"/id_map_04.txt" ) );
}
static list readNames( const std::string & fn ) {
	list r;
	std::ifstream is(fn);
	while(is.is_open() && !is.eof()) {
		std::string l;
		std::getline(is,l);
		if( l.back() == '\n' ) l.pop_back();
		if( l.size()>0 )
			r.append( l );
	}
	return r;
}
list labelsNYU() {
	return readNames( nyu_dir+"/names.txt" );
}
list labelsNYU04() {
	return readNames( nyu_dir+"/names_04.txt" );
}
list labelsNYU40() {
	return readNames( nyu_dir+"/names_40.txt" );
}

