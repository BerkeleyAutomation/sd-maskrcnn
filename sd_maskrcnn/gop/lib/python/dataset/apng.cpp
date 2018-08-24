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
#include <fstream>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <zlib.h>
#include <cstring>
#include <vector>
#include "apng.h"

static const bool CHECK_CRC = true;

uint16_t htons2( uint16_t v ) {
	return htons( v );
}

template<bool typed=false>
static std::vector<char> readChunk( std::ifstream & is, std::string & type ) {
	uint32_t len , crc;
	auto pos = is.tellg();
	
	is.read( (char*)&len, 4 );
	len = ntohl( len );
	if( len <= 0 )
		return std::vector<char>();
	
	char tag[5] = {0};
	is.read( tag, 4 );
	if( typed && type != tag ) {
		is.seekg( pos );
		return std::vector<char>();
	}
	type = tag;
	
	std::vector<char> buf( len );
	is.read( (char*)buf.data(), len );
	is.read( (char*)&crc, 4 );
	
	if( CHECK_CRC ) {
		uint32_t check_crc = 0L;
		check_crc = crc32(check_crc, (const Bytef*)type.data(), 4);
		if( buf.size() )
			check_crc = crc32(check_crc, (const Bytef*)buf.data(), buf.size());
		check_crc = ntohl( check_crc );
		if( crc != check_crc )
			printf("CRC failed!\n");
	}
	
	return buf;
}
static void writeChunk( std::ofstream & os, const std::string & type, const std::vector<char> & data ) {
	uint32_t len=ntohl( data.size() ), crc = 0L;
	crc = crc32(crc, (const Bytef*)type.data(), 4);
	if( data.size() )
		crc = crc32(crc, (const Bytef*)data.data(), data.size());
	crc = ntohl( crc );
	os.write( (const char*)&len, 4 );
	os.write( type.data(), 4 );
	os.write( (const char*)data.data(), data.size() );
	os.write( (const char*)&crc, 4 );
}
static bool readHDR( std::ifstream & is, int & w, int & h ) {
	std::string type;
	std::vector<char> data = readChunk( is, type );
	if( type == "IHDR" ) {
		uint32_t buf[5];
		memcpy( buf, data.data(), 2*sizeof(uint32_t) );
		w = ntohl( buf[0] );
		h = ntohl( buf[1] );
		return true;
	}
	return false;
}
static void writeHDR( std::ofstream & os, int w, int h ) {
	struct HDR {
		uint32_t w, h;
		uint8_t d, t, c, f, i;
	};
	int S = 2*sizeof(uint32_t) + 5*sizeof(uint8_t);
	HDR hdr = { ntohl( w ), ntohl( h ), 8, 3, 0, 0, 0 };
	std::vector<char> data( S );
	memcpy( data.data(), &hdr, S );
	writeChunk( os, "IHDR", data );
}

#define CHECK_ERR(err, msg) { \
		if (err != Z_OK) { \
			fprintf(stderr, "%s error: %d\n", msg, err); \
			exit(1); \
		} \
	}
static std::vector<char> readDat( std::ifstream & is ) {
	std::string type;
	std::vector<char> data = readChunk( is, type );
	if( !data.size() ) {
		is.close();
		return std::vector<char>();
	}
	std::vector<char> r;
	if (type == "IDAT" || type == "fdAT") {
		int o = 4*(type == "fdAT");
		Bytef buf[8192];
		
		z_stream d_stream; /* compression stream */
		int err;
		
		d_stream.zalloc = (alloc_func)0;
		d_stream.zfree = (free_func)0;
		d_stream.opaque = (voidpf)0;
		
		d_stream.next_in  = (Bytef*)(data.data()+o);
		d_stream.avail_in = data.size()-o;
		
		err = inflateInit(&d_stream);
		CHECK_ERR(err, "inflateInit");
		
		for (;;) {
			d_stream.next_out = buf;            /* discard the output */
			d_stream.avail_out = (uInt)sizeof(buf);
			err = inflate(&d_stream, Z_NO_FLUSH);
			r.insert( r.end(), buf, buf+(sizeof(buf)-d_stream.avail_out) );
			if (err == Z_STREAM_END) break;
			if (d_stream.avail_in == 0) {
				std::string new_type;
				data = readChunk( is, new_type );
				if( type != new_type )
					throw std::invalid_argument("Chunk type missmatch!\n");
				d_stream.next_in  = (Bytef*)(data.data()+o);
				d_stream.avail_in = data.size()-o;
			}
			CHECK_ERR(err, "inflate");
		}
		err = inflateEnd(&d_stream);
		CHECK_ERR(err, "inflateEnd");
	}
	return r;
}
static std::vector<uint16_t> readRemap( std::ifstream & is ) {
	std::string type = "tEXt";
	std::vector<char> t = readChunk<true>( is, type );
	std::vector<uint16_t> r;
	if( t.size()>3 && t[0]=='i' && t[1]=='d' && t[2]=='s' && t[3]=='\0' ) {
		r.resize( (t.size()-4)/2 );
		memcpy( r.data(), t.data()+4, t.size()-4);
		std::transform( r.begin(), r.end(), r.begin(), htons2 );
	}
	return r;
}
static void writePLTE( std::ofstream & os, bool binary=false ) {
	uint32_t colors[] = {0xeeeeec, 0xfce94f, 0x8ae234, 0x888a85, 0xfcaf3e, 0x729fcf, 0xe9b96e, 0xef2929, 0xad7fa8, 0xd3d7cf, 0xedd400, 0x73d216, 0x555753, 0xf57900, 0x3465a4, 0xc17d11, 0xcc0000, 0x75507b, 0xbabdb6, 0xc4a000, 0x4e9a06, 0x2e3436, 0xce5c00, 0x204a87, 0x8f5902, 0xa40000, 0x5c3566} ;
	if(binary) colors[1] = 0;
	std::vector<char> data( 3*sizeof(colors)/sizeof(*colors) );
	for( int i=0; i<sizeof(colors)/sizeof(*colors); i++ ) {
		memcpy( data.data()+3*i, colors+i, 3 );
		std::swap( data[3*i], data[3*i+2] );
	}
	writeChunk( os, "PLTE", data );
}
static void writeDat( std::ofstream & os, const std::vector<char> & data ) {
	std::vector<char> cdata( data.size() );
	uLongf len = data.size();
	compress2( (Bytef*)cdata.data(), &len, (const Bytef*)data.data(), data.size(), 9 );
	cdata.resize( len );
	writeChunk( os, "IDAT", cdata );
}
static void writeFDat( std::ofstream & os, const std::vector<char> & data, int i ) {
	std::vector<char> cdata( data.size()+4 );
	uint32_t ui = ntohl( i );
	memcpy( cdata.data(), &ui, sizeof(ui) );
	uLongf len = data.size();
	compress2( (Bytef*)(cdata.data()+4), &len, (const Bytef*)data.data(), data.size(), 9 );
	cdata.resize( len+4 );
	writeChunk( os, "fdAT", cdata );
}
static void writeACTL( std::ofstream & os, int N ) {
	std::vector<char> data( 8, 0 );
	uint32_t uN = ntohl( N );
	memcpy( data.data(), &uN, sizeof(uN) );
	writeChunk( os, "acTL", data );
}
static void writeFCTL( std::ofstream & os, int i, int w, int h ) {
	struct HDR {
		uint32_t seq_no, w, h, x_off, y_off;
		uint16_t delay_num, delay_den, tmp;
	};
	int S = 5*sizeof(uint32_t) + 3*sizeof(uint16_t);
	HDR hdr = { ntohl( i ), ntohl( w ), ntohl( h ), 0, 0, ntohs(1), ntohs(1), 0 };
	std::vector<char> data( S );
	memcpy( data.data(), &hdr, S );
	writeChunk( os, "fcTL", data );
}
np::ndarray readAPNG( const std::string & filename ) {
	std::ifstream is( filename.c_str(), std::fstream::in | std::fstream::binary );
	if( !is.is_open() )
		return np::zeros( make_tuple(), np::dtype::get_builtin<unsigned char>() );
	char magic[9]={0};
	is.read( magic, 8 );
	if ((std::string)magic != "\x89\x50\x4E\x47\x0D\x0A\x1A\x0A")
		printf("PNG bad magic!\n");
	
	int W=0, H=0;
	while(!readHDR( is, W, H ) && !is.eof());
	
	std::vector< char > data;
	int N=0;
	while(!is.eof() && is.is_open()){
		std::vector<char> r = readDat( is );
		if(r.size()) {
			size_t o = data.size();
			data.resize( o + W*H );
			for( int i=0; i<H; i++ )
				std::copy( r.begin()+i*(W+1)+1, r.begin()+i*(W+1)+W+1, data.begin()+o+i*W );
			N++;
		}
	}

	np::ndarray r = np::zeros( make_tuple(N,H,W), np::dtype::get_builtin<unsigned char>() );
	std::copy( data.begin(), data.end(), r.get_data() );
	return r;
}
void writeAPNG( const std::string & filename, const np::ndarray & array ) {
	int N=1,W,H;
	if( array.get_nd() < 2 )
		throw std::invalid_argument( "Array needs at least 2 dimensions" );
	if( array.get_nd() > 3 )
		throw std::invalid_argument( "Array has too many dimensions (>3)" );
	if( array.get_nd() == 2) {
		W = array.shape(1);
		H = array.shape(0);
	}
	else {
		W = array.shape(2);
		H = array.shape(1);
		N = array.shape(0);
	}
	std::ofstream os( filename.c_str(), std::fstream::out | std::fstream::binary );
	char magic[9]="\x89\x50\x4E\x47\x0D\x0A\x1A\x0A";
	os.write( magic, 8 );
	writeHDR( os, W, H );
	writePLTE( os, true );
	writeACTL( os, N );
	int seq_n = 0;
	for( int i=0; i< N; i++ ) {
		writeFCTL( os, seq_n++, W, H );
		std::vector<char> data( H*(W+1), 0 );
		int o = i*W*H;
		for( int j=0; j<H; j++ )
			std::copy( array.get_data()+o+j*W, array.get_data()+o+(j+1)*W, data.begin()+j*(W+1)+1 );
		if(!i)
			writeDat( os, data );
		else
			writeFDat( os, data, seq_n++ );
	}
	writeChunk( os, "IEND", std::vector<char>() );
}
template<typename T>
np::ndarray readIPNG_T( const std::string & filename ) {
	std::ifstream is( filename.c_str(), std::fstream::in | std::fstream::binary );
	if( !is.is_open() )
		return np::zeros( make_tuple(), np::dtype::get_builtin<T>() );
	char magic[9]={0};
	is.read( magic, 8 );
	if ((std::string)magic != "\x89\x50\x4E\x47\x0D\x0A\x1A\x0A")
		printf("PNG bad magic!\n");
	
	int W=0, H=0;
	while(!readHDR( is, W, H ) && !is.eof());
	
	std::vector< char > data;
	std::vector< uint16_t > lbl_map;
	while(!is.eof() && is.is_open()){
		if( lbl_map.size() == 0 )
			lbl_map = readRemap( is );
		std::vector<char> r = readDat( is );
		if(r.size()) {
			size_t o = data.size();
			data.resize( o + W*H );
			for( int i=0; i<H; i++ )
				std::copy( r.begin()+i*(W+1)+1, r.begin()+i*(W+1)+W+1, data.begin()+o+i*W );
			break;
		}
	}
	np::ndarray r = np::zeros( make_tuple(H,W), np::dtype::get_builtin<T>() );
	T * pr = (T*)r.get_data();
	if( lbl_map.size() > 0 ) {
		for( int i=0; i<data.size(); i++ )
			if( data[i] < lbl_map.size() )
				pr[i] = lbl_map[ data[i] ];
	}
	else
		std::copy( data.begin(), data.end(), r.get_data() );
	return r;
}
np::ndarray readIPNG( const std::string & filename, int bit ) {
	if( bit == 8 )
		return readIPNG_T<uint8_t>( filename );
	else if (bit == 16)
		return readIPNG_T<int16_t>( filename );
	return np::zeros( make_tuple(), np::dtype::get_builtin<uint8_t>() );
}
RMatrixXus readPNG16(const std::string &filename) {
	std::ifstream is( filename.c_str(), std::fstream::in | std::fstream::binary );
	if( !is.is_open() )
		return RMatrixXus();
	char magic[9]={0};
	is.read( magic, 8 );
	if ((std::string)magic != "\x89\x50\x4E\x47\x0D\x0A\x1A\x0A")
		printf("PNG bad magic!\n");
	
	int W=0, H=0;
	while(!readHDR( is, W, H ) && !is.eof());
	
	std::vector< short > data;
	while(!is.eof() && is.is_open()){
		std::vector<char> r = readDat( is );
		if(r.size()) {
			size_t o = data.size();
			data.resize( o + W*H );
			for( int i=0; i<H; i++ ) {
				uint16_t * pr = (uint16_t*)(r.data()+i*(W*sizeof(uint16_t)+1)+1);
				std::transform( pr, pr+W, data.begin()+o+i*W, htons2 );
			}
			break;
		}
	}
	if( data.size() >= H*W ) {
		RMatrixXus r( H, W );
		memcpy( r.data(), data.data(), W*H*sizeof(uint16_t) );
		return r;
	}
	return RMatrixXus();
}
