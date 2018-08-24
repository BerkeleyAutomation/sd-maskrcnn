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
#include "imgproc/color.h"
#include "imgproc/filter.h"
#include "imgproc/gradient.h"
#include "imgproc/morph.h"
#include "imgproc/nms.h"
#include "imgproc/resample.h"
#include "util/util.h"
#include "imgproc.h"
#include "gop.h"
#include "util.h"
#include <boost/python/def_visitor.hpp>


/****************  color.cpp  ****************/
typedef void (*ConvFunction)(Image &, const Image &);
static ConvFunction cfun[] = {rgb2luv,srgb2luv,rgb2lab,srgb2lab,rgb2hsv};
template<int T>
static Image convert( const Image & image ) {
	if( image.C() != 3 )
		throw std::invalid_argument( "3-channel image required" );
	Image r( image.W(), image.H(), 3 );
	cfun[T]( r, image );
	return r;
}

/****************  filter.cpp  ****************/
static RMatrixXf percentileFilter_m( const RMatrixXf & image, int rad, float p ) {
	RMatrixXf r( image.rows(), image.cols() );
	percentileFilter( r.data(), image.data(), image.cols(), image.rows(), 1, rad, p );
	return r;
}
#define defFilter( name )\
static RMatrixXf name##_m( const RMatrixXf & image, int rad ) {\
	RMatrixXf r( image.rows(), image.cols() );\
	name( r.data(), image.data(), image.cols(), image.rows(), 1, rad );\
	return r;\
}
defFilter( boxFilter )
defFilter( tentFilter )
defFilter( gaussianFilter )
static RMatrixXf exactGaussianFilter_m( const RMatrixXf & image, int rad, int R ) {
	RMatrixXf r( image.rows(), image.cols() );
	exactGaussianFilter( r.data(), image.data(), image.cols(), image.rows(), 1, rad, R );
	return r;
}

/****************  gradient.cpp  ****************/
static tuple gradient2( const Image & image ) {
	Image gx, gy;
	gradient( gx, gy, image );
	return make_tuple(gx, gy);
}
static tuple gradientMagAndOri2( const Image & image, int norm_rad, float norm_const ) {
	RMatrixXf gm, go;
	gradientMagAndOri( gm, go, image, norm_rad, norm_const );
	return make_tuple(gm, go);
}
static Image gradientHist2( const RMatrixXf & gm, const RMatrixXf & go, int nori, int nbins=1) {
	Image r;
	gradientHist( r, gm, go, nori, nbins );
	return r;
}

/****************  morph.cpp  ****************/
static RMatrixXb thin( const RMatrixXb & image ) {
	RMatrixXb r = image;
	thinningGuoHall( r );
	return r;
}

/****************  nms.cpp  ****************/
static RMatrixXf suppressBnd2( const RMatrixXf &im, int b ) {
	RMatrixXf r = im;
	suppressBnd( r, b );
	return r;
}
/****************  other  ****************/
static np::ndarray extractPatch( const np::ndarray &a, const RMatrixXi & l, const VectorXi & s ) {
	// Check the input
	if( !(a.get_flags() & np::ndarray::C_CONTIGUOUS) )
		throw std::invalid_argument( "Contiguous array required!" );
	if( l.cols() != a.get_nd() )
		throw std::invalid_argument("location dimension does not match matrix");
	if( l.cols() != s.size() )
		throw std::invalid_argument("patch size and locations have different dimension");
	for( int i=0; i<l.rows(); i++ )
		for( int j=0; j<s.size(); j++ )
			if( l(i,j)<0 || l(i,j)+s[j] > a.shape(j) )
				throw std::invalid_argument("location out of range! Dimension "+std::to_string(j)+" : "+std::to_string(l(i,j))+" + "+std::to_string(s[j])+"  ( 0 .. "+std::to_string(a.shape(j))+" )");
	
	const int itemsize = a.get_dtype().get_itemsize();
	
	// Create the output array
	std::vector<Py_intptr_t> out_size;
	out_size.push_back( l.rows() );
	for( int i=0; i<s.size(); i++ )
		if( s[i]>1 )
			out_size.push_back( s[i] );
	np::ndarray res = np::empty( out_size.size(), out_size.data(), a.get_dtype() );
	
	// Prepare the patches
	const int ls = s[s.size()-1]; // Radius of the last dimension
	std::vector<int> patch_offsets(1,0);
	for( int d=0; d<s.size()-1; d++ ) {
		std::vector<int> opo = patch_offsets;
		patch_offsets.clear();
		for( int i=0; i<s[d]; i++ )
			for( int o: opo )
				patch_offsets.push_back( a.shape(d+1)*(o+i) );
	}
	const char * pa = a.get_data();
	char * pres = res.get_data();
	for( int i=0,k=0; i<l.rows(); i++ ) {
		int o=0;
		for( int d=0; d<s.size(); d++ )
			o = a.shape(d)*o + l(i,d);
		// Extract the patches
		for( int po: patch_offsets ) {
			memcpy( pres+k*itemsize, pa+(po+o)*itemsize, ls*itemsize );
			k += ls;
		}
	}
	return res;
}
static np::ndarray extractBoxes( const Image &im, const RMatrixXi & b, int W, int H ) {
	// Create the output array
	const int C = im.C();
	const int im_W = im.W();
	
	np::ndarray res = np::empty( make_tuple( b.rows(), H, W, C ), np::dtype::get_builtin<float>() );
	const float * pim = (const float *)im.data();
	float * pres = (float *)res.get_data();
#pragma omp parallel for
	for( int i=0; i<b.rows(); i++ ) {
		// Build the lerp lookup table
		const float Y0=b(i,0), X0=b(i,1);
		const float dy=b(i,2)-Y0-1, dx=b(i,3)-X0-1;
		std::vector< int > x0( W ), y0( H ), x1( W ), y1( H );
		std::vector< float > wx( W ), wy( H );
		for( int j=0; j<H; j++ ) {
			float p = Y0 + j*dy/(H-1);
			y0[j] = floor(p); y1[j] = ceil(p);
			wy[j] = y1[j]-p;
		}
		for( int j=0; j<W; j++ ) {
			float p = X0 + j*dx/(W-1);
			x0[j] = floor(p); x1[j] = ceil(p);
			wx[j] = x1[j]-p;
		}
		
		// Run the lerp
		float * pr = pres + i*H*W*C;
		for( int j=0; j<H; j++ )
			for( int i=0; i<W; i++ ) 
				for( int k=0; k<C; k++ )
					pr[ (j*W+i)*C+k ] =    wx[i] *   wy[j] *pim[ ( y0[j]*im_W+x0[i] )*C+k ]
					                   +   wx[i] *(1-wy[j])*pim[ ( y1[j]*im_W+x0[i] )*C+k ]
					                   +(1-wx[i])*   wy[j] *pim[ ( y0[j]*im_W+x1[i] )*C+k ]
					                   +(1-wx[i])*(1-wy[j])*pim[ ( y1[j]*im_W+x1[i] )*C+k ];
	}
	return res;
}
template<typename T> struct ImageTypeStr {
	static const std::string s;
};
template<> const std::string ImageTypeStr<uint8_t>::s = "u1";
template<> const std::string ImageTypeStr<float>::s = "f4";

template<typename IM>
struct Image_indexing_suite: def_visitor<Image_indexing_suite<IM> > {
	typedef typename IM::value_type value_type;
	
	struct Image_pickle_suite : pickle_suite
	{
		static tuple getinitargs(const IM& im) {
			return make_tuple( im.W(), im.H(), im.C() );
		}
		
		static object getstate(const IM& im) {
			const int N = im.W()*im.H()*im.C()*sizeof(value_type);
			return object( handle<>( PyBytes_FromStringAndSize( (const char*)im.data(), N ) ) );
		}
		
		static void setstate(IM& im, const object & state) {
			if(!PyBytes_Check(state.ptr()))
				throw std::invalid_argument("Failed to unpickle, unexpected type!");
			const int N = im.W()*im.H()*im.C()*sizeof(value_type);
			if( PyBytes_Size(state.ptr()) != N )
				throw std::invalid_argument("Failed to unpickle, unexpected size!");
			memcpy( im.data(), PyBytes_AS_STRING(state.ptr()), N );
		}
	};
	
	template <class classT>	void visit(classT& c) const {
		c
		.def("__init__",make_constructor(&Image_indexing_suite::init1))
		.def("__init__",make_constructor(&Image_indexing_suite::init2))
		.def("__init__",make_constructor(&Image_indexing_suite::init3))
		.add_property("W",&IM::W)
		.add_property("H",&IM::H)
		.add_property("C",&IM::C)
		.def("tileC",&IM::tileC)
		.def_pickle(Image_pickle_suite())
		.add_property("__array_interface__", &Image_indexing_suite::array_interface);
	}
	static IM * init1() {
		return new IM();
	}
	static IM * init2( int W, int H, int C ) {
		return new IM( W, H, C );
	}
	static IM * init3( const np::ndarray & d ) {
		checkArray( d, value_type, 2, 3, true );
		
		IM* r = new IM(d.shape(1),d.shape(0),d.get_nd()>2?d.shape(2):1);
		memcpy( r->data(), d.get_data(), r->W()*r->H()*r->C()*sizeof(value_type) );
		return r;
	}
	static dict array_interface( IM & im ) {
		dict r;
		r["shape"] = make_tuple( im.H(), im.W(), im.C() );
		r["typestr"] = ImageTypeStr<value_type>::s;
		r["data"] = make_tuple((size_t)im.data(),1);
		r["version"] = 3;
		return r;
	}
};
template<typename I1, typename I2>
I1 convertImage( const I2 & im ) {
	return im;
}
void defineImgProc() {
	ADD_MODULE(imgproc);
	// Color
	def("rgb2luv",convert<0>);
	def("srgb2luv",convert<1>);
	def("rgb2lab",convert<2>);
	def("srgb2lab",convert<3>);
	def("rgb2hsv",convert<4>);
	// Filters
	def("boxFilter",boxFilter_m);
	def("tentFilter",tentFilter_m);
	def("gaussianFilter",gaussianFilter_m);
	def("exactGaussianFilter",exactGaussianFilter_m);
	def("percentileFilter",percentileFilter_m);
	// Gradient
	def("gradient",gradient2);
	def("gradientMag",gradientMag);
	def("gradientMagAndOri",gradientMagAndOri2);
	def("gradientHist",gradientHist2);
	// Morphology
	def("thin",thin);
	// NMS
	def("nms",nms);
	def("suppressBnd",suppressBnd2);
	// Upsampling
	def("upsample",upsample);
	def("upsampleLinear",upsampleLinear);
	def("downsample", (RMatrixXf(*)( const RMatrixXf &, int, int ))downsample);
	def("downsample", (Image(*)( const Image &, int, int ))downsample);
	def("padIm",padIm);
	// Other
	def("extractPatch",extractPatch);
	def("extractBoxes",extractBoxes);
	
	def("imread",imread);
	def("imwrite",imwrite);
	
	// Image and Image8u
	class_<Image,std::shared_ptr<Image> >("Image")
	.def(Image_indexing_suite<Image>())
	.def("toImage8u",convertImage<Image,Image8u>);
	class_<Image8u,std::shared_ptr<Image8u> >("Image8u")
	.def(Image_indexing_suite<Image8u>())
	.def("toImage",convertImage<Image,Image8u>);
}
