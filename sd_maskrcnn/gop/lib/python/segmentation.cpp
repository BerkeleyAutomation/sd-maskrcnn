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
#include "segmentation/segmentation.h"
#include "segmentation/iouset.h"
#include "contour/sketchtokens.h"
#include "contour/structuredforest.h"
#include "contour/directedsobel.h"
#include "map.h"
#include "gop.h"
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

static np::ndarray segmentBinaryGraph( const np::ndarray & DX, const np::ndarray & DY ) {
	// Used to evaluate CPMC
	Matrix<bool,Dynamic,Dynamic,RowMajor> dx = mapMatrixX<bool>(DX), dy = mapMatrixX<bool>(DY);
	Matrix<short,Dynamic,Dynamic,RowMajor> r = -Matrix<short,Dynamic,Dynamic,RowMajor>::Ones(dx.rows(),dy.cols());
	
	int id=0;
	std::vector<Vector2i> q;
	for( int j=0; j<r.rows(); j++ )
		for( int i=0; i<r.cols(); i++ ) 
			if( r(j,i) == -1 ){
				r(j,i)=id;
				q.clear();
				q.push_back( Vector2i(i,j) );
				while(!q.empty()) {
					int x = q.back()[0], y=q.back()[1];
					q.pop_back();
					if( x && dx(y,x-1) && r(y,x-1)==-1 ) {
						r(y,x-1) = id;
						q.push_back( Vector2i(x-1,y) );
					}
					if( x+1<r.cols() && dx(y,x) && r(y,x+1)==-1 ) {
						r(y,x+1) = id;
						q.push_back( Vector2i(x+1,y) );
					}
					if( y && dy(y-1,x) && r(y-1,x)==-1 ) {
						r(y-1,x) = id;
						q.push_back( Vector2i(x,y-1) );
					}
					if( y+1<r.rows() && dy(y,x) && r(y+1,x)==-1 ) {
						r(y+1,x) = id;
						q.push_back( Vector2i(x,y+1) );
					}
				}
				id++;
				if( id > (1<<15) )
					throw std::domain_error( "segmentBinaryGraph: Overflow (short)" );
			}
	return toNumpy(r);
}
template<typename BDetector>
std::vector< std::shared_ptr<ImageOverSegmentation> > generateGeodesicKMeans( const BDetector & det, const list & ims, int approx_N ) {
	const int N = len(ims);
	std::vector<Image8u*> img(N);
	for( int i=0; i<N; i++ )
		img[i] = extract<Image8u*>( ims[i] );
	std::vector< std::shared_ptr<ImageOverSegmentation> > ios( N );
#pragma omp parallel for
	for( int i=0; i<N; i++ )
		ios[i] = geodesicKMeans( *img[i], det, approx_N, 2 );
	return ios;
}
np::ndarray boundaryDistance( const np::ndarray & sg ) {
	checkArray( sg, short, 2, 2, true );
	const int W = sg.shape(1), H = sg.shape(0);
	np::ndarray r = np::zeros( make_tuple(H,W), np::dtype::get_builtin<float>() );
	const short * psg = (const short*)sg.get_data();
	float * pr = (float*) r.get_data();
	std::fill( pr, pr+W*H, 1e10 );
	for( int it=0; it<10; it++ ) {
		for( int j=0,k=0; j<H; j++ )
			for( int i=0; i<W; i++,k++ ) {
				if( i ) {
					if( psg[k] != psg[k-1] )
						pr[k] = 0;
					else
						pr[k] = std::min( pr[k], pr[k-1]+1 );
				}
				if( j ) {
					if( psg[k] != psg[k-W] )
						pr[k] = 0;
					else
						pr[k] = std::min( pr[k], pr[k-W]+1 );
				}
			}
		
		for( int j=H-1,k=W*H-1; j>=0; j-- )
			for( int i=W-1; i>=0; i--,k-- ) {
				if( i ) {
					if( psg[k] != psg[k-1] )
						pr[k-1] = 0;
					else
						pr[k-1] = std::min( pr[k]+1, pr[k-1] );
				}
				if( j ) {
					if( psg[k] != psg[k-W] )
						pr[k-W] = 0;
					else
						pr[k-W] = std::min( pr[k]+1, pr[k-W] );
				}
			}
	}
	return r;
}
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS( ImageOverSegmentation_boundaryMap_overload, ImageOverSegmentation::boundaryMap, 0, 1 )
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS( ImageOverSegmentation_projectSegmentation_overload, ImageOverSegmentation::projectSegmentation, 1, 2 )
void defineSegmentation() {
	ADD_MODULE(segmentation);
	
	// Helpers
	def("segmentBinaryGraph",segmentBinaryGraph);
	
	class_<IOUSet>("IOUSet", init<OverSegmentation>() )
	.def( "maxIOU", &IOUSet::maxIOU )
	.def( "intersects", static_cast<bool (IOUSet::*)(const VectorXb &, float) const>(&IOUSet::intersects) )
	.def( "intersects", static_cast<bool (IOUSet::*)(const VectorXb &, const VectorXf &) const>(&IOUSet::intersects) )
	.def( "add", &IOUSet::add );
	
	/***** Over Segmentation *****/
	class_<OverSegmentation,std::shared_ptr<OverSegmentation> >( "OverSegmentation", init<const Edges &, const VectorXf &>() )
	.def(init<const Edges &>())
	.add_property("Ns",&OverSegmentation::Ns)
	.add_property("edges",make_function(&OverSegmentation::edges,return_value_policy<return_by_value>()))
	.add_property("edge_weights",make_function(&OverSegmentation::edgeWeights,return_value_policy<return_by_value>()),&OverSegmentation::setEdgeWeights)
	.def_pickle( SaveLoad_pickle_suite_shared_ptr<OverSegmentation>() );
	
	class_< ImageOverSegmentation,std::shared_ptr<ImageOverSegmentation>,bases<OverSegmentation> >( "ImageOverSegmentation", init<>() )
	.def("boundaryMap",&ImageOverSegmentation::boundaryMap, ImageOverSegmentation_boundaryMap_overload())
	.def("projectSegmentation",&ImageOverSegmentation::projectSegmentation, ImageOverSegmentation_projectSegmentation_overload())
	.def("project",static_cast<VectorXf (ImageOverSegmentation::*)(const RMatrixXf&,const std::string &)const>( &ImageOverSegmentation::project ))
	.def("project",static_cast<RMatrixXf (ImageOverSegmentation::*)(const Image&,const std::string &)const>( &ImageOverSegmentation::project ))
	.def("projectBoundary",static_cast<VectorXf (ImageOverSegmentation::*)(const RMatrixXf&,const std::string &)const>( &ImageOverSegmentation::projectBoundary ))
	.def("projectBoundary",static_cast<VectorXf (ImageOverSegmentation::*)(const RMatrixXf&,const RMatrixXf&,const std::string &)const>( &ImageOverSegmentation::projectBoundary ))
	.def("maskToBox",&ImageOverSegmentation::maskToBox)
	.add_property("s",make_function(&ImageOverSegmentation::s,return_value_policy<return_by_value>()))
	.add_property("image",make_function(&ImageOverSegmentation::image,return_value_policy<return_by_value>()))
	.def_pickle( SaveLoad_pickle_suite_shared_ptr<ImageOverSegmentation>() );
	implicitly_convertible< std::shared_ptr<ImageOverSegmentation>, std::shared_ptr<OverSegmentation> >();
	
	class_< std::vector< std::shared_ptr<ImageOverSegmentation> > >("ImageOverSegmentationVec")
	.def( vector_indexing_suite< std::vector< std::shared_ptr<ImageOverSegmentation> >, true >() )
	.def_pickle( VectorSaveLoad_pickle_suite_shared_ptr<ImageOverSegmentation>() );
	
	def("geodesicKMeans",static_cast<std::shared_ptr<ImageOverSegmentation>(*)(const Image8u &, const BoundaryDetector &, int, int)>(geodesicKMeans));
	def("geodesicKMeans",static_cast<std::shared_ptr<ImageOverSegmentation>(*)(const Image8u &, const BoundaryDetector &, int)>(geodesicKMeans));
	def("geodesicKMeans",static_cast<std::shared_ptr<ImageOverSegmentation>(*)(const Image8u &, const SketchTokens &, int, int)>(geodesicKMeans));
	def("geodesicKMeans",static_cast<std::shared_ptr<ImageOverSegmentation>(*)(const Image8u &, const SketchTokens &, int)>(geodesicKMeans));
	def("geodesicKMeans",static_cast<std::shared_ptr<ImageOverSegmentation>(*)(const Image8u &, const StructuredForest &, int, int)>(geodesicKMeans));
	def("geodesicKMeans",static_cast<std::shared_ptr<ImageOverSegmentation>(*)(const Image8u &, const StructuredForest &, int)>(geodesicKMeans));
	def("geodesicKMeans",static_cast<std::shared_ptr<ImageOverSegmentation>(*)(const Image8u &, const DirectedSobel &, int, int)>(geodesicKMeans));
	def("geodesicKMeans",static_cast<std::shared_ptr<ImageOverSegmentation>(*)(const Image8u &, const DirectedSobel &, int)>(geodesicKMeans));
	
	def( "generateGeodesicKMeans", &generateGeodesicKMeans<BoundaryDetector> );
	def( "generateGeodesicKMeans", &generateGeodesicKMeans<SketchTokens> );
	def( "generateGeodesicKMeans", &generateGeodesicKMeans<StructuredForest> );
	def( "generateGeodesicKMeans", &generateGeodesicKMeans<DirectedSobel> );
	
	def( "boundaryDistance", &boundaryDistance );
}
