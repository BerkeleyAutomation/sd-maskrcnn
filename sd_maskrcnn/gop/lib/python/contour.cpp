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
#include "contour.h"
#include "util.h"
#include "gop.h"
#include "contour/directedsobel.h"
#include "contour/sketchtokens.h"
#include "contour/structuredforest.h"

static void StructuredForest_fitAndAddTree1( StructuredForest & that, const Features & f, const RMatrixXf & lbl, const RMatrixXb & patch_data, const VectorXi & fid ) {
	that.fitAndAddTree( f, lbl, patch_data, fid );
}
static void StructuredForest_fitAndAddTree2( StructuredForest & that, const Features & f, const RMatrixXf & lbl, const RMatrixXb & patch_data, const VectorXi & fid, TreeSettings settings) {
	that.fitAndAddTree( f, lbl, patch_data, fid, settings );
}
static void StructuredForest_fitAndAddTree3( StructuredForest & that, const Features & f, const RMatrixXf & lbl, const RMatrixXb & patch_data, const VectorXi & fid, TreeSettings settings, bool mt ) {
	that.fitAndAddTree( f, lbl, patch_data, fid, settings, mt );
}
static list BoundaryDetector_detectAndFilterAll( const BoundaryDetector & that, const list & ims ) {
	std::vector<float*> im_data(len(ims));
	std::vector<int> W(len(ims)),H(len(ims));
	std::vector<Image8u*> img(len(ims));
	for( int i=0; i<len(ims); i++ ) {
		img[i] = extract<Image8u*>( ims[i] );
		if( img[i]->C() != 3 )
			throw std::invalid_argument("RGB image expected!");
	}
	std::vector<RMatrixXf> d(len(ims));
#pragma omp parallel for
	for( int i=0; i<len(ims); i++ )
		d[i] = that.detectAndFilter( *img[i] );
	list r;
	for( int i=0; i<len(ims); i++ )
		r.append( d[i] );
	return r;
}
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS( SketchTokens_filter_overload, SketchTokens::filter, 1, 3 )

void defineContour() {
	ADD_MODULE(contour);
	
	class_< STFeatures, std::shared_ptr<STFeatures>, bases<Features> >("STFeatures", init<const Image8u &, int>() );
	implicitly_convertible< std::shared_ptr<SFFeatures>, std::shared_ptr<Features> >();
	
	class_< SFFeatures, std::shared_ptr<SFFeatures>, bases<Features> >("SFFeatures", init<const Image8u &>() )
	.def( init<const Image8u &, const StructuredForestSettings &>() )
	.add_property("x", make_function( &SFFeatures::x, return_value_policy<return_by_value>() ) )
	.add_property("y", make_function( &SFFeatures::y, return_value_policy<return_by_value>() ) )
	.add_property("patchFeatures", make_function( &SFFeatures::patchFeatures, return_value_policy<return_by_value>() ) )
	.add_property("ssimFeatures", make_function( &SFFeatures::ssimFeatures, return_value_policy<return_by_value>() ) );
	implicitly_convertible< std::shared_ptr<STFeatures>, std::shared_ptr<Features> >();
	
	class_< BoundaryDetector, boost::noncopyable>("BoundaryDetector",no_init)
	.def("detect", &BoundaryDetector::detect)
	.def("filter", &BoundaryDetector::filter)
	.def("detectAndFilter", &BoundaryDetector::detectAndFilter)
	.def("detectAndFilterAll", &BoundaryDetector_detectAndFilterAll);
	
	class_< DirectedSobel, bases<BoundaryDetector> >("DirectedSobel", init<>() )
	.def(init<bool>());
	
	class_< SketchTokens, bases<BoundaryDetector> >("SketchTokens", init<>() )
	.def( init<int>() )
	.def( init<int,int>() )
	.def( init<int,int,int>() )
	.def( "filter", (RMatrixXf(SketchTokens::*)(const RMatrixXf &, int, int) const)&SketchTokens::filter, SketchTokens_filter_overload( args("that","detection","suppress","nms"), "Filter a detection" ) )
	.def( "load", &SketchTokens::load );
	
	class_<StructuredForestSettings>("StructuredForestSettings",init<>())
	.def_readwrite("stride",&StructuredForestSettings::stride)
	.def_readwrite("shrink",&StructuredForestSettings::shrink)
	.def_readwrite("out_patch_size",&StructuredForestSettings::out_patch_size)
	.def_readwrite("feature_patch_size",&StructuredForestSettings::feature_patch_size)
	.def_readwrite("patch_smooth",&StructuredForestSettings::patch_smooth)
	.def_readwrite("sim_smooth",&StructuredForestSettings::sim_smooth)
	.def_readwrite("sim_cells",&StructuredForestSettings::sim_cells);
	
	class_< StructuredForest, bases<BoundaryDetector> >("StructuredForest", init<>() )
	.def( init<int>() )
	.def( init<int,int>() )
	.def( init<int,int,StructuredForestSettings>() )
	.def( init<StructuredForestSettings>() )
	.def( "compress", &StructuredForest::compress )
	.def( "duplicateAndAddTree", &StructuredForest::duplicateAndAddTree )
	.def( "fitAndAddTree", &StructuredForest_fitAndAddTree1 )
	.def( "fitAndAddTree", &StructuredForest_fitAndAddTree2 )
	.def( "fitAndAddTree", &StructuredForest_fitAndAddTree3 )
	.def( "predictLastTree", &StructuredForest::predictLastTree )
	.def( "load", &StructuredForest::load )
	.def( "save", &StructuredForest::save )
	.def( "setFromMatlab", &StructuredForest::setFromMatlab );
	
	class_< MultiScaleStructuredForest, bases<StructuredForest> >("MultiScaleStructuredForest", init<>() );
}
