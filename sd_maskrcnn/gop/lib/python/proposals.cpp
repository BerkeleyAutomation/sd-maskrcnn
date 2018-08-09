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
#include "proposals/seed.h"
#include "proposals/proposal.h"
#include "proposals/geodesics.h"
#include "proposals/unary.h"
#include "proposals/saliency.h"
#include "gop.h"
#include "util.h"
#include "map.h"
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

static void LearnedSeed_train1( LearnedSeed & that, const std::vector< std::shared_ptr<ImageOverSegmentation> > & ios, const list & lbl, int max_seed ) {
	// Load the data
	std::vector<VectorXs> v_lbl;
	if( ios.size() != len(lbl) )
		throw std::invalid_argument("gops and lbls size does not match!");
	for( int i=0; i<len(lbl); i++ )
		v_lbl.push_back( mapVectorX<short>(extract<np::ndarray>(lbl[i])) );
	that.train( ios, v_lbl, max_seed );
}
static void LearnedSeed_train2( LearnedSeed & that, const std::vector< std::shared_ptr<ImageOverSegmentation> > & ios, const list & lbl, int max_seed, int n_seed_per_obj ) {
	// Load the data
	std::vector<VectorXs> v_lbl;
	if( ios.size() != len(lbl) )
		throw std::invalid_argument("gops and lbls size does not match!");
	for( int i=0; i<len(lbl); i++ )
		v_lbl.push_back( mapVectorX<short>(extract<np::ndarray>(lbl[i])) );
	that.train( ios, v_lbl, max_seed, n_seed_per_obj );
}
static np::ndarray Unary_compute( const Unary & that, int seed ) {
	return toNumpy( that.compute( seed ) );
}
static ProposalSettings::UnarySettings* make_UnarySettings1( int n_seed ) {
	return new ProposalSettings::UnarySettings( n_seed );
}
static ProposalSettings::UnarySettings* make_UnarySettings2( int n_seed, int n_lvl_set ) {
	return new ProposalSettings::UnarySettings( n_seed, n_lvl_set );
}
static ProposalSettings::UnarySettings* make_UnarySettings3( int n_seed, int n_lvl_set, std::shared_ptr<UnaryFactory> fg_unary ) {
	return new ProposalSettings::UnarySettings( n_seed, n_lvl_set, fg_unary );
}
static ProposalSettings::UnarySettings* make_UnarySettings4( int n_seed, int n_lvl_set, std::shared_ptr<UnaryFactory> fg_unary, list background_seeds ) {
	return new ProposalSettings::UnarySettings( n_seed, n_lvl_set, fg_unary, backgroundUnary( to_vector<int>( background_seeds ) ) );
}
static ProposalSettings::UnarySettings* make_UnarySettings5( int n_seed, int n_lvl_set, std::shared_ptr<UnaryFactory> fg_unary, list background_seeds, float min_size ) {
	return new ProposalSettings::UnarySettings( n_seed, n_lvl_set, fg_unary, backgroundUnary( to_vector<int>( background_seeds ) ), min_size );
}
static ProposalSettings::UnarySettings* make_UnarySettings6( int n_seed, int n_lvl_set, std::shared_ptr<UnaryFactory> fg_unary, list background_seeds, float min_size, float max_size ) {
	return new ProposalSettings::UnarySettings( n_seed, n_lvl_set, fg_unary, backgroundUnary( to_vector<int>( background_seeds ) ), min_size, max_size );
}
static ProposalSettings::UnarySettings* make_UnarySettings7( int n_seed, int n_lvl_set, std::shared_ptr<UnaryFactory> fg_unary, std::shared_ptr<UnaryFactory> bg_unary ) {
	return new ProposalSettings::UnarySettings( n_seed, n_lvl_set, fg_unary, bg_unary );
}
static ProposalSettings::UnarySettings* make_UnarySettings8( int n_seed, int n_lvl_set, std::shared_ptr<UnaryFactory> fg_unary, std::shared_ptr<UnaryFactory> bg_unary, float min_size ) {
	return new ProposalSettings::UnarySettings( n_seed, n_lvl_set, fg_unary, bg_unary, min_size );
}
static ProposalSettings::UnarySettings* make_UnarySettings9( int n_seed, int n_lvl_set, std::shared_ptr<UnaryFactory> fg_unary, std::shared_ptr<UnaryFactory> bg_unary, float min_size, float max_size ) {
	return new ProposalSettings::UnarySettings( n_seed, n_lvl_set, fg_unary, bg_unary, min_size, max_size );
}
static void ProposalSettings_set_foreground_seeds( ProposalSettings & that, const SeedFunction & f ) {
	that.foreground_seeds = f.clone();
}
static np::ndarray Salienct_saliency( const Saliency & sal, const ImageOverSegmentation & ios ) {
	return toNumpy( sal.saliency( ios ) );
}

void defineProposals() {
	ADD_MODULE(proposals);
	
	//***** Seeds *****//
	class_< SeedFunction, std::shared_ptr<SeedFunction>, boost::noncopyable >( "SeedFunction", no_init )
	.def("compute", &SeedFunction::compute);
	class_< ImageSeedFunction, std::shared_ptr<ImageSeedFunction>, bases<SeedFunction>, boost::noncopyable >( "ImageSeedFunction", no_init );
	class_< RegularSeed, std::shared_ptr<RegularSeed>, bases<ImageSeedFunction> >( "RegularSeed" );
	class_< GeodesicSeed, std::shared_ptr<GeodesicSeed>, bases<SeedFunction> >( "GeodesicSeed" )
	.def(init<float>())
	.def(init<float,float>())
	.def(init<float,float,float>())
	.def_pickle(SaveLoad_pickle_suite<GeodesicSeed>());
	class_< RandomSeed, std::shared_ptr<RandomSeed>, bases<SeedFunction> >( "RandomSeed" )
	.def_pickle(SaveLoad_pickle_suite<RandomSeed>());
	class_< SaliencySeed, std::shared_ptr<SaliencySeed>, bases<SeedFunction> >( "SaliencySeed" )
	.def_pickle(SaveLoad_pickle_suite<SaliencySeed>());
	class_< SegmentationSeed, std::shared_ptr<SegmentationSeed>, bases<SeedFunction> >( "SegmentationSeed" )
	.def_pickle(SaveLoad_pickle_suite<SegmentationSeed>());
	class_< LearnedSeed, std::shared_ptr<LearnedSeed>, bases<SeedFunction> >("LearnedSeed")
	.def( "train", &LearnedSeed_train1 )
	.def( "train", &LearnedSeed_train2 )
	.def( "load", static_cast<void(LearnedSeed::*)(const std::string &)>(&LearnedSeed::load) )
	.def( "save", static_cast<void(LearnedSeed::*)(const std::string &)const>(&LearnedSeed::save) )
	.def_pickle(SaveLoad_pickle_suite<LearnedSeed>());
	
	//***** Unaries *****//
	class_<UnaryFactory,std::shared_ptr<UnaryFactory>,boost::noncopyable>("UnaryFactory",no_init)
	.def("create",&UnaryFactory::create);
	
	class_<ProposalSettings::UnarySettings>("UnarySettings")
	.def("__init__",make_constructor(&make_UnarySettings1))
	.def("__init__",make_constructor(&make_UnarySettings2))
	.def("__init__",make_constructor(&make_UnarySettings3))
	.def("__init__",make_constructor(&make_UnarySettings4))
	.def("__init__",make_constructor(&make_UnarySettings5))
	.def("__init__",make_constructor(&make_UnarySettings6))
	.def("__init__",make_constructor(&make_UnarySettings7))
	.def("__init__",make_constructor(&make_UnarySettings8))
	.def("__init__",make_constructor(&make_UnarySettings9))
	.def_readwrite("fg_unary",&ProposalSettings::UnarySettings::fg_unary)
	.def_readwrite("bg_unary",&ProposalSettings::UnarySettings::bg_unary)
	.def_readwrite("edge_weight",&ProposalSettings::UnarySettings::edge_weight)
	.def_readwrite("n_seed",&ProposalSettings::UnarySettings::n_seed)
	.def_readwrite("n_lvl_set",&ProposalSettings::UnarySettings::n_lvl_set)
	.def_readwrite("min_size",&ProposalSettings::UnarySettings::min_size)
	.def_readwrite("max_size",&ProposalSettings::UnarySettings::max_size);
	
	class_< std::vector<ProposalSettings::UnarySettings> >("UnarySettingsVec")
	.def(vector_indexing_suite<std::vector<ProposalSettings::UnarySettings> >());
	
	class_<ProposalSettings>("ProposalSettings")
	.add_property("foreground_seeds",make_getter(&ProposalSettings::foreground_seeds),ProposalSettings_set_foreground_seeds)
// 	.def_readwrite("foreground_seeds",&ProposalSettings::foreground_seeds)
	.def_readwrite("unaries",&ProposalSettings::unaries)
	.def_readwrite("max_iou",&ProposalSettings::max_iou);
	
	class_<Proposal>("Proposal",init<ProposalSettings>())
	.def("propose",&Proposal::propose);
	
	class_<GeodesicDistance>("GeodesicDistance",init<OverSegmentation>())
	.def("compute",static_cast<VectorXf (GeodesicDistance::*)(int)const>( &GeodesicDistance::compute ))
	.def("compute",static_cast<VectorXf (GeodesicDistance::*)(const VectorXf &)const>( &GeodesicDistance::compute ))
	.add_property("N",&GeodesicDistance::N);
	
	class_<Unary,std::shared_ptr<Unary>,boost::noncopyable>("Unary",no_init)
	.def( "compute", Unary_compute );
	
	def("seedUnary",&seedUnary);
	def("rgbUnary",&rgbUnary);
	def("labUnary",&labUnary);
	def("zeroUnary",&zeroUnary);
	def("learnedUnary",static_cast<std::shared_ptr<UnaryFactory>(*)(const FeatureSet &,const VectorXf &, float)>(&learnedUnary));
	def("learnedUnary",static_cast<std::shared_ptr<UnaryFactory>(*)(const VectorXf &, float)>(&learnedUnary));
	def("learnedUnary",static_cast<std::shared_ptr<UnaryFactory>(*)(const std::string &)>(&learnedUnary));
	def("binaryLearnedUnary",static_cast<std::shared_ptr<UnaryFactory>(*)(const FeatureSet &,const VectorXf &, float)>(&binaryLearnedUnary));
	def("binaryLearnedUnary",static_cast<std::shared_ptr<UnaryFactory>(*)(const VectorXf &, float)>(&binaryLearnedUnary));
	def("binaryLearnedUnary",static_cast<std::shared_ptr<UnaryFactory>(*)(const std::string &)>(&binaryLearnedUnary));
	def("saveLearnedUnary",&saveLearnedUnary);
	
	class_<FeatureSet>("FeatureSet")
	.def("add",static_cast<void (FeatureSet::*)( const FeatureType & )>(&FeatureSet::add))
	.def("add",static_cast<void (FeatureSet::*)( const FeatureSet & )>(&FeatureSet::add))
	.def("has",&FeatureSet::has)
	.def("remove",&FeatureSet::remove)
	.def_pickle(SaveLoad_pickle_suite<FeatureSet>());
	
	def("defaultUnaryFeatures",defaultUnaryFeatures);
	
	class_<UnaryFeature,std::shared_ptr<UnaryFeature>,boost::noncopyable>("UnaryFeature",no_init)
	.def( "compute", static_cast<RMatrixXf (UnaryFeature::*)(int)const>(&UnaryFeature::compute) )
	.def( "featureId", &UnaryFeature::featureId )
	.def( "dim", &UnaryFeature::dim )
	.def( "N", &UnaryFeature::N )
	.add_static_property("Constant",make_getter(UnaryFeature::Constant))
	.add_static_property("Indicator",make_getter(UnaryFeature::Indicator))
	.add_static_property("InverseIndicator",make_getter(UnaryFeature::InverseIndicator))
	.add_static_property("Position",make_getter(UnaryFeature::Position))
	.add_static_property("RGB",make_getter(UnaryFeature::RGB))
	.add_static_property("Lab",make_getter(UnaryFeature::Lab))
	.add_static_property("RGBHistogram",make_getter(UnaryFeature::RGBHistogram))
	.add_static_property("LabHistogram",make_getter(UnaryFeature::LabHistogram))
	.add_static_property("BoundaryIndicator",make_getter(UnaryFeature::BoundaryIndicator))
	.add_static_property("BoundaryID",make_getter(UnaryFeature::BoundaryID))
	.add_static_property("BoundaryDistance",make_getter(UnaryFeature::BoundaryDistance));
	
	class_<UnaryFeatures>("UnaryFeatures",init<ImageOverSegmentation>())
	.def(init<ImageOverSegmentation,FeatureSet>())
	.def("subset", &UnaryFeatures::subset )
	.def("get", &UnaryFeatures::get )
	.def("dim", &UnaryFeatures::dim )
	.def("compute", &UnaryFeatures::compute );
	
	// Unary Learning
	class_<Saliency>("Saliency")
	.def("saliency",Salienct_saliency);
}
