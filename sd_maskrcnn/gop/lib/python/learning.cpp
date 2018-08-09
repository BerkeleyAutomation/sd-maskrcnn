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
#include "learning/features.h"
#include "learning/tree.h"
#include "learning/forest.h"
#include "util/util.h"
#include "util.h"
#include "gop.h"

class FeaturesNDArray: public Features {
protected:
	np::ndarray data_;
	const float * pdata_;
	int nS_, nF_;
public:
	FeaturesNDArray( const np::ndarray & data ):data_(data){
		checkArray( data_, float, 2, 2, true );
		nS_ = data_.shape(0);
		nF_ = data_.shape(1);
		pdata_ = (const float*)data_.get_data();
	}
	virtual int featureSize() const {
		return nF_;
	}
	virtual int nSamples() const {
		return nS_;
	}
	virtual float get(int s, int f) const {
		return pdata_[(size_t)s*(size_t)nF_+(size_t)f];
	}
};
typedef std::vector< std::shared_ptr<Features> > FeaturesVec;

void defineLearning() {
	ADD_MODULE(learning);
	
	class_<Features, std::shared_ptr<Features>, boost::noncopyable >("Features",no_init)
	.def("nSamples",&Features::nSamples)
	.def("featureSize",&Features::featureSize)
	.def("getFeatures",(RMatrixXf(Features::*)() const)&Features::getFeatures)
	.def("getFeatures",(RMatrixXf(Features::*)( const VectorXi & ) const)&Features::getFeatures)
	.def("getFeatures",(RMatrixXf(Features::*)( const VectorXi &, const VectorXi & ) const)&Features::getFeatures)
	.def("get",&Features::get);
	class_< FeaturesVec >("FeaturesVec")
	.def( vector_indexing_suite< FeaturesVec, true >() );
	
	class_< FeaturesMatrix, std::shared_ptr<FeaturesMatrix>, bases<Features> >("FeaturesMatrix",init<RMatrixXf>() );
	implicitly_convertible< std::shared_ptr<FeaturesMatrix>, std::shared_ptr<Features> >();
	class_< FeaturesNDArray, std::shared_ptr<FeaturesNDArray>, bases<Features> >("FeaturesNDArray",init<np::ndarray>());
	implicitly_convertible< std::shared_ptr<FeaturesNDArray>, std::shared_ptr<Features> >();
	class_< FeaturesVector, std::shared_ptr<FeaturesVector>, bases<Features> >("FeaturesVector", init< FeaturesVec, RMatrixXi >() )
	.def(init< FeaturesVec, RMatrixXi, VectorXi >());
	implicitly_convertible< std::shared_ptr<FeaturesVector>, std::shared_ptr<Features> >();
	
	class_< BinaryTree >("BinaryTree")
	.def("setFromMatlab",&BinaryTree::setFromMatlab )
	.add_property("max_depth",&BinaryTree::maxDepth)
	.add_property("average_depth",&BinaryTree::averageDepth);
	
	class_< BinaryForest >("BinaryForest")
	.def("addTree",&BinaryForest::addTree )
	.def("load",(void (BinaryForest::*)(std::string))&BinaryForest::load)
	.def("save",(void (BinaryForest::*)(std::string) const)&BinaryForest::save)
	.add_property("max_depth",&BinaryForest::maxDepth)
	.add_property("average_depth",&BinaryForest::averageDepth);
	
// Tree and Forest settings
	enum_<TreeSettings::Criterion>("Criterion")
	.value("GINI", TreeSettings::GINI)
	.value("ENTROPY", TreeSettings::ENTROPY)
	.value("STRUCT_GINI", TreeSettings::STRUCT_GINI)
	.value("STRUCT_ENTROPY", TreeSettings::STRUCT_ENTROPY);
	
	enum_<TreeSettings::MaxFeature>("MaxFeature")
	.value("SQRT", TreeSettings::SQRT)
	.value("LOG2", TreeSettings::LOG2)
	.value("POW06", TreeSettings::POW06)
	.value("POW07", TreeSettings::POW07)
	.value("ALL", TreeSettings::ALL);
	
	class_< TreeSettings >("TreeSettings")
	.def_readwrite("criterion",&TreeSettings::criterion )
	.def_readwrite("max_feature",&TreeSettings::max_feature )
	.def_readwrite("max_depth",&TreeSettings::max_depth )
	.def_readwrite("min_samples_split",&TreeSettings::min_samples_split )
	.def_readwrite("min_samples_leaf",&TreeSettings::min_samples_leaf )
	.def_readwrite("n_structured_samples",&TreeSettings::n_structured_samples )
	.def_readwrite("use_single_leaf_label",&TreeSettings::use_single_leaf_label)
	.def_readwrite("extremely_random",&TreeSettings::extremely_random );
	
	class_< ForestSettings,bases<TreeSettings> >("ForestSettings")
	.def_readwrite("n_trees",&ForestSettings::n_trees )
	.def_readwrite("replacement",&ForestSettings::replacement );
}
