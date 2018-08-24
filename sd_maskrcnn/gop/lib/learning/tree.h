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
#pragma once
#include "util/eigen.h"
#include "util/algorithm.h"
#include "features.h"
#include "treedata.h"
#include <vector>
#include <memory>

struct TreeSettings {
	enum Criterion{
		GINI,
		ENTROPY,
		STRUCT_GINI,
		STRUCT_ENTROPY,
	};
	enum MaxFeature{
		SQRT,
		LOG2,
		POW06,
		POW07,
		ALL
	};
	Criterion criterion = GINI;
	MaxFeature max_feature = SQRT;
	int max_depth = 1<<10;
	int min_samples_split = 0;
	int min_samples_leaf = 0;
	int n_structured_samples = 256;
	bool extremely_random = false;
	bool use_single_leaf_label = false;
};
class SplitCriterion;
class BaseTree {
protected:
	struct TreeNode {
		int fid=-1; // feature id
		float t=0; // threshold
		int left_child=-1;
	};
	std::vector<TreeNode> nodes_;
	virtual int addLeaf( const VectorXi & ids, const VectorXf & weight, const void * data );
	virtual void clearLeafData();
	void fitMT( const Features & f, const VectorXi & ids, const RMatrixXf & lbl, const VectorXf & weight, const TreeSettings & settings = TreeSettings(), const void * data=NULL );
	void fit( const Features & f, const VectorXi & ids, const RMatrixXf & lbl, const VectorXf & weight, const TreeSettings & settings = TreeSettings(), const void * data=NULL );
	void refit( const Features & f, const VectorXi & ids, const RMatrixXf & lbl, const VectorXf & weight, const TreeSettings & settings = TreeSettings(), const void * data=NULL );
public:
	void remapFid( const VectorXi & fid );
	int maxDepth() const;
	float averageDepth() const;
	VectorXi predict( const Features & f, const VectorXi & ids ) const;
	int predict( const Features & f, int id ) const;
	VectorXi predict( const Features & f ) const;
	virtual void save( std::ostream & s ) const;
	virtual void load( std::istream & s );
	void set( const BaseTree & o ) {
		nodes_ = o.nodes_;
	}
};

template<typename T,bool SINGLE_LEAF_LABEL=false>
class Tree: public BaseTree {
protected:
	template<typename TT, bool SL> friend class Tree;
	std::vector<T> data_;
	virtual int addLeaf( const VectorXi & ids, const VectorXf & weight, const void * data );
	virtual void clearLeafData() {
		data_.clear();
	}
public:
	typedef T DataType;
template<bool MT=false>
	void fit( const Features & f, const VectorXi & ids, const RMatrixXf & lbl, const VectorXf & weight, const std::vector<T> & data, TreeSettings settings = TreeSettings() ) {
		if ( SINGLE_LEAF_LABEL )
			settings.use_single_leaf_label = true;
		if( MT )
			BaseTree::fitMT( f, ids, lbl, weight, settings, data.data() );
		else
			BaseTree::fit( f, ids, lbl, weight, settings, data.data() );
	}
	void refit( const Features & f, const VectorXi & ids, const RMatrixXf & lbl, const VectorXf & weight, const std::vector<T> & data, TreeSettings settings = TreeSettings() ) {
		if ( SINGLE_LEAF_LABEL )
			settings.use_single_leaf_label = true;
		BaseTree::refit( f, ids, lbl, weight, settings, data.data() );
	}
	void setData( const Features & f, const VectorXi & ids, const std::vector<T> & data ) {
		VectorXi nds = predict( f, ids );
		data_ = std::vector<T>( nodes_.size() );
		for( int i=0; i<nds.size(); i++ )
			data_[ nds[i] ] += data[i];
	}
	void setData( const Features & f, const std::vector<T> & data ) {
		setData( f, arange(f.nSamples()), data );
	}
	std::vector<T> predictData( const Features & f, const VectorXi & ids ) const {
		VectorXi nds = predict( f, ids );
		std::vector<T> r = std::vector<T>( ids.size() );
		for( int i=0; i<nds.size(); i++ )
			r[ i ] = data_[ nds[i] ];
		return r;
	}
	std::vector<T> predictData( const Features & f ) const {
		return predictData( f, arange(f.nSamples()) );
	}
	T predictData( const Features & f, int id ) const {
		return data_[ predict( f, id) ];
	}
	std::vector<T> & data() {
		return data_;
	}
	const std::vector<T> & data() const {
		return data_;
	}
	void save( std::ostream & s ) const {
		BaseTree::save( s );
		int n = data_.size();
		s.write( (const char*)&n, sizeof(n) );
		for( const auto & i: data_ )
			i.save(s);
	}
	void load( std::istream & s ) {
		BaseTree::load( s );
		int n;
		s.read( (char*)&n, sizeof(n) );
		data_.resize(n);
		for( auto & i: data_ )
			i.load(s);
	}
	void set( const Tree<T> & t ) {
		BaseTree::set( t );
		data_ = t.data_;
	}
template<typename S, bool SL>
	void set( const Tree<S,SL> & t ) {
		BaseTree::set( t );
		data_ = std::vector<T>( t.data_.size() );
	}
};
namespace private_ {
	template<typename T>
	int addLeaf( Tree<T,true> * that, const VectorXi & ids, const VectorXf & weight, const void * data ) {
		assert( ids.size() == 1 );
		const T * d = (const T*)data;
		const int id = that->data().size();
		that->data().push_back( d[ ids[0] ] );
		return id;
	}
	template<typename T>
	int addLeaf( Tree<T,false> * that, const VectorXi & ids, const VectorXf & weight, const void * data ) {
		const T * d = (const T*)data;
		T r = d[ ids[0] ];
		for( int i=1; i<ids.size(); i++ )
			r += d[ ids[i] ];
		const int id = that->data().size();
		that->data().push_back( r );
		return id;
	}
}
template<typename T, bool SINGLE_LEAF_LABEL>
int Tree<T,SINGLE_LEAF_LABEL>::addLeaf( const VectorXi & ids, const VectorXf & weight, const void * data ) {
	return private_::addLeaf( this, ids, weight, data );
}
class BinaryTree: public Tree<BinaryDistribution> {
public:
	VectorXf predictProb( const Features & f, const VectorXi & ids ) const;
	VectorXf predictProb( const Features & f ) const;
	void setFromMatlab(const VectorXf &thrs, const VectorXi &child, const VectorXi &fid, const RMatrixXf &dist);
};
class LabelTree: public Tree<LabelData,true> {
};
class RangeTree: public Tree<RangeData,true> {
public:
	void setFromMatlab(const VectorXf &thrs, const VectorXi &child, const VectorXi &fid, const VectorXi &rng);
};
class PatchTree: public Tree<PatchData> {
public:
	void setFromMatlab(const VectorXf &thrs, const VectorXi &child, const VectorXi &fid, const VectorXi &rng, const VectorXus & patch);
};
