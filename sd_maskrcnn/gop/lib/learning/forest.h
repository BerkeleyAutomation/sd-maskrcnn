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
#include "tree.h"
#include <stdexcept>
#include <iostream>
#include <fstream>
//#include "features.h"
//#include <algorithm>


struct ForestSettings: public TreeSettings {
	int n_trees = 10;
	bool replacement = true;
};

template<typename T1,typename T2>
void add( std::vector<T1> & a, const std::vector<T2> & b ) {
	for( int i=0; i<a.size() && i<b.size(); i++ )
		a[i] += b[i];
}
template<typename T>
class Forest {
protected:
	std::vector< T > trees_;
public:
	typedef typename T::DataType DataType;
	typedef typename DataType::AccumType AccumType;
	void addTree( const T & tree ) {
		trees_.push_back( tree );
	}
	int nTrees() const {
		return trees_.size();
	}
	const std::vector<T> trees() const {
		return trees_;
	}
	const T & tree( int i ) const {
		return trees_[i];
	}
	T & tree( int i ) {
		return trees_[i];
	}
	int maxDepth() const {
		int md = 0;
		for( const auto & t: trees_ )
			md = std::max( md, t.maxDepth() );
		return md;
	}
	float averageDepth() const {
		float md = 0;
		for( const auto & t: trees_ )
			md += t.averageDepth();
		return md / trees_.size();
	}
	std::vector<AccumType> predict( const Features & f, const VectorXi & ids ) const {
		std::vector<AccumType> r;
		for( const auto & t: trees_ ) {
			if( r.size()==0 )
				r = t.predictData( f, ids );
			else
				add( r, t.predictData( f, ids ) );
		}
		return r;
	}
	std::vector<AccumType> predict( const Features & f ) const {
		return predict( f, arange(f.nSamples()) );
	}
	void save( std::ostream &s ) const {
#define put(x) s.write( (const char*)&x, sizeof(x) )
		int n = trees_.size();
		put(n);
		for( const auto & i: trees_ )
			i.save(s);
#undef put
	}
	void load( std::istream &s ) {
#define get(x) s.read( (char*)&x, sizeof(x) )
		int n;
		get(n);
		trees_.resize(n);
		for( auto & i: trees_ )
			i.load(s);
#undef get
	}
	void save( std::string fn ) const {
		std::ofstream fs( fn, std::ios::out | std::ios::binary );
		if(!fs.is_open())
			throw std::invalid_argument( "Could not write file '"+fn+"'!" );
		save( fs );
	}
	void load( std::string fn ) {
		std::ifstream fs( fn, std::ios::in | std::ios::binary );
		if(!fs.is_open())
			throw std::invalid_argument( "Could not open file '"+fn+"'!" );
		load( fs );
	}
};
class BinaryForest: public Forest<BinaryTree> {
public:
	VectorXf predictProb( const Features & f, const VectorXi & ids ) const;
	VectorXf predictProb( const Features & f ) const;
};
class LabelForest: public Forest<LabelTree> {
public:
};
class RangeForest: public Forest<RangeTree> {
public:
};
class PatchForest: public Forest<PatchTree> {
public:
};

