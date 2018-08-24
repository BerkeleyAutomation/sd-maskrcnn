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
#include "util/graph.h"
#include <queue>

class OverSegmentation;
struct Node {
	explicit Node(int to=0, float w=0);
	int to;
	float w;
	bool operator<( const Node & o ) const;
	bool operator>( const Node & o ) const;
};
struct Node2 {
	explicit Node2(int to=0, int from=0, float w=0);
	int to, from;
	float w;
	bool operator<( const Node2 & o ) const;
	bool operator>( const Node2 & o ) const;
};

struct GeodesicDistance {
protected:
	typedef std::priority_queue< Node ,std::vector<Node >,std::greater<Node > > PQ;
	typedef std::priority_queue< Node2,std::vector<Node2>,std::greater<Node2> > PQ2;
	virtual void updatePQ( VectorXf & d, PQ & q ) const;
	virtual VectorXf backPropGradientPQ( PQ2 & q, const VectorXf & d ) const;
	
	const int N_;
	VectorXf d_;
	std::vector< std::vector< Node > > adj_list_;
public:
	// Geodesic distance functions
	GeodesicDistance( const OverSegmentation & os );
	GeodesicDistance( const Edges & edges, const VectorXf & edge_weights );
	
	virtual GeodesicDistance& reset( float v = 1e10 );
	virtual GeodesicDistance& update( int nd, float v = 0 );
	virtual GeodesicDistance& update( const VectorXf & new_min );
	virtual VectorXf compute( int nd ) const;
	virtual VectorXf compute( const VectorXf & start ) const;
	virtual RMatrixXf compute( const RMatrixXf & start ) const;
	virtual VectorXf backPropGradient( int nd, const VectorXf & g ) const;
	virtual VectorXf backPropGradient( const VectorXf & start, const VectorXf & g ) const;
	virtual int N() const;
	virtual const VectorXf & d() const;
};

int geodesicCenter( const Edges & edges, const VectorXf & edge_weights );
