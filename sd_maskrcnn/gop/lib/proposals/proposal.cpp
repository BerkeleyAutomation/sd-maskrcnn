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
#include "proposal.h"
#include "geodesics.h"
#include "util/util.h"
#include "edgefeature.h"
#include "segmentation/iouset.h"
#include "segmentation/segmentation.h"
#include <queue>
#include <set>
#include <iostream>

const float LVL_SET_BLUR=1.5;

Proposal::Proposal(const ProposalSettings &psettings):psettings_(psettings) {
}
void Proposal::proposeAll( const ImageOverSegmentation & ios, const VectorXi & seeds, const std::function<void(const ArrayXf &,float,float,int,int,int)> & f ) const {
	// Iterate over all unaries
	int nu=0;
	// Compute all the features we need for our proposals
	FeatureSet feature_set;
	for( const auto & us: psettings_.unaries ) {
		feature_set.add( us.fg_unary->requiredFeatures() );
		feature_set.add( us.bg_unary->requiredFeatures() );
	}
	UnaryFeatures features( ios, feature_set );
	
	for( const auto & us: psettings_.unaries ) {
		std::shared_ptr<Unary> fg_unary = us.fg_unary->create( features ), bg_unary = us.bg_unary->create( features );
		eassert( us.fg_unary->dim()==1 );
		// we cant generate background only segmentations with non static background unaries
		if( !us.bg_unary->isStatic() && us.n_seed==0 )
			continue;
		
		// Seed only proposal
		if( us.max_size == 0 ) {
			for( int k=0; k<us.n_seed && k<seeds.size(); k++ ) {
				VectorXf fg = VectorXf::Ones( ios.Ns() );
				fg[seeds[k]] = 0;
				f( fg, 0.5, 1e-3, nu, k, 0 );
			}
			continue;
		}
		
		GeodesicDistance gdist( ios.edges(), us.edge_weight->compute( ios ) );
		// Compute the BG geodesics
		std::vector<VectorXf> background;
		RMatrixXf bg;
		if( us.bg_unary->isStatic() )
			bg = gdist.compute( bg_unary->compute(0) );
		// Compute the FG geodesics
		for( int k=-!us.n_seed; k<us.n_seed && k<seeds.size(); k++ ) {
			// Compute the foreground geodesics
			VectorXf fg = VectorXf::Zero( ios.Ns() );
			if( k>=0 )
				fg = gdist.compute( fg_unary->compute( seeds[k] ) );
			if( !us.bg_unary->isStatic() )
				bg = gdist.compute( bg_unary->compute( seeds[k] ) );
			
			int nb=0;
			for( int j=0; j<bg.cols(); j++ ) {
				VectorXf d = fg - bg.col(j);
				// Find levelsets and segments
				VectorXf score;
				VectorXf bp = computeLevelSets( d, us.n_lvl_set, psettings_.max_iou, ios.Ns()*us.min_size, ios.Ns()*us.max_size, &score );
				for( int i=0; i<bp.size(); i++ )
					f( d, bp[i], score[i], nu, k>=0?seeds[k]:-1, j );
				nb++;
			}
		}
		nu++;
	}
}
RMatrixXb Proposal::propose(const ImageOverSegmentation &ios ) const {
// 	GeodesicDistance gdist( ios.edges(), ios.edgeWeights().array().pow(3)+2e-3 );
	
	std::vector<VectorXb> proposals;
	std::vector<float> scores;
	
	// Compute all the seeds we would ever want
	VectorXi seeds = makeSeeds( ios );
	proposeAll( ios, seeds, [&]( const ArrayXf & d, float t, float s, int, int, int ){ proposals.push_back( d < t); scores.push_back( s ); } );
	
	// Keep only unique proposals
	VectorXi order = range(proposals.size());
	std::sort( order.data(), order.data()+order.size(), [&](int a, int b)->bool{ return scores[a] > scores[b] || (scores[a] == scores[b] && a<b); } );
	
	// Filter out all duplicates
	std::vector<VectorXb> unique_props;
	IOUSet iou_set( ios );
	
	//	for( int i: order )
	for( int k=0; k<order.size(); k++ ) {
		const int i = order[k];
		if( proposals[i].any() && !iou_set.intersects( proposals[i], psettings_.max_iou ) ) {
			iou_set.add( proposals[i] );
			unique_props.push_back( proposals[i] );
		}
	}
	
// 	unique_props = proposals;
	// Return the unique proposal matrix
	RMatrixXb r(unique_props.size(),ios.Ns());
	for( int i=0; i<unique_props.size(); i++ )
		r.row(i) = unique_props[i].transpose();
	return r;
}
VectorXf Proposal::computeLevelSets(const VectorXf &d, int n_lvl_set, float max_iou, int min_size, int max_size, VectorXf * score) {
	const float MAX_D = 1e9;
	const int D = d.size();
	if( max_size < 0 )
		max_size = D;
	if( n_lvl_set < 0 )
		n_lvl_set = D;
	VectorXf x = 1*d;
	std::sort( &x[0], &x[D-1] );
	
	std::vector<Node> b; // We could use a PQ here to sort the element (for O( log(n_lvl_set) ), but it's most likely overkill)
	for( int i=std::max(min_size,2); i<D && i<=max_size && x[i]<MAX_D; i++ ) {
		float x0 = x[ std::max(0,i-1) ], x1 = x[ i ], x2 = x[ std::min(D-1,i+1) ], x3 = x[ std::min(D-1,i+2) ];
		if(x2 > MAX_D)
			x2 = x3 = x1;
		if(x3 > MAX_D)
			x3 = x2;
		const float dx1 = x1-x0, dx2 = x2-x1, dx3 = x3-x2;
		// Add the current threshold if it's a inflection point (and it's not a plateau)
		if( dx2 > 0 && dx2 > dx1 && dx2 > dx3 )
			b.push_back( Node( i, dx2 ) );
	}
	if( b.size() == 0 ) {
		// Return at least one level set
		if(score)
			*score = VectorXf::Zero(1);
		return VectorXf::Ones(1) * ((x[0]+x[D-1])/2);
	}
	std::sort( b.begin(), b.end(), std::greater<Node>() );
	std::set<int> selected;
	std::vector<float> r;
	std::vector<float> vals;
	for( const Node & n: b ) {
		auto up = selected.upper_bound( n.to );
		// Use relative error
		if( up != selected.end() && n.to > max_iou*(*up)  )
			continue;
		if( up != selected.begin() && (*--up) > max_iou*n.to )
			continue;
		
		selected.insert( n.to );
		vals.push_back( n.w );
		r.push_back( 0.5*(x[n.to]+x[n.to+1]) );
		if( r.size() >= n_lvl_set )
			break;
	}
	if( score )
		*score = VectorXf::Map(vals.data(), vals.size());
	return VectorXf::Map(r.data(), r.size());
}
VectorXi Proposal::makeSeeds( const ImageOverSegmentation & ios ) const {
	int max_n_seed = 0;
	for( const auto & us: psettings_.unaries )
		if( max_n_seed < us.n_seed )
			max_n_seed = us.n_seed;
	return psettings_.foreground_seeds->compute( ios, max_n_seed );
}
