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
#include "covering.h"
#include "python/map.h"
#include <Eigen/Sparse>
using namespace Eigen;
#include <iostream>
#include <unordered_set>

const float RECALL_OVERLAP = 0.5;

VectorXf bestOverlap( const std::vector<Polygons> & regions, const RMatrixXs &over_seg, const RMatrixXb &segments, VectorXf * area = NULL ) {
	const int N_sp = over_seg.maxCoeff()+1, N_gt = regions.size();
	
	// Compute the sparse proposal matrix
	SparseMatrix<float> props( segments.rows(), segments.cols() );
	std::vector< Triplet<float> > t;
	for( int j=0; j<props.rows(); j++ )
		for( int i=0; i<props.cols(); i++ ) 
			if( segments(j,i) )
				t.push_back( Triplet<float>(j,i,1) );
	props.setFromTriplets( t.begin(), t.end() );
	
	// Compute the superpixel areas
	VectorXf sp_area = VectorXf::Zero( N_sp );
	for( int j=0; j<over_seg.rows(); j++ )
		for( int i=0; i<over_seg.cols(); i++ )
			sp_area[ over_seg(j,i) ]++;
		
	if(area)
		*area = VectorXf::Zero( N_gt );
	VectorXf r( N_gt );
	// Compute the best overlap with each GT object
	for( int i=0; i<N_gt; i++ ) {
		float gt_area = 0;
		VectorXf cur_sp_area = sp_area;
		SparseVector<float> intersection( N_sp );
		rasterize( [&](int x,int y,RasterType t){
			if (0<=x && x<over_seg.cols() && 0<=y && y<over_seg.rows()) {
				const int s = over_seg(y,x);
				if( t==INSIDE ){
					gt_area += 1;
					intersection.coeffRef( s ) += 1;
				}
				// Ignore boundary pixels
				else if (t==OUTSIDE_BOUNDARY)
					cur_sp_area[s]-=1;
			}
		}, regions[i] );
		VectorXf prop_area = props * cur_sp_area;
		SparseVector<float> prop_intersection = props * intersection;
		
		float bo = 0;
		for( SparseVector<float>::InnerIterator it(prop_intersection); it; ++it )
			bo = std::max( bo, it.value() / (gt_area + prop_area[it.row()] - it.value()) );
		
		r[i] = bo;
		if(area)
			(*area)[i] = gt_area;
	}
	return r;
}

VectorXf bestOverlap( const short * gt_seg, int W, int H, int D, const RMatrixXs &over_seg, const RMatrixXb &segments, VectorXf * area = NULL ) {
	if( W != over_seg.cols() || H != over_seg.rows() )
		throw std::invalid_argument("Ground truth and over segmentation shape does not match!");
	Map<const VectorXs> sp( (const short*)over_seg.data(), W*H );
	Map<const RMatrixXs> gt( gt_seg, D, W*H );
	
	VectorXs N_sgt = gt.array().rowwise().maxCoeff()+1;
	const int N_gt = N_sgt.array().sum(), N_sp = sp.maxCoeff()+1;
	if( N_sp != segments.cols() )
		throw std::invalid_argument("Number of superpixels does not match segment size!");
	
	VectorXf gt_area = VectorXf::Zero( N_gt ), sp_area = VectorXf::Zero( N_sp );
	SparseMatrix<float> intersection( N_gt, N_sp );
	for( int i=0; i<W*H; i++ ) {
		const int s = sp[i];
		bool cnt_area = false;
		for( int d=0,o=0; d<D; o+=N_sgt[d++] ) {
			if( !cnt_area )
				cnt_area = (gt(d,i)>=-1);
			if( gt(d,i)>=0 ){
				const int t = o+gt(d,i);
				gt_area[t]++;
				if( s >= 0 ) {
					intersection.coeffRef(t,s) += 1;
				}
			}
		}
		if( cnt_area && s >= 0 )
			sp_area[s]++;
	}
	intersection.makeCompressed();
	
	SparseMatrix<float> props( segments.rows(), segments.cols() );
	std::vector< Triplet<float> > t;
	for( int j=0; j<props.rows(); j++ )
		for( int i=0; i<props.cols(); i++ ) 
			if( segments(j,i) )
				t.push_back( Triplet<float>(j,i,1) );
	props.setFromTriplets( t.begin(), t.end() );
	
	VectorXf seg_area = props * sp_area;
	SparseMatrix<float> seg_intersection = intersection * props.transpose();
	
	VectorXf r = VectorXf::Zero( N_gt );
	for (int k=0; k<seg_intersection.outerSize(); ++k)
		for (SparseMatrix<float>::InnerIterator it(seg_intersection,k); it; ++it)
			r[it.row()] = std::max( r[it.row()], it.value() / ( seg_area[ it.col() ] + gt_area[ it.row() ] - it.value() ) );
	if( area )
		*area = gt_area;
	return r;
}

ProposalEvaluation::ProposalEvaluation(const std::vector<Polygons> & regions, const std::vector<RMatrixXs> & over_seg, const std::vector<RMatrixXb> & props) {
	if( over_seg.size() != props.size() )
		throw std::invalid_argument("Different number of over segmentations and proposals!");
	if( over_seg.size() < 1 )
		throw std::invalid_argument("At least one proposal required!");
	
	VectorXf area, bo;
	pool_size_ = 0;
	for( int k=0; k <over_seg.size(); k++ ) {
		VectorXf o = bestOverlap( regions, over_seg[k], props[k], &area );
		if( k )
			bo = bo.array().max( o.array() );
		else
			bo = o;
		pool_size_ += props[k].rows();
	}
	bo_ = bo;
	area_ = area;
}
ProposalEvaluation::ProposalEvaluation(const std::vector<Polygons> & regions, const RMatrixXs &over_seg, const RMatrixXb &props ):ProposalEvaluation( regions, std::vector<RMatrixXs>(1,over_seg), std::vector<RMatrixXb>( 1, props ) ) {
}
ProposalEvaluation::ProposalEvaluation(const short int *gt_seg, int W, int H, int D, const std::vector<RMatrixXs> & over_seg, const std::vector<RMatrixXb> & props) {
	if( over_seg.size() != props.size() )
		throw std::invalid_argument("Different number of over segmentations and proposals!");
	if( over_seg.size() < 1 )
		throw std::invalid_argument("At least one proposal required!");
	
	VectorXf area, bo;
	pool_size_ = 0;
	for( int k=0; k <over_seg.size(); k++ ) {
		VectorXf o = bestOverlap( gt_seg, W, H, D, over_seg[k], props[k], &area );
		if( k )
			bo = bo.array().max( o.array() );
		else
			bo = o;
		pool_size_ += props[k].rows();
	}
	bo_ = bo;
	area_ = area;
}
ProposalEvaluation::ProposalEvaluation( const short int *gt_seg, int W, int H, int D, const RMatrixXs &over_seg, const RMatrixXb &props ):ProposalEvaluation( gt_seg, W, H, D, std::vector<RMatrixXs>(1,over_seg), std::vector<RMatrixXb>( 1, props )/*, recall_overlap*/ ) {
}
float boxOverlap( const VectorXi & b1, const VectorXi & b2 ) {
	float intersec = (std::max(0,std::min(b1[2],b2[2]) - std::max(b1[0],b2[0]))*std::max(0,std::min(b1[3],b2[3]) - std::max(b1[1],b2[1])));
	float _union   = (std::max(0,std::max(b1[2],b2[2]) - std::min(b1[0],b2[0]))*std::max(0,std::max(b1[3],b2[3]) - std::min(b1[1],b2[1])));
	if( _union > 0 )
		return intersec / _union;
	return 0;
}
ProposalBoxEvaluation::ProposalBoxEvaluation(const RMatrixXi &bbox, const std::vector<RMatrixXi> & prop_boxes) {
	if( prop_boxes.size() < 1 )
		throw std::invalid_argument("At least one proposal required!");
	
	if (bbox.rows() == 0) {
		bo_ = VectorXf::Zero(0);
		pool_size_ = NAN;
		return;
	}
	
	VectorXf bo = VectorXf::Zero( bbox.rows() );
	pool_size_ = 0;
	for( int k=0; k<prop_boxes.size(); k++ ) {
		const int nProp = prop_boxes[k].rows();
		// Compute the pool size
		std::unordered_set<uint64_t> ht;
		for( int j=0; j<nProp; j++ ) {
			uint64_t h = 0;
			for( int l=0; l<4; l++ )
				h = (h<<16) + ((unsigned short)prop_boxes[k](j,l));
			if( !ht.count( h ) ) {
				ht.insert( h );
				pool_size_++;
			}
		}
		
		for( int i=0; i<bo.size(); i++ )
			for( int j=0; j<nProp; j++ )
				bo[i] = std::max( bo[i], boxOverlap( prop_boxes[k].row(j), bbox.row(i) ) );
	}
	bo_ = bo;
}
ProposalBoxEvaluation::ProposalBoxEvaluation( const RMatrixXi &bbox, const RMatrixXi &prop_boxes):ProposalBoxEvaluation( bbox, std::vector<RMatrixXi>(1,prop_boxes) ) {
}
