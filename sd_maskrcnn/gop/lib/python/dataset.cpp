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
#include "dataset/apng.h"
#include "dataset/berkeley.h"
#include "dataset/coco.h"
#include "dataset/covering.h"
#include "dataset/evaluation.h"
#include "dataset/image.h"
#include "dataset/nyu.h"
#include "dataset/voc.h"
#include "dataset/weizmann.h"
#include "gop.h"
#include "util.h"
#include "map.h"
#include <proposals/proposal.h>


tuple proposeAndEvaluate( const std::vector< std::shared_ptr<ImageOverSegmentation> > & ios, const list & gt_segs, const list & gt_boxes, const Proposal & prop ) {
	const int N = ios.size();
	const bool has_segs = len(gt_segs)>0, has_box = len(gt_boxes)>0;
	
	if( has_segs && len(gt_segs)!=N )
		throw std::invalid_argument("Same number of oversegs and ground truth segments required!");
	if( has_box && len(gt_boxes)!=N )
		throw std::invalid_argument("Same number of oversegs and ground truth gt_boxes required!");
	
	// Convert the Segment Data
	std::vector<const short*> gt_data(len(gt_segs));
	std::vector<int> W(len(gt_segs)),H(len(gt_segs)),D(len(gt_segs));
	std::vector< std::vector<Polygons> > regions(len(gt_segs));
	bool has_regions = len( gt_segs )>0 && extract< std::vector<Polygons> >( gt_segs[0] ).check();
	for( int i=0; i<len(gt_segs); i++ ) {
		if( has_regions )
			regions[i] = extract< std::vector<Polygons> >( gt_segs[i] );
		else {
			np::ndarray gt_seg = extract<np::ndarray>( gt_segs[i] );
			checkArray( gt_seg, short, 2, 3, true );
			W[i] = gt_seg.shape(gt_seg.get_nd()-1);
			H[i] = gt_seg.shape(gt_seg.get_nd()-2);
			D[i] = gt_seg.get_nd()<3?1:gt_seg.shape(0);
			gt_data[i] = (const short*)gt_seg.get_data();
		}
	}
	// Convert the Bounding Box Data
	std::vector<RMatrixXi> box_data(len(gt_boxes));
	for( int i=0; i<len(gt_boxes); i++ ) {
		box_data[i] = extract<RMatrixXi>( gt_boxes[i] );
		if( box_data[i].cols() != 4 )
			throw std::invalid_argument("Nx4 gt_boxes expected!");
	}
	
	std::vector< VectorXf > bo(N), area(N), box_bo(N);
	VectorXf pool_size(N), box_pool_size(N);
	int n=0, box_n=0;
#pragma omp parallel for
	for( int i=0; i<N; i++ ) {
		RMatrixXb props = prop.propose( *ios[i] );
		if( has_segs && has_regions ) {
			ProposalEvaluation peval( regions[i], ios[i]->s(), props );
			bo[i] = peval.bo_;
			area[i] = peval.area_;
			pool_size[i] = peval.pool_size_;
#pragma omp atomic
			n += peval.bo_.size();
		}
		else if( has_segs ) {
			ProposalEvaluation peval( gt_data[i], W[i], H[i], D[i], ios[i]->s(), props );
			bo[i] = peval.bo_;
			area[i] = peval.area_;
			pool_size[i] = peval.pool_size_;
#pragma omp atomic
			n += peval.bo_.size();
		}
		if( has_box ) {
			RMatrixXi boxes = ios[i]->maskToBox( props );
			ProposalBoxEvaluation beval( box_data[i], boxes );
			box_bo[i] = beval.bo_;
			box_pool_size[i] = beval.pool_size_;
#pragma omp atomic
			box_n += beval.bo_.size();
		}
	}
	RMatrixXf r_bo = RMatrixXf::Zero( n, 2 );
	VectorXf r_b_bo = VectorXf::Zero( box_n );
	for( int i=0,k=0; i<N; i++ )
		for( int j=0; j<bo[i].size(); j++, k++ ) {
			r_bo(k,0) = bo[i][j];
			r_bo(k,1) = area[i][j];
		}
	for( int i=0,k=0; i<N; i++ )
		for( int j=0; j<box_bo[i].size(); j++, k++ )
			r_b_bo[k] = box_bo[i][j];
	return make_tuple( toNumpy(r_bo), toNumpy(r_b_bo), toNumpy(pool_size), toNumpy(box_pool_size) );
}
template<typename T> std::vector< RMatrixX<T> > mapMatrixXList( const list & l ) {
	std::vector< RMatrixX<T> > r;
	for( int i=0; i<len(l); i++ )
		r.push_back( mapMatrixX<T>( extract<np::ndarray>(l[i]) ) );
	return r;
}
template<typename T> RMatrixX<T> mapMatrixXList( const np::ndarray & l ) {
	return mapMatrixX<T>( l );
}
static std::vector<RMatrixXi> maskToBox( const std::vector<RMatrixXs> & s, const std::vector<RMatrixXb> & p ) {
	eassert( s.size() == p.size() );
	std::vector<RMatrixXi> r( s.size() );
	for( int i=0; i<(int)s.size(); i++ )
		r[i] = maskToBox( s[i], p[i] );
	return r;
}
template<typename T>
tuple evaluate( const T & s, const T & prop, const np::ndarray & gt_segs, const np::ndarray & gt_boxes ) {
	const bool has_segs = !gt_segs.is_none(), has_box = !gt_boxes.is_none();
	VectorXf res = VectorXf::Zero( 11 );
	
	auto ms = mapMatrixXList<short>( s );
	auto mprop = mapMatrixXList<bool>( prop );
	
	RMatrixXf bo;
	VectorXf b_bo;
	float n=0, box_n=0;
	if( has_segs ) {
		np::ndarray gt_seg = extract<np::ndarray>( gt_segs );
		checkArray( gt_seg, short, 2, 3, true );
		const int W = gt_seg.shape(gt_seg.get_nd()-1), H = gt_seg.shape(gt_seg.get_nd()-2), D = gt_seg.get_nd()<3?1:gt_seg.shape(0);
		const short * gt_data = (const short*)gt_seg.get_data();
		
		ProposalEvaluation peval( gt_data, W, H, D, ms, mprop );
		bo = RMatrixXf( peval.bo_.size(), 2 );
		bo.col(0) = peval.bo_;
		bo.col(1) = peval.area_;
		n = peval.pool_size_;
	}
	if( has_box ) {
		RMatrixXi mboxes = mapMatrixX<int>( gt_boxes );
		ProposalBoxEvaluation beval( mboxes, maskToBox( ms, mprop ) );
		b_bo = beval.bo_;
		box_n = beval.pool_size_;
	}
	return make_tuple( toNumpy(bo), toNumpy(b_bo), n, box_n );
}
template<typename T>
RMatrixXf evaluate2( const T & s, const T & prop, const std::vector<Polygons> & regions ) {
	auto ms = mapMatrixXList<short>( s );
	auto mprop = mapMatrixXList<bool>( prop );
	
	RMatrixXf bo = RMatrixXf::Zero( regions.size(), 2 );
	
	ProposalEvaluation peval( regions, ms, mprop );
	bo.col(0) = peval.bo_;
	bo.col(1) = peval.area_;
	return bo;
}
class EmptyDirs{};
BOOST_PYTHON_FUNCTION_OVERLOADS( loadWeizmann_overload, loadWeizmann, 2, 3 )
BOOST_PYTHON_FUNCTION_OVERLOADS( loadGrabcut_overload, loadGrabcut, 2, 3 )

void defineDataset() {
	ADD_MODULE(dataset);
	def("imread", imreadShared);
	def("imwrite", imwrite);
	def("readAPNG", readAPNG);
	def("writeAPNG", writeAPNG);
	def("loadWeizmann", loadWeizmann, loadWeizmann_overload(args("train", "test", "n_train"), "Load the Weizmann horse dataset"));
	def("loadBSD500", loadBSD500);
	def("loadBSD300", loadBSD300);
	def("loadBSD50", loadBSD50);
	def("loadVOC2007", loadVOC2007);
	def("loadVOC2007_detect", loadVOC2007_detect);
	def("loadVOC2007_detect_noim", loadVOC2007_detect_noim);
	def("loadVOC2010", loadVOC2010);
	def("loadVOC2012", loadVOC2012);
	def("loadVOC2012_small", loadVOC2012_small);
	def("loadVOC2012_detect", loadVOC2012_detect);
	def("loadNYU_nocrop", loadNYU_nocrop);
	def("loadNYU04_nocrop", loadNYU04_nocrop);
	def("loadNYU40_nocrop", loadNYU40_nocrop);
	def("loadNYU", loadNYU);
	def("loadNYU04", loadNYU04);
	def("loadNYU40", loadNYU40);
	def("labelsNYU", labelsNYU);
	def("labelsNYU04", labelsNYU04);
	def("labelsNYU40", labelsNYU40);
	def("loadCOCO2014", loadCOCO2014);
	def("cocoNFolds", cocoNFolds);
	def("matchAny", matchAny );
	def("matchBp", matchBp );
	def("evalBoundaryBinary", evalBoundaryBinary );
	def("evalBoundary", evalBoundary );
	def("evalBoundaryAll", evalBoundaryAll );
	
	def("proposeAndEvaluate",proposeAndEvaluate);
	
	def("evaluate",evaluate<list>);
	def("evaluate",evaluate<np::ndarray>);
	def("evaluate",evaluate2<list>);
	def("evaluate",evaluate2<np::ndarray>);
	
	class_<EmptyDirs>("dirs")
	.add_static_property("berkeley",make_getter(berkeley_dir),make_setter(berkeley_dir))
	.add_static_property("nyu",make_getter(nyu_dir),make_setter(nyu_dir))
	.add_static_property("weizmann",make_getter(weizmann_dir),make_setter(weizmann_dir))
	.add_static_property("voc",make_getter(voc_dir),make_setter(voc_dir));
}
