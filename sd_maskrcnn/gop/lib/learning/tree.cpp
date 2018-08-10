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
#include "tree.h"
#include "features.h"
#include "splitcriterion.h"
#include <random>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include "util/threading.h"
#include "util/algorithm.h"
#include "util/util.h"

// #define TIME
#define SHOW_PROGRESS

static float pickRandom( const VectorXf & f, int min_leaf ) {
	static std::mt19937 gen;
// 	float min = quickSelect( f, min_leaf ), max = quickSelect( f, f.size()-min_leaf-1 );
	float min = f.minCoeff(), max = f.maxCoeff();
	if( min < max ) {
		std::uniform_real_distribution<> dis(min,max);
		return dis( gen );
	}
	return 0.5*(min+max);
}
// For debugging purposes only
struct LeafStats {
	std::atomic<int> n_, n0_, n_zero_, n_pure_, n_pure0_, n_r_;
	LeafStats():n_(0),n0_(0),n_zero_(0),n_pure_(0),n_pure0_(0),n_r_(0){
	}
	void add( const RMatrixXf & lbl, int rep ) {
		RMatrixXf dlbl = (lbl.colwise() - lbl.col(0)).array().abs();
		const float EPS = 1e-2;
		bool pure = true, has_zero=false;
		for( int i=0; i<dlbl.rows(); i++ ) {
			if( dlbl.row(i).maxCoeff() < EPS ) {
				has_zero = true;
				n_zero_++;
			}
			if( (dlbl.row(0)-dlbl.row(i)).maxCoeff() > EPS )
				pure = false;
		}
		n_ += 1;
		n_r_ += (dlbl.row(rep).maxCoeff() < EPS);
		n0_ += has_zero;
		n_pure_ += pure;
		n_pure0_ += (has_zero && pure);
	}
	void print() {
		printf("Leaf Stats (%d leaves)\n", (int)n_ );
		printf("  pure     \t %0.1f%% [%d]\n", 100.*n_pure_ / n_, (int)n_pure_ );
		printf("  has zero \t %0.1f%% [%d]\n", 100.*n0_ / n_, (int)n0_ );
		printf("  rep zero \t %0.1f%% [%d]\n", 100.*n_pure0_ / n_, (int)n_pure0_ );
		printf("  all zero \t %0.1f%% [%d]\n", 100.*n_pure0_ / n_, (int)n_pure0_ );
		printf("  rep zero \t %0.1f%% [%d]\n", 100.*n_r_ / n_, (int)n_r_ );
		printf("  zeros    \t %d\n", (int)n_zero_ );
	}
};
struct FitData {
	VectorXi ids;
	int d, node_id;
	FitData( const VectorXi & ids=VectorXi(), int d=0, int node_id=0 ):ids(ids),d(d),node_id(node_id){}
	std::tuple<FitData,FitData> split( const VectorXb & is_left, int node_id ) const {
		const int n_left = is_left.cast<float>().array().sum(), n_right = ids.size() - n_left;
		VectorXi left( n_left ), right( n_right );
		for( int i=0,li=0,ri=0; i<(int)ids.size(); i++ )
			if( is_left[i] )
				left[li++] = ids[i];
			else
				right[ri++] = ids[i];
		return std::make_tuple( FitData{left,d+1,node_id}, FitData{right,d+1,node_id+1} );
	}
};
void BaseTree::remapFid( const VectorXi & fid ) {
	for( auto & n: nodes_ )
		if( n.fid >= 0 ) {
			if( n.fid >= fid.size() )
				throw std::invalid_argument( "Feature id map too small!" );
			n.fid = fid[ n.fid ];
		}
}
// Sample some random features (indexed 0..N-1)
VectorXi sampleFeatures( int N, TreeSettings::MaxFeature M) {
	VectorXi fids = arange(N);
	if( M != TreeSettings::ALL ) {
		int m = N;
		if( M == TreeSettings::SQRT )
			m = ceil(sqrt(N));
		if( M == TreeSettings::LOG2 )
			m = ceil(log2(N));
		if( M == TreeSettings::POW06 )
			m = ceil(pow(N,0.6));
		if( M == TreeSettings::POW07 )
			m = ceil(pow(N,0.7));
		fids = randomChoose( fids, m );
		std::sort( fids.data(), fids.data()+fids.size() );
	}
	return fids;
}
std::shared_ptr<SplitCriterion> makeSplit( TreeSettings settings ) {
	if( settings.criterion == TreeSettings::GINI )
		return giniSplit();
	else if( settings.criterion == TreeSettings::ENTROPY )
		return entropySplit();
	else if( settings.criterion == TreeSettings::STRUCT_GINI )
		return structGiniSplit( settings.n_structured_samples );
	else if( settings.criterion == TreeSettings::STRUCT_ENTROPY )
		return structEntropySplit( settings.n_structured_samples );
	else
		throw std::invalid_argument( "Invalid split criterion" );
}
void BaseTree::fitMT( const Features &f, const VectorXi & ids, const RMatrixXf & lbl, const VectorXf & weight, const TreeSettings &settings, const void * data ) {
	std::shared_ptr<SplitCriterion> split = makeSplit( settings );
	
	// Train the tree
	nodes_.resize(2*ids.size()-1);
	std::atomic<int> n_nodes(1);
#ifdef TIME
	ThreadedTimer t;
#define TIC t.tic()
#define TOC(x) t.toc((x))
#else
#define TIC
#define TOC(x)
#endif
	
#ifdef SHOW_PROGRESS
	UpdateProgressPrint progress("  * training tree ...   ", ids.size() );
#endif
	std::mutex m;
// 	LeafStats stat;
	auto fit_function = [&](ThreadedQueue<FitData>::Queue * queue, FitData d ){
		TIC;
		// Train the current node
		TreeNode & current_node = nodes_[d.node_id];
		const int N = (int)d.ids.size();
		TOC("init");
		
		// Collect the data
		RMatrixXf d_lbl( N, lbl.cols() );
		VectorXf d_weight( N );
		for( int i=0; i<N; i++ ) {
			d_lbl.row(i) = lbl.row( d.ids[i] );
			d_weight[i]  = weight[ d.ids[i] ];
		}
		std::shared_ptr<SplitCriterion> csplit = split->create( d_lbl, d_weight );
		TOC("split");
		
		float best_gain = 0;
		VectorXb best_split;
		int best_f=-1;
		float best_t = 0;
		// Find the best split
		if( N>1 && N >= settings.min_samples_split && N >= 2*settings.min_samples_leaf && d.d <= settings.max_depth && !csplit->is_pure()  ) {
			// Randomly sample features
			VectorXi fids = sampleFeatures( f.featureSize(), settings.max_feature );
			TOC("fid");
			
			// Evaluate those features and find the best split
// 			for( int fi: fids )
			for( int fii=0; fii<fids.size(); fii++ ) {
				const int & fi = fids[fii];
				// Extract the feature
				VectorXf fv( N );
				for( int j=0; j<N; j++ )
					fv[j] = f.get(d.ids[j],fi);
				
				// This feature wont work
				if( fv.maxCoeff() - fv.minCoeff() < 1e-4 )
					continue;
				
				float t, g;
				if( settings.extremely_random ) {
					t = pickRandom( fv, settings.min_samples_leaf );
					g = csplit->gain( fv.array() < t );
				}
				else
					t = csplit->bestThreshold( fv, &g );
				if( g > best_gain ) {
					best_gain = g;
					best_f = fi;
					best_t = t;
					best_split = fv.array() < t;
				}
			}
			TOC("find best");
		}
		// Split the data and continue
		if( best_gain > 0 && best_f >= 0 ) {
			const int n_left = best_split.cast<int>().sum();
			const int n_right = best_split.size() - n_left;
			
			if( n_left >= settings.min_samples_leaf && n_right >= settings.min_samples_leaf ) {
				int id = n_nodes.fetch_add(2);
				
				FitData left, right;
				std::tie(left,right) = d.split( best_split, id );
				
				current_node.fid = best_f;
				current_node.t = best_t;
				current_node.left_child = id;
				
				queue->push( left );
				queue->push( right );
			}
			TOC("spawn");
		}
		if( current_node.fid == -1 ) {
			std::lock_guard<std::mutex> lock( m );
			// Add a leaf node
			int rep_lbl = csplit->repLabel();
			if( rep_lbl>=0 )
				current_node.left_child = addLeaf( VectorXi::Constant(1,d.ids[rep_lbl]), d_weight, data );
			else
				current_node.left_child = addLeaf( d.ids, d_weight, data );
// 			stat.add( d_lbl, rep_lbl );
#ifdef SHOW_PROGRESS
			progress.updateDelta( d.ids.size() );
#endif
			TOC("leaf");
		}
	};
	ThreadedQueue<FitData> tq;
	tq.process( fit_function, {ids,1,0} );
	nodes_.resize(n_nodes);
// 	stat.print();
#undef TIC
#undef TOC
}

void BaseTree::fit( const Features &f, const VectorXi & ids, const RMatrixXf & lbl, const VectorXf & weight, const TreeSettings &settings, const void * data ) {
	std::shared_ptr<SplitCriterion> split = makeSplit( settings );
	
	// Train the tree
	nodes_.resize(2*ids.size()-1);
	int n_nodes=1;
	std::vector<FitData> queue;
	queue.push_back( {ids,1,0} );
	
#ifdef TIME
	Timer t;
#define TIC t.tic()
#define TOC(x) t.toc((x))
#else
#define TIC
#define TOC(x)
#endif
	
#ifdef SHOW_PROGRESS
	UpdateProgressPrint progress("  * Training Tree ... ", ids.size() );
#endif
	while(!queue.empty()) {
		TIC;
		
		// Train the current node
		FitData d = queue.back();
		queue.pop_back();
		TreeNode & current_node = nodes_[d.node_id];
		const int N = (int)d.ids.size();
		TOC("init");
		
		// Collect the data
		RMatrixXf d_lbl( N, lbl.cols() );
		VectorXf d_weight( N );
		for( int i=0; i<N; i++ ) {
			d_lbl.row(i) = lbl.row( d.ids[i] );
			d_weight[i]  = weight[ d.ids[i] ];
		}
		std::shared_ptr<SplitCriterion> csplit = split->create( d_lbl, d_weight );
		TOC("split");
		
		float best_gain = 0;
		VectorXb best_split;
		int best_f=-1;
		float best_t = 0;
		// Find the best split
		if( N>1 && N >= settings.min_samples_split && N >= 2*settings.min_samples_leaf && d.d <= settings.max_depth && !csplit->is_pure()  ) {
			// Randomly sample features
			VectorXi fids = sampleFeatures( f.featureSize(), settings.max_feature );
			TOC("fid");

			// Evaluate those features and find the best split
#pragma omp parallel for
			for( int i=0; i<(int)fids.size(); i++ ) {
				// Extract the feature
				int fi = fids[i];
				VectorXf fv( N );
				for( int j=0; j<N; j++ )
					fv[j] = f.get(d.ids[j],fi);
				
				// This feature wont work
				if( fv.maxCoeff() - fv.minCoeff() < 1e-4 )
					continue;
				
				float t, g;
				if( settings.extremely_random ) {
					t = pickRandom( fv, settings.min_samples_leaf );
					g = csplit->gain( fv.array() < t );
				}
				else
					t = csplit->bestThreshold( fv, &g );
#pragma omp critical(best_update)
				{
					if( g > best_gain ) {
						best_gain = g;
						best_f = fi;
						best_t = t;
						best_split = fv.array() < t;
					}
				}
			}
			TOC("find best");
		}
		// Split the data and continue
		if( best_gain > 0 && best_f >= 0 ) {
			const int n_left = best_split.cast<int>().sum();
			const int n_right = best_split.size() - n_left;
			
			if( n_left >= settings.min_samples_leaf && n_right >= settings.min_samples_leaf ) {
				int id = n_nodes;
				n_nodes += 2;
				
				FitData left, right;
				std::tie(left,right) = d.split( best_split, id );
				
				current_node.fid = best_f;
				current_node.t = best_t;
				current_node.left_child = id;
				
				queue.push_back( left );
				queue.push_back( right );
			}
			TOC("spawn");
		}
		if( current_node.fid == -1 ) {
			// Add a leaf node
			int rep_lbl = csplit->repLabel();
			if( rep_lbl>=0 )
				current_node.left_child = addLeaf( VectorXi::Constant(1,d.ids[rep_lbl]), d_weight, data );
			else
				current_node.left_child = addLeaf( d.ids, d_weight, data );
#ifdef SHOW_PROGRESS
			progress.updateDelta( d.ids.size() );
#endif
			TOC("leaf");
		}
	}
	nodes_.resize(n_nodes);
#undef TIC
#undef TOC
}
void BaseTree::refit( const Features &f, const VectorXi & ids, const RMatrixXf & lbl, const VectorXf & weight, const TreeSettings &settings, const void * data ) {
	std::shared_ptr<SplitCriterion> split = makeSplit( settings );
	
	// Retrain the tree
	
// #ifdef SHOW_PROGRESS
// 	UpdateProgressPrint progress("  * Retraining Tree ... ", ids.size() );
// #endif
	VectorXi id( ids.rows() );
#pragma omp parallel for
	for( int i=0; i<ids.rows(); i++ )
		id[i] = predict( f, ids[i] );
	
	// Compute which leaf each piece of data ends up in
	int n_leaf = 0;
	for( TreeNode n: nodes_ )
		if(n.fid==-1 && n_leaf <= n.left_child)
			n_leaf = n.left_child+1;
	
	std::vector< std::vector< int > > l_ids( n_leaf );
	for( int i=0; i<id.size(); i++ )
		l_ids[ id[i] ].push_back( i );
	
	// Find the leaf elements
	// TODO: This works only for rep labels!!!
	std::vector<VectorXi> leaf_ids( n_leaf );
	std::vector<VectorXf> leaf_weight( n_leaf );
#pragma omp parallel for
	for( int i=0; i<l_ids.size(); i++ ) {
		const int N = l_ids[i].size();
		if( N>0 ) {
			RMatrixXf d_lbl( N, lbl.cols() );
			VectorXf d_weight( N );
			for( int j=0; j<N; j++ ) {
				int k = l_ids[i][j];
				d_lbl.row(j) = lbl.row( k );
				d_weight[j]  = weight[ k ];
			}
			if( settings.use_single_leaf_label ) {
				std::shared_ptr<SplitCriterion> csplit = split->create( d_lbl, d_weight );
				leaf_ids[i] = VectorXi::Constant(1,l_ids[i][csplit->repLabel()]);
				leaf_weight[i] = VectorXf::Constant(1,1.0f);
			}
			else {
				leaf_ids[i] = VectorXi::Map( l_ids[i].data(), l_ids[i].size() );
				leaf_weight[i] = d_weight;
			}
		}
	}
	
	// Clear all the leaf data
	clearLeafData();
	
	// Merge leafs with no data
	for( int i=nodes_.size()-1; i>=0; i-- ) {
		if( nodes_[i].fid==-1 ) {
			if( leaf_ids[ nodes_[i].left_child ].size()==0 )
				nodes_[i].left_child = -1;
			else
				nodes_[i].left_child = addLeaf( leaf_ids[ nodes_[i].left_child ], leaf_weight[ nodes_[i].left_child ], data );
		}
		else {
			const int l = nodes_[i].left_child;
			for ( int k=0; k<2; k++ )
				if( nodes_[l+k].left_child == -1 ) {
					nodes_[i].left_child = nodes_[l+!k].left_child;
					nodes_[i].fid = -1;
				}
		}
	}
}
void BaseTree::clearLeafData() {
}
int BaseTree::addLeaf( const VectorXi & ids, const VectorXf & weight, const void * data ) {
	return -1;
}
int BaseTree::maxDepth() const {
	int max_d = 0;
	std::vector< std::tuple<int,int> > s;
	s.push_back( std::make_tuple(0,1) );
	while(!s.empty()) {
		int i,d;
		std::tie(i,d) = s.back();
		s.pop_back();
		if( d > max_d ) max_d = d;
		if( nodes_[i].fid != -1 ) {
			s.push_back( std::make_tuple(nodes_[i].left_child+0,d+1));
			s.push_back( std::make_tuple(nodes_[i].left_child+1,d+1));
		}
	}
	return max_d;
}
float BaseTree::averageDepth() const {
	double sum_d = 0, cnt = 0;
	std::vector< std::tuple<int,int> > s;
	s.push_back( std::make_tuple(0,1) );
	while(!s.empty()) {
		int i,d;
		std::tie(i,d) = s.back();
		s.pop_back();
		if( nodes_[i].fid != -1 ) {
			s.push_back( std::make_tuple(nodes_[i].left_child+0,d+1));
			s.push_back( std::make_tuple(nodes_[i].left_child+1,d+1));
		}
		else {
			sum_d += d;
			cnt += 1;
		}
	}
	return sum_d / cnt;
}
int BaseTree::predict( const Features & f, int fid ) const {
	int id = 0;
	while( nodes_[id].fid > -1 ) {
		if( f.get(fid,nodes_[id].fid) < nodes_[id].t )
			id = nodes_[id].left_child;
		else
			id = nodes_[id].left_child+1;
	}
	return nodes_[id].left_child;
}
VectorXi BaseTree::predict( const Features & f, const VectorXi & ids ) const {
	VectorXi r( ids.size() );
	for( int i=0; i<ids.size(); i++ )
		r[i] = predict( f, ids[i] );
	return r;
}
VectorXi BaseTree::predict( const Features & f ) const {
	return predict( f, arange(f.nSamples()) );
}
void BaseTree::save( std::ostream &s ) const {
#define put(x) s.write( (const char*)&x, sizeof(x) )
	int n = nodes_.size();
	put(n);
	for( const auto & i: nodes_ ) {
		put(i.fid);
		put(i.t);
		put(i.left_child);
	}
#undef put
}
void BaseTree::load( std::istream &s ) {
#define get(x) s.read( (char*)&x, sizeof(x) )
	int n;
	get(n);
	nodes_.resize(n);
	for( auto & i: nodes_ ) {
		get(i.fid);
		get(i.t);
		get(i.left_child);
	}
#undef get
}
VectorXf BinaryTree::predictProb( const Features & f, const VectorXi & ids ) const {
	VectorXi nds = predict( f, ids );
	VectorXf r = VectorXf::Zero( ids.size() );
	for( int i=0; i<nds.size(); i++ ) {
		r[ i ] = (data_[ nds[i] ].p_[1]+1e-10) / (data_[ nds[i] ].p_[0] + data_[ nds[i] ].p_[1]+2e-10 );
	}
	return r;
}
VectorXf BinaryTree::predictProb( const Features & f ) const {
	return predictProb( f, arange(f.nSamples()) );
}

void BinaryTree::setFromMatlab(const VectorXf &thrs, const VectorXi &child, const VectorXi &fid, const RMatrixXf &dist) {
	int N = dist.rows();
	for( int i=0; i<N; i++ ) {
		TreeNode n;
		n.fid = fid[i];
		n.left_child = child[i];
		n.t = thrs[i];
		if( n.left_child < 0  ) {
			n.fid = -1;
			n.left_child = data_.size();
			BinaryDistribution d;
			d.p_[0] = dist(i,0);
			d.p_[1] = dist(i,1);
			data_.push_back( d );
		}
		nodes_.push_back( n );
	}
}

void RangeTree::setFromMatlab(const VectorXf &thrs, const VectorXi &child, const VectorXi &fid, const VectorXi &rng) {
	const int N = thrs.size();
	for( int i=0; i<N; i++ ) {
		TreeNode n;
		n.fid = fid[i];
		n.left_child = child[i];
		n.t = thrs[i];
		if( n.left_child < 0  ) {
			n.fid = -1;
			n.left_child = data_.size();
			RangeData d;
			d.begin = rng[i];
			d.end = rng[i+1];
			data_.push_back( d );
		}
		nodes_.push_back( n );
	}
}
void PatchTree::setFromMatlab(const VectorXf &thrs, const VectorXi &child, const VectorXi &fid, const VectorXi &rng, const VectorXus & patch) {
	const int N = thrs.size();
	for( int i=0; i<N; i++ ) {
		TreeNode n;
		n.fid = fid[i];
		n.left_child = child[i];
		n.t = thrs[i];
		if( n.left_child < 0  ) {
			n.fid = -1;
			n.left_child = data_.size();
			PatchData d;
			for( int j=rng[i]; j<rng[i+1]; j++ )
				d.offsets_.push_back( patch[j] );
			data_.push_back( d );
		}
		nodes_.push_back( n );
	}
}
