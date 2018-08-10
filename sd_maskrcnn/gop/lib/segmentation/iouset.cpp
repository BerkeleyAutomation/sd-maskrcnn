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
#include "iouset.h"
#include "segmentation.h"
#include <queue>

IOUSet::IOUSet( const OverSegmentation &os ) {
	try {
		initImage( dynamic_cast<const ImageOverSegmentation&>( os ) );
	}
	catch( std::bad_cast ) {
		init( os );
	}
}

void IOUSet::initImage(const ImageOverSegmentation &os) {
	const int N = os.Ns();
	const RMatrixXs & s = os.s();
	// Compute the x and y position and id
	std::vector< Vector3f > pos( N, Vector3f::Zero() );
	for( int j=0; j<s.rows(); j++ )
		for( int i=0; i<s.cols(); i++ )
			pos[ s(j,i) ] += Vector3f( i, j, 1 );
	for( int i=0; i<pos.size(); i++ ) {
		pos[i] /= pos[i][2] + 1e-10;
		pos[i][2] = i;
	}
	// Compute the kd-tree
	std::queue< std::tuple<int,int,int> > q;
	q.push( std::make_tuple( 0, N, -1 ) );
	
	int nid = 2*N-1;
	parent_.resize( nid, -1 );
	left_.resize( nid, -1 );
	right_.resize( nid, -1 );
	while( !q.empty() ) {
		int a, b, pid;
		std::tie(a,b,pid) = q.front();
		q.pop();
		
		// Leaf node
		if( a+1>=b ) {
			parent_[pos[a][2]] = pid;
			if( left_[pid] == -1 )
				left_[pid] = pos[a][2];
			else if( right_[pid] == -1 )
				right_[pid] = pos[a][2];
		}
		else {
			// Build the graph [add the node]
			int id = --nid;
			if( id < N )
				printf("EVIL %d < %d!  [%d %d %d]\n", id, N, a, b, pid);
			parent_[id] = pid;
			if( pid >=0 && left_[pid] == -1 )
				left_[pid] = id;
			else if( pid >=0 && right_[pid] == -1 )
				right_[pid] = id;
			
			// Find a split point [x or y]
			int best_d = 0;
			float dist = 0, split=0;
			for( int d=0; d<2; d++ ) {
				float x0 = pos[a][d], x1 = pos[a][d];
				float sx = 0, ct = 0;
				for( int i=a; i<b; i++ ) {
					x0 = std::min( x0, pos[i][d] );
					x1 = std::max( x1, pos[i][d] );
					sx += pos[i][d];
					ct += 1;
				}
				if (x1-x0 >= dist) {
					dist = x1-x0;
					split = sx / ct;
					best_d = d;
				}
			}
			// Split
			int s = a, e = b-1;
			while( s<e ) {
				while( s < b && pos[s][best_d] < split ) s++;
				while( e >= a && pos[e][best_d] >= split ) e--;
				if( s<e )
					std::swap( pos[s], pos[e] );
			}
			if( s==a )
				s++;
			// Add to q
			q.push( std::make_tuple(a,s,id) );
			q.push( std::make_tuple(s,b,id) );
		}
	}
	cnt_ = VectorXi::Zero( 2*N-1 );
	for( int i=0; i<2*N-1; i++ ) {
		cnt_[i] += (i<N);
		if( parent_[i] >= 0 )
			cnt_[parent_[i]] += cnt_[i];
	}
}
namespace IOUPrivate {
	struct WeightedEdge: public Edge {
		float w_;
		WeightedEdge( int a=0, int b=0, float w=0.f ):Edge{a,b},w_(w){
		}
		bool operator>( const WeightedEdge & o ) const {
			return w_ > o.w_;
		}
	};
	struct Node {
		int id;
		float w;
		Node( int id=0, float w=0.f ):id(id),w(w){
		}
		bool operator>( const Node & o ) const {
			return w > o.w;
		}
	};
}

void IOUSet::init(const OverSegmentation &os) {
	using namespace IOUPrivate;
	const float size_weight = 0., depth_weight = 0.1;
	int Ns = os.Ns();
	const Edges & e = os.edges();
	const VectorXf & w = os.edgeWeights();
	std::vector<int> depth(2*Ns-1,0);
	
	std::priority_queue<WeightedEdge, std::vector<WeightedEdge>, std::greater<WeightedEdge> > q;
	for( int i=0; i<e.size(); i++ )
		q.push( {e[i].a,e[i].b,w[i]} );
	
	parent_.resize( 2*Ns-1, -1 );
	left_.resize( 2*Ns-1, -1 );
	right_.resize( 2*Ns-1, -1 );
	cnt_ = VectorXi::Zero( 2*Ns-1 );
	cnt_.head(Ns).setOnes();
	int k=0;
	for( k=Ns; k<2*Ns-1 && !q.empty(); ) {
		WeightedEdge p = q.top();
		q.pop();
		// One of the nodes got merged already (update the edge)
		if( parent_[p.a]>=0 || parent_[p.b]>=0 ) {
			int a = parent_[p.a]>=0 ? parent_[p.a] : p.a;
			int b = parent_[p.b]>=0 ? parent_[p.b] : p.b;
			if( a == b )
				continue;
			int d  = std::max( depth[a], depth[b] ), s = cnt_[a] + cnt_[b];
			int od = std::max( depth[p.a], depth[p.b] ), os = cnt_[p.a] + cnt_[p.b];
			float w = p.w_ + (d-od)*depth_weight + (s-os)*size_weight;
			q.push( {a,b,w} );
			continue;
		}
		// Merge the edges
		parent_[p.a] = k;
		parent_[p.b] = k;
		left_[k] = p.a;
		right_[k] = p.b;
		depth[k] = std::max(depth[p.a],depth[p.b])+1;
		cnt_[k] = cnt_[p.a] + cnt_[p.b];
		k++;
	}
	if( k<2*Ns-1 ) {
		std::priority_queue<Node, std::vector<Node>, std::greater<Node> > q2;
		for( int i=0; i<k; i++ )
			if( parent_[i] == -1 )
				q2.push( Node( i, depth[i] ) );
		for( ; k<2*Ns-1 && q2.size()>1; ) {
			Node n1 = q2.top();
			q2.pop();
			Node n2 = q2.top();
			q2.pop();
			while ( parent_[n1.id] != -1 && !q2.empty() ) {
				n1 = q2.top();
				q2.pop();
			}
			while ( parent_[n2.id] != -1 && !q2.empty() ) {
				n2 = q2.top();
				q2.pop();
			}
			if( parent_[n1.id] == -1 && parent_[n2.id] == -1 ) {
				parent_[n1.id] = k;
				parent_[n2.id] = k;
				left_[k]  = n1.id;
				right_[k] = n2.id;
				depth[k]  = std::max(depth[n1.id],depth[n2.id])+1;
				cnt_[k]   = cnt_[n1.id] + cnt_[n2.id];
				q2.push( Node( k, depth[k] ) );
				k++;
			}
		}
	}
	if( k!=2*Ns-1 )
		throw std::invalid_argument( "Great Evil, the graph is not connected and we coundn't recover a IOUSet from it" );
}
VectorXi IOUSet::computeTree(const VectorXb & s) const {
	VectorXi r = VectorXi::Zero(parent_.size());
	std::copy( s.data(), s.data()+s.size(), r.data() );
	
	for( int i=0; i<r.size(); i++ )
		if( parent_[i] >= 0 )
			r[ parent_[i] ] += r[i];
	return r;
}
void IOUSet::addTree(const VectorXi &v) {
	set_.push_back( v );
}
bool IOUSet::intersectsTree(const VectorXi &v, float max_iou) const {
	for( const VectorXi & i: set_ )
		if( cmpIOU(v,i,max_iou) )
			return true;
	return false;
}
bool IOUSet::intersectsTree(const VectorXi &v, const VectorXf & iou_list) const {
	const int N = iou_list.size();
	VectorXi max_int( N );
	for( int i=0; i<N; i++ )
		max_int[i] = N-i;
	
	for( const VectorXi & i: set_ ) {
		for( int j=0; j<N; j++ )
			if( cmpIOU(v,i,iou_list[j]) ) {
				if( --max_int[j] <= 0 )
					return true;
			}
			else
				break;
	}
	return false;
}
float IOUSet::maxIOUTree( const VectorXi & v ) const {
	if( set_.size() <= 1 )
		return 0;
	float a=0,b=1;
	VectorXi good = VectorXi::Ones( set_.size() );
	for( int it=0; it<10; it++ ) {
		VectorXi new_good = 1*good;
		float iou = (a+b)/2;
		bool inter = false;
		for( int i=0; i<set_.size(); i++ )
			if( good[i] )
				if( (new_good[i] = cmpIOU(v,set_[i],iou)) )
					inter = true;
		if( inter ) {
			good = new_good;
			a = iou;
		}
		else
			b = iou;
	}
	if( a == 0 )
		return 0;
	if( b == 1 )
		return 1;
	return (a+b)/2;
}
void IOUSet::add(const VectorXb &s) {
	addTree( computeTree( s ) );
}
bool IOUSet::intersects(const VectorXb &s, float max_iou) const {
	return intersectsTree( computeTree(s), max_iou );
}
bool IOUSet::intersects(const VectorXb &s, const VectorXf & iou) const {
	return intersectsTree( computeTree(s), iou );
}
float IOUSet::maxIOU( const VectorXb & s ) const {
	return maxIOUTree( computeTree(s) );
}
bool IOUSet::cmpIOU(const VectorXi &a, const VectorXi &b, float max_iou) const {
	assert( a.size() == b.size() );
	const int N = a.size();
	const float t = max_iou / (1 + max_iou);
	const float t_inter = t*(a[N-1]+b[N-1]);
	
	float upper = std::min(a[N-1], b[N-1]), lower = std::max(a[N-1]+b[N-1]-cnt_[N-1],0);
	if( lower >= t_inter ) return true;
	if( upper <= t_inter ) return false;
	std::queue<int> q;
	q.push( N-1 );
	while(!q.empty()) {
		int n = q.front();
		q.pop();
		// Split the current node
		float c_lower = std::max(a[n]+b[n]-cnt_[n],0), c_upper = std::min(a[n],b[n]);
		float left_lower = std::max(a[left_[n]]+b[left_[n]]-cnt_[left_[n]],0), left_upper = std::min(a[left_[n]],b[left_[n]]);
		float right_lower = std::max(a[right_[n]]+b[right_[n]]-cnt_[right_[n]],0), right_upper = std::min(a[right_[n]],b[right_[n]]);
		lower += left_lower+right_lower-c_lower;
		upper += left_upper+right_upper-c_upper;
		if( lower >= t_inter ) return true;
		if( upper <= t_inter ) return false;
		if( left_lower < left_upper && left_[left_[n]] != -1 )
			q.push( left_[n] );
		if( right_lower < right_upper && right_[right_[n]] != -1 )
			q.push( right_[n] );
	}
	printf("l:%f  u:%f\n", lower, upper );
	return true;
}

