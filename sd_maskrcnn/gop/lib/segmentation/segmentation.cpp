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
#include "segmentation.h"
#include "aggregation.h"
#include "util/util.h"
#include "util/algorithm.h"
#include "imgproc/filter.h"
#include "contour/sketchtokens.h"
#include "contour/structuredforest.h"
#include "contour/directedsobel.h"
#include <stdexcept>
#include <limits>
#include <queue>

Edges computeEdges( const RMatrixXs & seg ) {
	std::unordered_set<Edge> edges;
	for( int j=0; j<seg.rows(); j++ )
		for( int i=0; i<seg.cols(); i++ ) {
			if( i && seg(j,i-1) != seg(j,i) )
				edges.insert( Edge( seg(j,i-1), seg(j,i) ) );
			if( j && seg(j-1,i) != seg(j,i) ) 
				edges.insert( Edge( seg(j-1,i), seg(j,i) ) );
		}
	return Edges( edges.begin(), edges.end() );
}

/************ Geodesic K-Means ************/
class PlanarGeodesicDistanceBase {
public:
	virtual void compute( RMatrixXf & d, RMatrixXs & id, int NIT=2 ) const = 0;
	virtual float sumDist( int x, int y ) const = 0;
};

template<typename F1, typename F2, typename F3>
void floodFill( const RMatrixXs & id, F1 start, F2 visit, F3 end ) {
	std::vector< std::tuple<int,int> > q;
	RMatrixXb visited = RMatrixXb::Zero( id.rows(), id.cols() );
	for( int j=0; j<id.rows(); j++ )
		for( int i=0; i<id.cols(); i++ )
			if( !visited(j,i) ) {
				int cid = id(j,i);
				start( cid );
				// Do a DFS
				q.clear();
				q.push_back( std::make_tuple(i,j) );
				while(!q.empty()) {
					int x,y;
					std::tie(x,y) = q.back();
					q.pop_back();
					if( visited(y,x) )
						continue;
					visit( cid, x, y );
					visited(y,x) = 1;
					if( x && id(y,x-1) == cid && !visited(y,x-1) )
						q.push_back( std::make_tuple(x-1,y) );
					if( y && id(y-1,x) == cid && !visited(y-1,x) )
						q.push_back( std::make_tuple(x,y-1) );
					if( x+1<id.cols() && id(y,x+1) == cid && !visited(y,x+1) )
						q.push_back( std::make_tuple(x+1,y) );
					if( y+1<id.rows() && id(y+1,x) == cid && !visited(y+1,x) )
						q.push_back( std::make_tuple(x,y+1) );
				}
				end( cid );
			}
}
struct Point{
	int x, y;
};
static Point reseed( const Point & p, const PlanarGeodesicDistanceBase & dist, int W, int H, int R=2, float w = 1e-2 ) {
	const int x = p.x, y=p.y;
	float b = dist.sumDist( x, y );
	int bx=x, by=y;
	for( int j=std::max(0,y-R); j<H && j<=y+R; j++ )
		for( int i=std::max(0,x-R); i<W && i<=x+R; i++ ) {
			float bb = dist.sumDist( i, j )+w*sqrt((x-i)*(x-i)+(y-j)*(y-j));
			if( bb < b ){
				b = bb;
				bx = i;
				by = j;
			}
		}
	return {bx,by};
}
static RMatrixXs runKMeans( const PlanarGeodesicDistanceBase & dist, int W, int H, int approx_N, int NIT ) {
	RMatrixXf d = RMatrixXf::Zero( H, W );
	RMatrixXs id = RMatrixXs::Zero( H, W );
	
	// Fill the initial lattice structured centers
	int nx = ceil( sqrt( approx_N*W*sqrt(3.) / (2*H) ) );
	int ny = ceil( 1.0*approx_N / nx );
	std::vector< Point > centers;
	for( int j=0; j<ny; j++ )
		for( int i=0; i<nx; i++ )
			centers.push_back( reseed({(4*i+1+2*(j&1))*W / (4*nx), (2*j+1)*H / (2*ny) }, dist, W, H, 2 ) );
	for(int it=0; it<NIT; it++ ) {
		if( it ) {
			// Recompute the centers
			std::vector<float> sm_x(nx*ny,0), sm_y(nx*ny,0), sm(nx*ny,0);
			for( int j=0; j<H; j++ )
				for( int i=0; i<W; i++ ) {
					sm_x[ id(j,i) ] += i;
					sm_y[ id(j,i) ] += j;
					sm  [ id(j,i) ] += 1;
				}
			for( int i=0; i<nx*ny; i++ ) {
				if( sm[i] > 0 )
					centers[i] = reseed({(int)round(sm_x[i]/sm[i]), (int)round(sm_y[i]/sm[i])}, dist, W, H, 2);
// 				else
// 					printf("[%d] Superpixel collapsed! %d  %f\n", it, i, sm[i]);
			}
		}
		d.setConstant( std::numeric_limits<float>::infinity() );
		id.setConstant( -1 );
		int k=0;
		for( const auto & c: centers ) {
			int x = c.x, y = c.y;
			d(y,x) = 0;
			id(y,x) = k++;
		}
		dist.compute(d,id,4);
	}
	// Remap ids
	RMatrixXs new_id( H, W );
	std::vector< int > new_cnt,max_cnt(nx*ny,0);
	int cnt=0, nid=0;
	floodFill( id, [&]( int ){cnt=0;}, [&]( int,int x, int y ){cnt++;new_id(y,x)=nid;}, [&]( int id ){new_cnt.push_back(cnt);max_cnt[id]=std::max(max_cnt[id],cnt);nid++;} );
	for( int j=0; j<H; j++ )
		for( int i=0; i<W; i++ ) {
			int old_id = id(j,i);
			id(j,i) = new_id(j,i);
			if( new_cnt[ new_id(j,i) ] < max_cnt[ old_id ] ) {
				if( i )
					id(j,i) = id(j,i-1);
				else if( j )
					id(j,i) = id(j-1,i);
			}
		}
	// Remove empty ids
	std::vector<int> id_map( nid, -1 );
	nid = 0;
	for( int j=0; j<H; j++ )
		for( int i=0; i<W; i++ ) {
			if( id_map[ id(j,i) ] == -1 )
				id_map[ id(j,i) ] = nid++;
			id(j,i) = id_map[ id(j,i) ];
		}
	// Set the new ids
	return id;
}
class PlanarGeodesicDistance: public PlanarGeodesicDistanceBase {
protected:
	int W, H;
	const RMatrixXf & dx, & dy;
public:
	PlanarGeodesicDistance( const RMatrixXf & dx, const RMatrixXf & dy ):dx(dx),dy(dy){
		W = dy.cols();
		H = dx.rows();
		if( W != dx.cols()+1 || H != dy.rows()+1 )
			throw std::invalid_argument( "dx and dy shape does not match!" );
	}
	virtual void compute( RMatrixXf & d, RMatrixXs & id, int NIT=2 ) const {
#define CHECK_SET( c, ax, ay, bx, by, df ) if( (c) && d((by),(bx)) + (df) < d((ay),(ax)) ) { d((ay),(ax)) = d((by),(bx)) + (df); id((ay),(ax)) = id((by),(bx)); }
		for( int it=0; it<NIT; it++ ) {
			for( int j=0; j<H; j++ )
				for( int i=0; i<W; i++ ) {
					CHECK_SET( i, i, j, i-1, j, dx(j,i-1) );
					CHECK_SET( j, i, j, i, j-1, dy(j-1,i) );
				}
			for( int j=H-1; j>=0; j-- )
				for( int i=W-1; i>=0; i-- ) {
					CHECK_SET( i, i-1, j, i, j, dx(j,i-1) );
					CHECK_SET( j, i, j-1, i, j, dy(j-1,i) );
				}
		}
#undef CHECK_SET
	}
	virtual float sumDist( int x, int y ) const {
		return dx(y,std::min(x,W-2)) + dx(y,std::max(x-1,0)) + dy(std::min(y,H-2),x) + dy(std::max(y-1,0),x);
	}
};

static RMatrixXs geodesicKMeans( const RMatrixXf & dx, const RMatrixXf & dy, int approx_N, int NIT ) {
	const int W = dy.cols(), H = dx.rows();
	PlanarGeodesicDistance dist( dx, dy );
	return runKMeans( dist, W, H, approx_N, NIT );
}

/************ Over Segmentation ************/
OverSegmentation::OverSegmentation():Ns_(0){
}
OverSegmentation::OverSegmentation( const Edges & e ):edges_(e),edge_weights_(VectorXf::Zero(e.size())) {
	Ns_ = getN( e );
}
OverSegmentation::OverSegmentation( const Edges & e, const VectorXf & w ):edges_(e),edge_weights_(w) {
	Ns_ = getN( e );
}
OverSegmentation::~OverSegmentation(){
}
int OverSegmentation::Ns() const {
	return Ns_;
}
const Edges &OverSegmentation::edges() const {
	return edges_;
}
const VectorXf &OverSegmentation::edgeWeights() const {
	return edge_weights_;
}
void OverSegmentation::setEdgeWeights(const VectorXf &w) {
	edge_weights_ = w;
}
ImageOverSegmentation::ImageOverSegmentation(const Image8u & rgb_im, const RMatrixXs &s) : OverSegmentation(computeEdges(s)), rgb_im_(rgb_im), s_(s){
	if( s_.rows()!=rgb_im.H() || s_.cols() != rgb_im.W() )
		throw std::invalid_argument("Image and segmentation shape do not match!");
}
ImageOverSegmentation::ImageOverSegmentation(){
}
const RMatrixXs &ImageOverSegmentation::s() const {
	return s_;
}
const Image8u &ImageOverSegmentation::image() const {
	return rgb_im_;
}
RMatrixXf ImageOverSegmentation::boundaryMap(bool thin) const {
	std::unordered_map< Edge, int > edge_id;
	for( int i=0; i<edges_.size(); i++ )
		edge_id[ edges_[i] ] = i+1;
	
	RMatrixXf r = RMatrixXf::Zero( s_.rows(), s_.cols() );
	for( int j=0; j<s_.rows(); j++ )
		for( int i=0; i<s_.cols(); i++ ) {
			if( i && s_(j,i) != s_(j,i-1) ) {
				int id = edge_id[ Edge(s_(j,i),s_(j,i-1)) ]-1;
				if( id >= 0 ) {
					r(j,i  ) = std::max( r(j,i  ), edge_weights_[id] );
					if( !thin )
						r(j,i-1) = std::max( r(j,i-1), edge_weights_[id] );
				}
			}
			if( j && s_(j,i) != s_(j-1,i) ){
				int id = edge_id[ Edge(s_(j,i),s_(j-1,i)) ]-1;
				if( id >= 0 ) {
					r(j  ,i) = std::max( r(j  ,i), edge_weights_[id] );
					if( !thin )
						r(j-1,i) = std::max( r(j-1,i), edge_weights_[id] );
				}
			}
		}
	return r;
}
static std::shared_ptr<ImageOverSegmentation> computeGeodesicKMeans( const Image8u & im, const RMatrixXf & dx, const RMatrixXf & dy, int approx_N, int NIT ) {
	return std::make_shared<ImageOverSegmentation>( im, geodesicKMeans(dx, dy, approx_N, NIT) );
}
std::shared_ptr<ImageOverSegmentation> geodesicKMeans( const Image8u & im, const BoundaryDetector & detector, int approx_N ) {
	return geodesicKMeans( im, detector, approx_N, 1 );
}
std::shared_ptr<ImageOverSegmentation> geodesicKMeans( const Image8u & im, const BoundaryDetector & detector, int approx_N, int NIT ) {
	RMatrixXf thin_bnd = detector.detectAndFilter( im );
	
	const int W = im.W(), H = im.H();
	RMatrixXf dx = thin_bnd.leftCols(W-1).array().max( thin_bnd.rightCols(W-1).array() ) + 1e-2;
	RMatrixXf dy = thin_bnd.topRows(H-1).array().max( thin_bnd.bottomRows(H-1).array() ) + 1e-2;
	std::shared_ptr<ImageOverSegmentation> r = computeGeodesicKMeans(im, dx, dy, approx_N, NIT);
	
	percentileFilter( thin_bnd.data(), thin_bnd.data(), W, H, 1, 1, 1 );
	r->setEdgeWeights( r->projectBoundary( thin_bnd, "p65" ) );
	return r;
}
std::shared_ptr<ImageOverSegmentation> geodesicKMeans( const Image8u & im, const SketchTokens & detector, int approx_N ) {
	return geodesicKMeans( im, detector, approx_N, 1 );
}
std::shared_ptr<ImageOverSegmentation> geodesicKMeans( const Image8u & im, const SketchTokens & detector, int approx_N, int NIT ) {
	RMatrixXf thick_bnd = detector.detect( im );
	// Filter without suppression
	RMatrixXf thin_bnd = detector.filter( thick_bnd, 0 );
	
	RMatrixXf os_bnd = thin_bnd.array() + 1e-2*thick_bnd.array() + 1e-2;
	
	const int W = im.W(), H = im.H();
	RMatrixXf dx = os_bnd.leftCols(W-1).array().max( os_bnd.rightCols(W-1).array() );
	RMatrixXf dy = os_bnd.topRows(H-1).array().max( os_bnd.bottomRows(H-1).array() );
	std::shared_ptr<ImageOverSegmentation> r = computeGeodesicKMeans(im, dx, dy, approx_N, NIT);
	
	r->setEdgeWeights( r->projectBoundary( thick_bnd, "p25" ) );
	return r;
}
std::shared_ptr<ImageOverSegmentation> geodesicKMeans( const Image8u & im, const StructuredForest & detector, int approx_N ) {
	return geodesicKMeans( im, detector, approx_N, 1 );
}
std::shared_ptr<ImageOverSegmentation> geodesicKMeans( const Image8u & im, const StructuredForest & detector, int approx_N, int NIT ) {
	RMatrixXf thick_bnd = detector.detect( im );
	// Filter without suppression
	RMatrixXf thin_bnd = detector.filter( thick_bnd, 0 );
	
	RMatrixXf os_bnd = thin_bnd.array() + 1e-2*thick_bnd.array() + 1e-2;
	
	const int W = im.W(), H = im.H();
	RMatrixXf dx = os_bnd.leftCols(W-1).array().max( os_bnd.rightCols(W-1).array() );
	RMatrixXf dy = os_bnd.topRows(H-1).array().max( os_bnd.bottomRows(H-1).array() );
	
	std::shared_ptr<ImageOverSegmentation> r = computeGeodesicKMeans(im, dx, dy, approx_N, NIT);
	r->setEdgeWeights( r->projectBoundary( thick_bnd, "p25" ) );
	
	return r;
}
std::shared_ptr<ImageOverSegmentation> geodesicKMeans( const Image8u & im, const RMatrixXf & thick_bnd, const RMatrixXf & thin_bnd, int approx_N ) {
	return geodesicKMeans( im, thick_bnd, thin_bnd, approx_N, 1 );
}
std::shared_ptr<ImageOverSegmentation> geodesicKMeans( const Image8u & im, const RMatrixXf & thick_bnd, const RMatrixXf & thin_bnd, int approx_N, int NIT ) {
	RMatrixXf os_bnd = thin_bnd.array() + 1e-2*thick_bnd.array() + 1e-2;
	
	const int W = im.W(), H = im.H();
	RMatrixXf dx = os_bnd.leftCols(W-1).array().max( os_bnd.rightCols(W-1).array() );
	RMatrixXf dy = os_bnd.topRows(H-1).array().max( os_bnd.bottomRows(H-1).array() );
	
	std::shared_ptr<ImageOverSegmentation> r = computeGeodesicKMeans(im, dx, dy, approx_N, NIT);
	r->setEdgeWeights( r->projectBoundary( thick_bnd, "p25" ) );
	return r;
}
std::shared_ptr<ImageOverSegmentation> geodesicKMeans( const Image8u & im, const DirectedSobel & detector, int approx_N ) {
	return geodesicKMeans( im, detector, approx_N, 1 );
}
std::shared_ptr<ImageOverSegmentation> geodesicKMeans( const Image8u & im, const DirectedSobel & detector, int approx_N, int NIT ) {
	RMatrixXf dx, dy;
	std::tie( dx, dy ) = detector.detectXY( im, false, true );
	
	// Let a spix grow approx 20x average size until we hit a spatial penalty of 1
	const float sx = 0.05*sqrt(1.0*approx_N/(im.W()*im.H()));
	std::shared_ptr<ImageOverSegmentation> r = computeGeodesicKMeans(im, dx.array()+sx, dy.array()+sx, approx_N, NIT);
	
	std::tie( dx, dy ) = detector.detectXY( im, false, false );
	r->setEdgeWeights( r->projectBoundary( dx, dy, "med" ).array().pow(0.8).matrix() );
	return r;
}
void OverSegmentation::save(std::ostream &s) const {
	saveEdges( s, edges_ );
	saveMatrixX( s, edge_weights_ );
}
void OverSegmentation::load(std::istream &s) {
	loadEdges( s, edges_ );
	Ns_ = getN(edges_);
	loadMatrixX( s, edge_weights_ );
}
void ImageOverSegmentation::save(std::ostream &s) const {
	OverSegmentation::save( s );
	saveMatrixX(s, s_ );
	rgb_im_.save( s );
}
void ImageOverSegmentation::load(std::istream &s) {
	OverSegmentation::load( s );
	loadMatrixX(s, s_ );
	rgb_im_.load( s );
}
VectorXs ImageOverSegmentation::projectSegmentation(const RMatrixXs &seg, bool conservative) const {
	if( s_.rows() != seg.rows() || s_.cols() != seg.cols() )
		throw std::invalid_argument( "Image and segmentation size do not match!" );
	
	const int N = seg.rows()*seg.cols();
	
	VectorXs r = -VectorXs::Ones( Ns_ );
	int nseg = seg.maxCoeff()+1;
	// Start voting
	RMatrixXi vote = RMatrixXi::Zero( Ns_, nseg );
	VectorXb neg = VectorXb::Zero( Ns_ );
	for( int i=0; i<N; i++ )
		if( seg.data()[i] >= 0 )
			vote( s_.data()[i], seg.data()[i] ) += 1;
		else
			neg( s_.data()[i] ) = 1;
		
	// Return the highest voted segment
	for( int i=0; i<Ns_; i++ ) {
		int m;
		if( vote.row(i).maxCoeff( &m ) > 0 ) {
			r[i] = m;
			if( conservative && neg[i] )
				r[i] = -1;
		}
	}
	return r;
}
VectorXf ImageOverSegmentation::project(const RMatrixXf &data, const std::string &type) const {
	eassert( s_.cols() == data.cols() && s_.rows() == data.rows() );
	
	std::vector< std::shared_ptr<AggregationFunction> > accum_buf( Ns_ );
	accum_buf[0] = AggregationFunction::create( type );
	for( int i=1; i<Ns_; i++ )
		accum_buf[i] = accum_buf[0]->clone();
	
	for( int i=0; i<s_.rows()*s_.cols(); i++ )
		accum_buf[ s_.data()[i] ]->add( data.data()[i] );
	VectorXf r( Ns_ );
	for( int i=0; i<Ns_; i++ )
		r.data()[i] = accum_buf[i]->get();
	return r;
}
RMatrixXf ImageOverSegmentation::project(const Image &data, const std::string &type) const {
	eassert( s_.cols() == data.W() && s_.rows() == data.H() );
	const int C = data.C();
	std::vector< std::shared_ptr<AggregationFunction> > accum_buf( Ns_*C );
	accum_buf[0] = AggregationFunction::create( type );
	for( int i=1; i<Ns_*C; i++ )
		accum_buf[i] = accum_buf[0]->clone();
	
	for( int j=0; j<s_.rows(); j++ )
		for( int i=0; i<s_.cols(); i++ )
			for( int k=0; k<C; k++ )
				accum_buf[ s_(j,i)*C+k ]->add( data(j,i,k) );
	RMatrixXf r( Ns_, C );
	for( int i=0; i<Ns_*C; i++ )
		r.data()[i] = accum_buf[i]->get();
	return r;
}
VectorXf ImageOverSegmentation::projectBoundary( const RMatrixXf & im, const std::string & type ) const {
	if( (s_.cols() != im.cols() || s_.rows() != im.rows() ) )
		throw std::invalid_argument( "Segmentation and image shape need to match!" );
	
	// Assign them an id
	std::unordered_map<Edge,int> edge_id;
	for( int i=0; i<edges_.size(); i++ )
		edge_id[ edges_[i] ] = i+1;
	
	// Upsample the edges
	std::vector< std::shared_ptr<AggregationFunction> > accum_buf( edges_.size() );
	accum_buf[0] = AggregationFunction::create( type );
	for( int i=1; i<edges_.size(); i++ )
		accum_buf[i] = accum_buf[0]->clone();
	
	for( int j=0; j<s_.rows(); j++ )
		for( int i=0; i<s_.cols(); i++ ) {
			if( i && s_(j,i) != s_(j,i-1) ) {
				int id = edge_id[ Edge(s_(j,i),s_(j,i-1)) ]-1;
				if( id >= 0 ) {
					accum_buf[id]->add( im(j,i  ) );
					accum_buf[id]->add( im(j,i-1) );
				}
			}
			if( j && s_(j,i) != s_(j-1,i) ) {
				int id = edge_id[ Edge(s_(j,i),s_(j-1,i)) ]-1;
				if( id >= 0 ) {
					accum_buf[id]->add( im(j  ,i) );
					accum_buf[id]->add( im(j-1,i) );
				}
			}
		}
	
	// Create the result
	VectorXf r( edges_.size() );
	for( int i=0; i<edges_.size(); i++ )
		r[i] = accum_buf[i]->get();
	return r;
}
VectorXf ImageOverSegmentation::projectBoundary( const RMatrixXf & dx, const RMatrixXf & dy, const std::string & type ) const {
	if( s_.cols() != dx.cols()+1 || s_.rows() != dx.rows() || s_.cols() != dy.cols() || s_.rows() != dy.rows()+1 )
		throw std::invalid_argument( "Segmentation and image shape need to match!" );
	
	// Assign them an id
	std::unordered_map<Edge,int> edge_id;
	for( int i=0; i<edges_.size(); i++ )
		edge_id[ edges_[i] ] = i+1;
	
	// Upsample the edges
	std::vector< std::shared_ptr<AggregationFunction> > accum_buf( edges_.size() );
	accum_buf[0] = AggregationFunction::create( type );
	for( int i=1; i<edges_.size(); i++ )
		accum_buf[i] = accum_buf[0]->clone();
	
	for( int j=0; j<s_.rows(); j++ )
		for( int i=0; i<s_.cols(); i++ ) {
			if( i && s_(j,i) != s_(j,i-1) ) {
				int id = edge_id[ Edge(s_(j,i),s_(j,i-1)) ]-1;
				if( id >= 0 )
					accum_buf[id]->add( dx(j,i-1) );
			}
			if( j && s_(j,i) != s_(j-1,i) ) {
				int id = edge_id[ Edge(s_(j,i),s_(j-1,i)) ]-1;
				if( id >= 0 )
					accum_buf[id]->add( dy(j-1,i) );
			}
		}
	
	// Create the result
	VectorXf r( edges_.size() );
	for( int i=0; i<edges_.size(); i++ )
		r[i] = accum_buf[i]->get();
	return r;
}
RMatrixXi ImageOverSegmentation::maskToBox( const RMatrixXb &masks ) const {
	return ::maskToBox( s_, masks );
}
RMatrixXi maskToBox( const RMatrixXs &s, const RMatrixXb &masks ) {
	const int Ns = s.maxCoeff()+1;
	if ( masks.cols() != Ns )
		throw std::invalid_argument( "Mask and segmentation have different size!" );
	
	RMatrixXi seg_box( Ns, 4 );
	for( int i=0; i<Ns; i++ ) {
		seg_box(i,0) = seg_box(i,1) = std::max(s.cols(), s.rows());
		seg_box(i,2) = seg_box(i,3) = 0;
	}
	for( int j=0; j<s.rows(); j++ )
		for( int i=0; i<s.cols(); i++ ) {
			int l = s(j,i);
			if( 0 <= l && l < Ns ) {
				seg_box(l,0) = std::min( i, seg_box(l,0) );
				seg_box(l,1) = std::min( j, seg_box(l,1) );
				seg_box(l,2) = std::max( i, seg_box(l,2) );
				seg_box(l,3) = std::max( j, seg_box(l,3) );
			}
		}
	
	RMatrixXi boxes(masks.rows(),4);
	for( int i=0; i<masks.rows(); i++ ) {
		boxes(i,0) = boxes(i,1) = std::max(s.cols(), s.rows());
		boxes(i,2) = boxes(i,3) = 0;
		for( int j=0; j<Ns; j++ ) 
			if( masks(i,j) ){
				boxes(i,0) = std::min( seg_box(j,0), boxes(i,0) );
				boxes(i,1) = std::min( seg_box(j,1), boxes(i,1) );
				boxes(i,2) = std::max( seg_box(j,2), boxes(i,2) );
				boxes(i,3) = std::max( seg_box(j,3), boxes(i,3) );
			}
	}
	return boxes;
}
