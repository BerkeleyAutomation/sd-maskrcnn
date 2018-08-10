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
#include "imgproc/image.h"

Edges computeEdges( const RMatrixXs & seg );

class BoundaryDetector;
class DirectedSobel;
class SketchTokens;
class StructuredForest;

class OverSegmentation {
protected:
	int Ns_;
	// Graph Structure
	Edges edges_;
	VectorXf edge_weights_;
public:
	OverSegmentation();
	OverSegmentation( const Edges & e );
	OverSegmentation( const Edges & e, const VectorXf & w );
	virtual ~OverSegmentation();
	int Ns() const;
	const Edges & edges() const;
	const VectorXf & edgeWeights() const;
	void setEdgeWeights( const VectorXf & w );
	
	virtual void save( std::ostream & s ) const;
	virtual void load( std::istream & s );
};
RMatrixXi maskToBox( const RMatrixXs & s, const RMatrixXb & masks );
class ImageOverSegmentation: public OverSegmentation {
protected:
	// Over segmentation segment ids
	Image8u rgb_im_;
	RMatrixXs s_;
	// Protected contructor (don't create this object)
public:
	ImageOverSegmentation();
	ImageOverSegmentation( const Image8u & im, const RMatrixXs & s );
	const RMatrixXs & s() const;
	const Image8u & image() const;
	RMatrixXf boundaryMap( bool thin=false ) const;
	
	VectorXs projectSegmentation( const RMatrixXs & seg, bool conservative=false ) const;
	VectorXf project( const RMatrixXf & data, const std::string & type ) const;
	RMatrixXf project( const Image & data, const std::string & type ) const;
	
	VectorXf projectBoundary( const RMatrixXf & im, const std::string & type ) const;
	VectorXf projectBoundary( const RMatrixXf & dx, const RMatrixXf & dy, const std::string & type ) const;
// 	RMatrixXf colorHistogram(const Ref<const RMatrixXs> &s, const float * im, int W, int H, int C, int N_BIN=5);
	
	RMatrixXi maskToBox( const RMatrixXb & masks ) const;
	
	virtual void save( std::ostream & s ) const;
	virtual void load( std::istream & s );
};

std::shared_ptr<ImageOverSegmentation> geodesicKMeans( const Image8u & im, const BoundaryDetector & detector, int approx_N );
std::shared_ptr<ImageOverSegmentation> geodesicKMeans( const Image8u & im, const BoundaryDetector & detector, int approx_N, int NIT );
std::shared_ptr<ImageOverSegmentation> geodesicKMeans( const Image8u & im, const SketchTokens & detector, int approx_N );
std::shared_ptr<ImageOverSegmentation> geodesicKMeans( const Image8u & im, const SketchTokens & detector, int approx_N, int NIT );
std::shared_ptr<ImageOverSegmentation> geodesicKMeans( const Image8u & im, const StructuredForest & detector, int approx_N );
std::shared_ptr<ImageOverSegmentation> geodesicKMeans( const Image8u & im, const StructuredForest & detector, int approx_N, int NIT );
std::shared_ptr<ImageOverSegmentation> geodesicKMeans( const Image8u & im, const RMatrixXf & thick_bnd, const RMatrixXf & thin_bnd, int approx_N );
std::shared_ptr<ImageOverSegmentation> geodesicKMeans( const Image8u & im, const RMatrixXf & thick_bnd, const RMatrixXf & thin_bnd, int approx_N, int NIT );
std::shared_ptr<ImageOverSegmentation> geodesicKMeans( const Image8u & im, const DirectedSobel & detector, int approx_N );
std::shared_ptr<ImageOverSegmentation> geodesicKMeans( const Image8u & im, const DirectedSobel & detector, int approx_N, int NIT );
