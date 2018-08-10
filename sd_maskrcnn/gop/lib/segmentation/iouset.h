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
#include <vector>
#include "util/eigen.h"
#include "util/graph.h"

class OverSegmentation;
class ImageOverSegmentation;
class IOUSet {
protected:
	std::vector<int> parent_, left_, right_;
	VectorXi cnt_;
	std::vector<VectorXi> set_;
	VectorXi computeTree( const VectorXb & p ) const;
	void addTree( const VectorXi & v );
	bool cmpIOU( const VectorXi & a, const VectorXi & b, float max_iou ) const;
	bool intersectsTree( const VectorXi & v, float max_iou ) const;
	bool intersectsTree( const VectorXi & v, const VectorXf & iou_list ) const;
	float maxIOUTree( const VectorXi & v ) const;
	void init( const OverSegmentation & os );
	void initImage( const ImageOverSegmentation & os );
public:
	IOUSet( const OverSegmentation & os );
	void add( const VectorXb & p );
	bool intersects( const VectorXb & p, float max_iou ) const;
	bool intersects( const VectorXb & p, const VectorXf & iou ) const;
	float maxIOU( const VectorXb & p ) const;
};
