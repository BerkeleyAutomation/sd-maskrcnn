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
#include "edgeweight.h"
#include "geodesics.h"
#include "segmentation/segmentation.h"
#include "util/optimization.h"
#include <iostream>

VectorXi8 makeLbl( const VectorXs &pseg, int cls ) {
	return 2*(pseg.array() == cls).cast<int8_t>() - 1*(pseg.array() >= -1).cast<int8_t>();
}

EdgeWeight EdgeWeight::makeDefault() {
	EdgeWeight r;
	r.w_ = VectorXf::Ones( 1 );
	r.f_.addWeighted( 1.0, 3.0, 2e-3 );
	return r;
}
EdgeWeight EdgeWeight::makeAll() {
	EdgeWeight r;
	r.f_.addWeighted( 0.0, 1.0, 1.0 );
	r.f_.addWeighted( 1.0, 1.0, 0.0 );
	r.f_.addWeighted( 1.0, 2.0, 0.0 );
	r.f_.addWeighted( 1.0, 3.0, 0.0 );
	r.f_.addWeighted( 1.0, 4.0, 0.0 );
	r.f_.addLength();
	r.f_.addRGB();
	r.f_.addLAB();
	r.w_ = VectorXf::Zero( r.f_.dim() );
	r.w_[0] = 1e-3;
	r.w_[3] = 1;
	return r;
}
EdgeWeight::EdgeWeight() {
}
VectorXf EdgeWeight::compute( const ImageOverSegmentation & ios ) const {
	return (f_.compute(ios) * w_).array().max(0);
}

int EdgeWeight::fdim() const {
	return f_.dim();
}
