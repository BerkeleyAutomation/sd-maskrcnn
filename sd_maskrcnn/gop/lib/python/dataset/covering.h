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
#include <boost/python.hpp>
#include <boost/numpy.hpp>
#include "util/eigen.h"
#include "util/rasterize.h"

class ImageOverSegmentation;
using namespace boost::python;
namespace np = boost::numpy;

struct ProposalEvaluation {
	ProposalEvaluation( const short * gt_seg, int W, int H, int D, const RMatrixXs & over_seg, const RMatrixXb & props );
	ProposalEvaluation( const short * gt_seg, int W, int H, int D, const std::vector<RMatrixXs> & over_seg, const std::vector<RMatrixXb> & props );
	ProposalEvaluation( const std::vector<Polygons> & regions, const RMatrixXs & over_seg, const RMatrixXb & props );
	ProposalEvaluation( const std::vector<Polygons> & regions, const std::vector<RMatrixXs> & over_seg, const std::vector<RMatrixXb> & props );
	VectorXf bo_;
	VectorXf area_;
	float pool_size_;
};

struct ProposalBoxEvaluation {
	ProposalBoxEvaluation( const RMatrixXi & bbox, const RMatrixXi & prop_boxes );
	ProposalBoxEvaluation( const RMatrixXi & bbox, const std::vector<RMatrixXi> & prop_boxes );
	VectorXf bo_;
	float pool_size_;
};
