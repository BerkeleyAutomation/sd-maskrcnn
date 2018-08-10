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
#include "seed.h"
#include "unary.h"
#include "edgeweight.h"

struct ProposalSettings {
	struct UnarySettings {
		int n_seed;
		int n_lvl_set;
		std::shared_ptr<UnaryFactory> fg_unary, bg_unary;
		std::shared_ptr<EdgeWeight> edge_weight;
		float min_size, max_size;
		UnarySettings( int n_seed = 10, int n_lvl_set = 5, std::shared_ptr<UnaryFactory> fg_unary = seedUnary(), std::shared_ptr<UnaryFactory> bg_unary = backgroundUnary(), float min_size=0, float max_size=0.75 ):n_seed(n_seed), n_lvl_set(n_lvl_set), fg_unary(fg_unary), bg_unary(bg_unary), min_size(min_size), max_size(max_size) {
			edge_weight = std::make_shared<EdgeWeight>(EdgeWeight::makeDefault());
		}
		bool operator==(const UnarySettings & o) const {
			// Makeing the boost python indexing suite happy (and totally breaking the contains functions)
			return n_seed == o.n_seed && n_lvl_set == o.n_lvl_set && min_size == o.min_size && max_size == o.max_size;
		}
	};
	std::shared_ptr<SeedFunction> foreground_seeds;
	std::vector<UnarySettings> unaries;
	float max_iou;
	ProposalSettings( const ProposalSettings & o ):foreground_seeds(o.foreground_seeds->clone()),unaries(o.unaries){
		max_iou = o.max_iou;
	}
	ProposalSettings(){
		foreground_seeds = std::make_shared<GeodesicSeed>();
		max_iou = 0.85;
	}
};

class Proposal {
protected:
	ProposalSettings psettings_;
	void proposeAll( const ImageOverSegmentation & ios, const VectorXi & seeds, const std::function<void(const ArrayXf &,float,float,int,int,int)> & f ) const;
	VectorXi makeSeeds( const ImageOverSegmentation & ios ) const;
public:
	static VectorXf computeLevelSets( const VectorXf & d, int n_lvl_set, float max_iou, int min_size, int max_size, VectorXf * score=NULL );
	Proposal( const ProposalSettings & psettings );
	RMatrixXb propose( const ImageOverSegmentation & ios ) const;
};

