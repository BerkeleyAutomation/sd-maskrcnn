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
#include "eigen.h"

class EnergyFunction {
public:
	virtual VectorXf optimizationGradient( const VectorXf & x, float & e ) const;
	virtual VectorXf optimizationInitialGuess() const;
	virtual VectorXf optimizationTransformResult( const VectorXf & x ) const;
	virtual ~EnergyFunction();
	virtual VectorXf initialGuess() const = 0;
	virtual VectorXf gradient( const VectorXf & x, float & e ) const = 0;
};
class PositiveConstrainedEnergyFunction: public EnergyFunction {
protected:
	int n_positive_;
public:
	// An energy function with the first n elements contrained to be strictly positive
	PositiveConstrainedEnergyFunction( int n_positive );
	virtual VectorXf optimizationGradient( const VectorXf & x, float & e ) const;
	virtual VectorXf optimizationInitialGuess() const;
	virtual VectorXf optimizationTransformResult( const VectorXf & x ) const;
};
VectorXf minimizeLBFGS( const EnergyFunction & f, float & e, int verbose=0 );
VectorXf minimizeLBFGS( const EnergyFunction & f, int verbose=0 );
// VectorXf minimizeLBFGSB( const EnergyFunction & f, const VectorXf & min, const VectorXf & max, float & e, int verbose=0 );
// VectorXf minimizeLBFGSB( const EnergyFunction & f, const VectorXf & min, const VectorXf & max, int verbose=0 );
float gradCheck( const EnergyFunction & f, const VectorXf & x0, int verbose=1 );
float gradCheck( const EnergyFunction & f, int verbose=1 );

class LBFGS {
protected:
	float eps_;
	int max_iter_, max_line_search_, max_history_;
	bool restart_;
//	float ;
	void progress(const EnergyFunction & f, int n_it, const VectorXf & x, const VectorXf & g, float fx, float step ) const;
	float backtrackWolfe( const EnergyFunction & f, VectorXf & x, VectorXf & g, float & fx, const VectorXf & z, int k ) const;
//	float backtrackmMoreThuente( const EnergyFunction & f, VectorXf & x, VectorXf & g, float & fx, const VectorXf & z, int k ) const;
public:
	LBFGS( float eps=1e-5, int max_iter=500, int max_line_search=20, int max_history=6, bool restart=false );
	VectorXf minimize( const EnergyFunction & f, float & e, int verbose=0 ) const;
	VectorXf minimize( const EnergyFunction & f, int verbose=0 ) const;
};
class SGD {
protected:
	float alpha_;
	int n_iter_, mb_size_;
public:
	SGD( float alpha=1e-3, int n_iter=100, int mini_batch_size=10 );
	
	VectorXf minimize( const EnergyFunction & f, float & e, int verbose=0 ) const;
	VectorXf minimize( const EnergyFunction & f, int verbose=0 ) const;
};

