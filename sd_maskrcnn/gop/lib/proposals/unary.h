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
#include "unaryfeature.h"
#include "segmentation/segmentation.h"

class Unary {
public:
	virtual ~Unary();
	virtual RMatrixXf compute( int seed ) const = 0;
};
class UnaryFactory {
public:
	virtual ~UnaryFactory();
	virtual std::shared_ptr<Unary> create( const UnaryFeatures & f ) const = 0;
	virtual FeatureSet requiredFeatures() const = 0;
	virtual int dim() const = 0;
	virtual bool isStatic() const;
};

std::shared_ptr<UnaryFactory> seedUnary();
std::shared_ptr<UnaryFactory> zeroUnary();
std::shared_ptr<UnaryFactory> rgbUnary( float scale );
std::shared_ptr<UnaryFactory> labUnary( float scale );
std::shared_ptr<UnaryFactory> learnedUnary( const std::string & filename );
std::shared_ptr<UnaryFactory> learnedUnary( const FeatureSet & features, const VectorXf & w, float b );
std::shared_ptr<UnaryFactory> learnedUnary( const VectorXf & w, float b );
std::shared_ptr<UnaryFactory> binaryLearnedUnary( const std::string & filename );
std::shared_ptr<UnaryFactory> binaryLearnedUnary( const FeatureSet & features, const VectorXf & w, float b );
std::shared_ptr<UnaryFactory> binaryLearnedUnary( const VectorXf & w, float b );
void saveLearnedUnary( const std::string & filename, std::shared_ptr<UnaryFactory> unary );
std::shared_ptr<UnaryFactory> backgroundUnary( const std::vector<int> & t = std::vector<int>(1,0) );

