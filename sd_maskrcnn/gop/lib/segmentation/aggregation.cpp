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
#include "aggregation.h"
#include "util/algorithm.h"

template<typename T>
struct ClonableAggregationFunction:public AggregationFunction {
	virtual std::shared_ptr<AggregationFunction> clone() const {
		return std::make_shared<T>( dynamic_cast<const T&>(*this) );
	}
};
struct MaxFunction: public ClonableAggregationFunction<MaxFunction> {
	float v_=0;
	virtual void add( float v, float w=1 ){
		if(w>0)
			v_ = std::max( v_, v );
	}
	virtual float get() const{
		return v_;
	}
};
struct MinFunction: public ClonableAggregationFunction<MinFunction> {
	float v_=std::numeric_limits<float>::infinity();
	virtual void add( float v, float w=1 ){
		if(w>0)
			v_ = std::min( v_, v );
	}
	virtual float get() const{
		return v_;
	}
};
struct AvgFunction: public ClonableAggregationFunction<AvgFunction> {
	float v_=0, n_=0;
	virtual void add( float v, float w=1 ){
		v_ += w*v;
		n_ += w;
	}
	virtual float get() const{
		return v_ / (n_+1e-10);
	}
};
struct PercentileFunction: public ClonableAggregationFunction<PercentileFunction> {
	PercentileFunction( float p ):p_(p){
	}
	float p_;
	std::vector<float> v_;
	virtual void add( float v, float w=1 ){
		v_.push_back( v );
	}
	virtual float get() const{
		return quickSelect( v_, p_*(v_.size()-1)+0.5 );
	}
};
std::shared_ptr< AggregationFunction > AggregationFunction::create(const std::string &name) {
	std::string lname = name;
	std::transform( lname.begin(), lname.end(), lname.begin(), tolower );
	
	if( lname == "mean" || lname == "avg" )
		return std::make_shared<AvgFunction>();
	else if( lname == "min" )
		return std::make_shared<MinFunction>();
	else if( lname == "max" )
		return std::make_shared<MaxFunction>();
	else if( lname == "med" || lname == "median" )
		return std::make_shared<PercentileFunction>( 0.5 );
	else if( lname.length()>1 && lname[0] == 'p' )
		return std::make_shared<PercentileFunction>( std::stoi( lname.substr(1) ) / 100. );
	else
		throw std::invalid_argument( "Unknown aggregation function '"+name+"'!" );
}
