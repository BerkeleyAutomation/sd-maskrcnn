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
#include "treedata.h"

BinaryDistribution::BinaryDistribution() {
	p_[0] = p_[1] = 0;
}
BinaryDistribution& BinaryDistribution::operator+=( const BinaryDistribution & o ) {
	p_[0] += o.p_[0];
	p_[1] += o.p_[1];
	return *this;
}
void BinaryDistribution::save( std::ostream &s ) const {
	s.write( (const char*)p_, sizeof(p_) );
}
void BinaryDistribution::load( std::istream &s ) {
	s.read( (char*)p_, sizeof(p_) );
}
void BinaryDistribution::normalize() {
	float n = p_[0] + p_[1] + 1e-20;
	p_[0] /= n;
	p_[1] /= n;
}

LabelData::LabelData() {
	lbl = -1;
}
std::vector<int>& operator+=( std::vector<int> & v, const LabelData & o ) {
	v.push_back( o.lbl );
	return v;
}
void LabelData::save( std::ostream &s ) const {
	s.write( (const char*)&lbl, sizeof(lbl) );
}
void LabelData::load( std::istream &s ) {
	s.read( (char*)&lbl, sizeof(lbl) );
}

RangeData::RangeData() {
	begin = end = 0;
}
std::vector<RangeData>& operator+=( std::vector<RangeData> & v, const RangeData & o ) {
	v.push_back( o );
	return v;
}
void RangeData::save( std::ostream &s ) const {
	s.write( (const char*)&begin, sizeof(begin) );
	s.write( (const char*)&end, sizeof(end) );
}
void RangeData::load( std::istream &s ) {
	s.read( (char*)&begin, sizeof(begin) );
	s.read( (char*)&end, sizeof(end) );
}

PatchData& PatchData::operator+=( const PatchData & o ) {
	offsets_.reserve( offsets_.size()+o.offsets_.size() );
	offsets_.insert( offsets_.end(), o.offsets_.begin(), o.offsets_.end() );
	return *this;
}
void PatchData::save( std::ostream & s ) const {
	int n = offsets_.size();
	s.write( (const char*)&n, sizeof(n) );
	s.write( (const char*)offsets_.data(), n*sizeof(offsets_[0]) );
}
void PatchData::load( std::istream & s ) {
	int n;
	s.read( (char*)&n, sizeof(n) );
	offsets_.resize( n );
	s.read( (char*)offsets_.data(), n*sizeof(offsets_[0]) );
}

