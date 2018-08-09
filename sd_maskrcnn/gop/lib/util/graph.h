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
#include <cstdint>
#include <vector>
#include <iosfwd>

uint64_t hash_edge( uint32_t a, uint32_t b );
uint32_t edge_a( uint64_t h );
uint32_t edge_b( uint64_t h );

struct Edge {
	int a, b;
	Edge( int a=0, int b=0, bool ordered=false ):a(a),b(b){
		if (!ordered && this->b > this->a)
			std::swap( this->a, this->b );
	}
	static Edge fromHash( uint64_t h ){
		return Edge( edge_a(h), edge_b(h) );
	}
	bool operator==(const Edge & o ) const {
		return a==o.a && b==o.b;
	}
	uint64_t hash() const {
		return hash_edge( a, b );
	}
};
namespace std {
	template <>
	struct hash<Edge> {
		size_t operator () (const Edge &f) const { return f.hash(); }
	};
}
typedef std::vector<Edge> Edges;
int getN( const Edges & graph );

void saveEdges( std::ostream & s, const Edges & e );
void loadEdges( std::istream & s, Edges & e );
