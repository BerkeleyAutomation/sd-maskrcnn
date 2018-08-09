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
#include "graph.h"
#include "util.h"
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <iostream>

uint64_t hash_edge( uint32_t a, uint32_t b ) {
	return ((uint64_t)std::min(a,b) << 32) | std::max(a,b);
}
uint32_t edge_a( uint64_t h ) {
	return h>>32;
}
uint32_t edge_b( uint64_t h ) {
	return h&((1ll<<32)-1);
}
int getN(const Edges &graph) {
	int N = 1;
	for( auto e: graph )
		N = std::max( N, std::max(e.a, e.b)+1 );
	return N;
}
void saveEdges(std::ostream &s, const Edges &e)
{
	unsigned int ne = e.size();
	s.write((char *)&ne, sizeof(ne));
	s.write((char *)e.data(), ne * sizeof(Edge));
}
void loadEdges(std::istream &s, Edges &e)
{
	unsigned int ne;
	s.read((char *)&ne, sizeof(ne));
	e.resize(ne);
	s.read((char *)e.data(), ne * sizeof(Edge));
}
