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
#include "morph.h"
#include <util/util.h>
#include <cstring>
#include <vector>

int thinningGuoHallIteration(bool * out, const bool * in, int W, int H, int iter) {
	int cnt = 0;
	memcpy( out, in, W*H*sizeof(bool) );
	// Maybe we can vectorize this?
	for (int j = 0; j < H; j++)
		for (int i = 0; i < W; i++){
			int n = j*W+i;
			if( in[n] ){
				int x0 = i?-1:1, x1 = i+1<W?1:-1;
				int y0 = j?-W:W, y1 = j+1<H?W:-W;
				bool p2 = in[n+y0];
				bool p3 = in[n+y0+x1];
				bool p4 = in[n+x1];
				bool p5 = in[n+y1+x1];
				bool p6 = in[n+y1];
				bool p7 = in[n+y1+x0];
				bool p8 = in[n+x0]; 
				bool p9 = in[n+y0+x0];
				
				int C  = (!p2 & (p3 | p4)) + (!p4 & (p5 | p6)) + (!p6 & (p7 | p8)) + (!p8 & (p9 | p2));
				int N1 = (p9 | p2) + (p3 | p4) + (p5 | p6) + (p7 | p8);
				int N2 = (p2 | p3) + (p4 | p5) + (p6 | p7) + (p8 | p9);
				int N  = N1 < N2 ? N1 : N2;
				int m  = iter == 1 ? ((p6 | p7 | !p9) & p8) : ((p2 | p3 | !p5) & p4);
				
				if (C == 1 && (N >= 2 && N <= 3) && m == 0) {
					out[n] = 0;
					cnt++;
				}
			}
		}
	return cnt;
}
void thinningGuoHall( RMatrixXb & b ) {
	std::vector<char> tmp(b.cols()*b.rows(),false);
	for( int i=0,n=1; n>0; i++ ) {
		n=0;
		n += thinningGuoHallIteration((bool*)tmp.data(), b.data(), b.cols(), b.rows(), 0);
		n += thinningGuoHallIteration(b.data(), (bool*)tmp.data(), b.cols(), b.rows(), 1);
	}
}
