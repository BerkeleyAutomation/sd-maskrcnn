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

// Variance of the box filter is (rad+1)*rad/3
void boxFilter( float * r, const float * im, int W, int H, int C, int rad );
// Variance of the tent filter is (rad+2)*rad/6
// To approximate a gaussian with std s use r=sqrt(1+6*s*s)-1, for large s -> r=sqrt(6)*s-1
// or use to tent filters with r=sqrt(1+3*s*s)-1  ~= sqrt(3)*s-1
void tentFilter( float * r, const float * im, int W, int H, int C, int rad );
void gaussianFilter( float * r, const float * im, int W, int H, int C, float sigma );
void exactGaussianFilter( float * r, const float * im, int W, int H, int C, float sigma, int R );

void percentileFilter( float * r, const float * im, int W, int H, int C, int rad, float p );
void minFilter( float * r, const float * im, int W, int H, int C, int rad );
void maxFilter( float * r, const float * im, int W, int H, int C, int rad );
