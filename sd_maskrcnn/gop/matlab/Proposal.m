%{
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
%}
% Usage example:
% I = imread('peppers.png');
% os = OverSegmentation( I );
% p = Proposal('max_iou', 0.8,...
%              'unary', 130, 5, 'seedUnary()', 'backgroundUnary({0,15})',...
%              'unary', 0, 5, 'zeroUnary()', 'backgroundUnary({0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15})'
%              );
% props = p.propose( os );

classdef Proposal
    properties (SetAccess = private, GetAccess = private)
        c_p
    end
    
    methods
        function obj = Proposal( varargin )
            obj.c_p = gop_mex( 'newProposal', varargin{:} );
        end
        function r = propose(this,os)
            r = gop_mex( 'Proposal_propose', this.c_p, os.c_s );
        end
        function sobj = saveobj(this)
            error( 'You cannot load/save a Proposal object!' );
        end
        function loadobj(this, sobj)
            error( 'You cannot load/save a Proposal object!' );
        end
        function delete(this)
            gop_mex( 'freeProposal', this.c_p );
        end
    end
end
