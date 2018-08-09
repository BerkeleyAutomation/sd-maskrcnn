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
% Set a boundary detector by calling (before creating an OverSegmentation!):
% gop_mex( 'setDetector', 'SketchTokens("../data/st_full_c.dat")' );
% gop_mex( 'setDetector', 'StructuredForest("../data/sf.dat")' );
% gop_mex( 'setDetector', 'MultiScaleStructuredForest("../data/sf.dat")' );

% Usage example:
% I = imread('peppers.png');
% os = OverSegmentation( I );
% imagesc( os.s() ) % To visualize the over-segmentation

classdef OverSegmentation
    properties (SetAccess = private, GetAccess = public)
        c_s
    end
    methods
        function obj = OverSegmentation( I, NS, thick_bnd, thin_bnd )
            if nargin == 0
                error( 'Input image required' );
            end
            if nargin < 2
                NS = 1000;
            end
            if nargin == 3
                error( 'Cannot have a thick boundary map without a thinned out one!' );
            end
            if nargin < 3
                obj.c_s = gop_mex( 'newImageOverSegmentation', I, NS );
            else
                obj.c_s = gop_mex( 'newImageOverSegmentation', I, NS, thick_bnd, thin_bnd );
            end
        end
        function sobj = saveobj(this)
            sobj = this;
            sobj.c_s = gop_mex( 'ImageOverSegmentation_serialize', this.c_s );
        end
        function r = s(this)
            r = gop_mex( 'ImageOverSegmentation_s', this.c_s );
        end
        function r = boundaryMap(this)
            r = gop_mex( 'ImageOverSegmentation_boundaryMap', this.c_s );
        end
        function r = maskToBox(this,mask)
            r = gop_mex( 'ImageOverSegmentation_maskToBox', this.c_s, mask );
        end
        function delete(this)
            gop_mex( 'freeImageOverSegmentation', this.c_s );
        end
    end
    methods (Static)
        function obj = loadobj(obj)
            data = obj.c_s;
            obj.c_s = gop_mex( 'newImageOverSegmentationEmpty' );
            gop_mex( 'ImageOverSegmentation_unserialize', obj.c_s, data )
        end
    end
end
