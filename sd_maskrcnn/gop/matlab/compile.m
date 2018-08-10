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
function compile()

    util_files = addCppFiles('../lib/util');
    imgproc_files = addCppFiles('../lib/imgproc');
    learning_files = addCppFiles('../lib/learning');
    contour_files = addCppFiles('../lib/contour');
    segmentation_files = addCppFiles('../lib/segmentation');
    proposals_files = addCppFiles('../lib/proposals');
    all_files = [util_files,imgproc_files,learning_files,contour_files,segmentation_files,proposals_files];
    EIGEN_DIR = '/usr/include/eigen3';
    if ismac
        EIGEN_DIR = '/opt/local/include/eigen3';
        cmd = ['mex -DLBFGS_FLOAT=32 -I/opt/local/include/ -DEIGEN_DONT_PARALLELIZE -DNO_IMREAD -I../lib -I',EIGEN_DIR,' -I../external/liblbfgs-1.10/include/ CFLAGS="-fPIC -O3" CXXFLAGS="-fPIC -std=c++11 -O3 -stdlib=libc++" gop_mex.cpp ',' ../external/liblbfgs-1.10/lib/lbfgs.c ', all_files];
    elseif ispc
        EIGEN_DIR = '../external/eigen';
        cmd = ['mex -DLBFGS_FLOAT=32 -DEIGEN_DONT_PARALLELIZE -DNO_IMREAD -I../lib -I',EIGEN_DIR,' -I../external/liblbfgs-1.10/include/ gop_mex.cpp ',' ../external/liblbfgs-1.10/lib/lbfgs.c ', all_files];
    else
        cmd = ['mex -DLBFGS_FLOAT=32 -DEIGEN_DONT_PARALLELIZE -DNO_IMREAD -D_GLIBCXX_USE_NANOSLEEP -I../lib -I',EIGEN_DIR,' -I../external/liblbfgs-1.10/include/ CFLAGS="-fPIC -fno-omit-frame-pointer -pthread -O3 -DNDEBUG" CXXFLAGS="-fPIC -fno-omit-frame-pointer -pthread -O3 -DNDEBUG -std=c++11" gop_mex.cpp ',' ../external/liblbfgs-1.10/lib/lbfgs.c ', all_files];
    end
    eval( cmd )

end
function r = addCppFiles(d)
    files = dir([d,'/*.cpp']);
    r = sprintf([d,'/%s '],files.name);
end