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
#include "mex.h"
#include <string>
#include <map>
#include <memory>
#include <sstream>
#include "contour/directedsobel.h"
#include "contour/sketchtokens.h"
#include "contour/structuredforest.h"
#include "proposals/proposal.h"

#define MEX_ARGS int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs
typedef void (*MatlabFunction)(MEX_ARGS);


//////////////// class handling code /////////////////////
// Most of this code is from:
//  http://www.mathworks.com/matlabcentral/fileexchange/38964-example-matlab-class-wrapper-for-a-c++-class
#define CLASS_HANDLE_SIGNATURE 0xFF00F0A5
template<typename T> class Handle
{
    uint32_t sig_;
    std::string name_;
    std::shared_ptr<T> ptr_;
public:
    Handle(std::shared_ptr<T> ptr) : ptr_(ptr), name_(typeid(T).name()) { sig_ = CLASS_HANDLE_SIGNATURE; }
    ~Handle() { sig_ = 0; ptr_.reset(); }
    bool isValid() { return sig_ == CLASS_HANDLE_SIGNATURE && name_ == typeid(T).name(); }
    operator std::shared_ptr<T>() { return ptr_; }
};

template<typename T> inline mxArray * ptr2Mat(std::shared_ptr<T> ptr)
{
    mexLock();
    mxArray *out = mxCreateNumericMatrix(1, 1, mxUINT64_CLASS, mxREAL);
    *((uint64_t *)mxGetData(out)) = reinterpret_cast<uint64_t>(new Handle<T>(ptr));
    return out;
}

template<typename T> inline Handle<T> * mat2HandlePtr(const mxArray *in)
{
    if (mxGetNumberOfElements(in) != 1 || mxGetClassID(in) != mxUINT64_CLASS || mxIsComplex(in))
        mexErrMsgTxt("Input must be a real uint64 scalar.");
    Handle<T> *ptr = reinterpret_cast<Handle<T> *>(*((uint64_t *)mxGetData(in)));
    if (!ptr->isValid())
        mexErrMsgTxt("Handle not valid.");
    return ptr;
}
template<typename T> inline std::shared_ptr<T> mat2Ptr(const mxArray *in)
{
    return *mat2HandlePtr<T>(in);
}
template<typename T> inline void destroyObject(const mxArray *in)
{
    delete mat2HandlePtr<T>(in);
    mexUnlock();
}
//////////////// End class code /////////////////////

std::string toString( const mxArray * s ) {
    if( !mxIsChar(s) )
        mexErrMsgTxt( "Expected a string argument" );
    char* cs = mxArrayToString(s);
    std::string ss = cs;
    mxFree(cs);
    return ss;
}

/////////////// OverSegmentation ///////////////
static std::shared_ptr<BoundaryDetector> detectorFromString( const std::string & s ) {
    size_t p = s.find('(');
    std::string d_name = s.substr(0, p );
    std::string args       = s.substr(p+1, s.rfind(')')-p-1 );
    p = args.find('"');
    std::string fn = args.substr(p+1, args.rfind('"')-p-1 );
    
    if( d_name == "DirectedSobel" )
        return std::make_shared<DirectedSobel>();
    else if( d_name == "SketchTokens" ) {
        std::shared_ptr<SketchTokens> d = std::make_shared<SketchTokens>();
        d->load( fn );
        return d;
    }
    else if( d_name == "StructuredForest" ) {
        std::shared_ptr<StructuredForest> d = std::make_shared<StructuredForest>();
        d->load( fn );
        return d;
    }
    else if( d_name == "MultiScaleStructuredForest" ) {
        std::shared_ptr<MultiScaleStructuredForest> d = std::make_shared<MultiScaleStructuredForest>();
        d->load( fn );
        return d;
    }
    else {
        mexPrintf("Unknown detector '%s'", s.c_str());
        return std::make_shared<DirectedSobel>();
    }
}
static std::shared_ptr<BoundaryDetector> detector;
static void setDetector( MEX_ARGS ) {
    if( nrhs != 1 ) {
        mexErrMsgTxt( "Expected a single string argument" );
        return ;
    }
    detector = detectorFromString( toString( prhs[0] ) );
}
static void newImageOverSegmentationEmpty( MEX_ARGS ) {
    if( nlhs != 1 ) {
        mexErrMsgTxt("newImageOverSegmentation expected one return argument");
        return;
    }
    if( nrhs > 0 ) {
        mexErrMsgTxt( "Expected no arguments" );
        return ;
    }
    // Create the ImageOverSegmentation
    plhs[0] = ptr2Mat( std::make_shared<ImageOverSegmentation>() );
}
static void newImageOverSegmentation( MEX_ARGS ) {
    if( nlhs != 1 ) {
        mexErrMsgTxt("newImageOverSegmentation expected one return argument");
        return;
    }
    if( nrhs < 1 ) {
        mexErrMsgTxt( "Expected arguments: Image:HxWx3 uint8-array [,N_SPIX:int]" );
        return ;
    }
    // Read the image
    const mwSize * dims = mxGetDimensions(prhs[0]);
    if( mxGetNumberOfDimensions(prhs[0])!=3 || !mxIsUint8(prhs[0]) || dims[2]<3 ){
        mexErrMsgTxt( "HxWx3 uint8 image required" );
        return ;
    }
    int W = dims[1], H = dims[0];
    Image8u im( W, H, 3 );
    // For some reason MATLAB likes the fortran order
    uint8_t* pim = (uint8_t*)mxGetData(prhs[0]);
    for( int j=0; j<H; j++ )
        for( int i=0; i<W; i++ )
            for( int c=0; c<3; c++ )
                im(j,i,c) = pim[c*W*H+i*H+j];
    
    // Read the number of superpixels
    int n_spix=1000;
    if( nrhs > 1 && mxIsNumeric(prhs[1]) )
        n_spix = mxGetScalar(prhs[1]);
    
    if( nrhs > 3 ) {
        RMatrixXf bnd[2];
        for( int i=2; i<4; i++ ) {
            const mwSize * dims2 = mxGetDimensions(prhs[i]);
            if( mxGetNumberOfDimensions(prhs[i])!=2 || !mxIsSingle(prhs[i]) || dims2[0]!=dims[0] || dims2[1]!=dims[1] )
                mexErrMsgTxt( "Expected arguments: Image:HxWx3 uint8-array N_SPIX:int ThickBoundryMap:HxW single-array ThinBoundryMap:HxW single-array" );
            float * pBnd = (float *)mxGetData(prhs[i]);
			bnd[i-2] = MatrixXf::Map( pBnd, H, W );
        }
        plhs[0] = ptr2Mat( geodesicKMeans( im, bnd[0], bnd[1], n_spix ) );
    }
    else {
        if( !detector )
            detector = detectorFromString("MultiScaleStructuredForest(\"../data/sf.dat\")");
        // Create the OverSegmentation
        plhs[0] = ptr2Mat( geodesicKMeans( im, *detector, n_spix ) );
    }
}
static void ImageOverSegmentation_boundaryMap( MEX_ARGS ) {
    if( nrhs != 1 ) {
        mexErrMsgTxt( "Expected a ImageOverSegmentation" );
        return ;
    }
    if( nlhs != 1 ) {
        mexErrMsgTxt( "Expected a single return argument" );
        return ;
    }
    std::shared_ptr<ImageOverSegmentation> os = mat2Ptr<ImageOverSegmentation>( prhs[0] );
    RMatrixXf s = os->boundaryMap();
    // Create and write the resulting segmentation
    mwSize dims[2] = {(mwSize)s.rows(), (mwSize)s.cols()};
    plhs[0] = mxCreateNumericArray( 2, dims, mxSINGLE_CLASS, mxREAL );
    MatrixXf::Map( (float*)mxGetData(plhs[0]), s.rows(), s.cols() ) = s;
}
static void ImageOverSegmentation_s( MEX_ARGS ) {
    if( nrhs != 1 ) {
        mexErrMsgTxt( "Expected a ImageOverSegmentation" );
        return ;
    }
    if( nlhs != 1 ) {
        mexErrMsgTxt( "Expected a single return argument" );
        return ;
    }
    std::shared_ptr<ImageOverSegmentation> os = mat2Ptr<ImageOverSegmentation>( prhs[0] );
    RMatrixXs s = os->s();
    // Create and write the resulting segmentation
    mwSize dims[2] = {(mwSize)s.rows(), (mwSize)s.cols()};
    plhs[0] = mxCreateNumericArray( 2, dims, mxINT16_CLASS, mxREAL );
    MatrixXs::Map( (short*)mxGetData(plhs[0]), s.rows(), s.cols() ) = s;
}
static void ImageOverSegmentation_serialize( MEX_ARGS ) {
    if( nrhs != 1 ) {
        mexErrMsgTxt( "Expected a ImageOverSegmentation" );
        return ;
    }
    if( nlhs != 1 ) {
        mexErrMsgTxt( "Expected a single return argument" );
        return ;
    }
    std::shared_ptr<ImageOverSegmentation> os = mat2Ptr<ImageOverSegmentation>( prhs[0] );
	// Save to a string stream
	std::stringstream ss;
	os->save( ss );
	std::string data = ss.str();
	// And then copy it to a matlab array
	mwSize dims[1] = {(mwSize)data.size()};
	plhs[0] = mxCreateNumericArray( 1, dims, mxUINT8_CLASS, mxREAL );
	memcpy( mxGetData(plhs[0]), data.c_str(), data.size() );
}
static void ImageOverSegmentation_unserialize( MEX_ARGS ) {
    if( nrhs != 2 ) {
        mexErrMsgTxt( "Expected a ImageOverSegmentation and a buffer" );
        return ;
    }
    if( nlhs > 0 ) {
        mexErrMsgTxt( "Expected no return argument" );
        return ;
    }
    if( !mxIsUint8(prhs[1]) ) {
        mexErrMsgTxt( "Can only unserialize an 8-bit unsigned int array" );
        return ;
    }
    std::shared_ptr<ImageOverSegmentation> os = mat2Ptr<ImageOverSegmentation>( prhs[0] );
    std::stringstream ss( std::string( (char*) mxGetData(prhs[1]), mxGetM(prhs[1]) ) );
    os->load( ss );
}
static void ImageOverSegmentation_maskToBox( MEX_ARGS ) {
    if( nrhs != 2 || !mxIsLogical(prhs[1]) ) {
        mexErrMsgTxt( "Expected a ImageOverSegmentation and mask array" );
        return ;
    }
    if( nlhs != 1 ) {
        mexErrMsgTxt( "Expected a single return argument" );
        return ;
    }
    std::shared_ptr<ImageOverSegmentation> os = mat2Ptr<ImageOverSegmentation>( prhs[0] );
    RMatrixXi r = os->maskToBox( MatrixXb::Map( (mxLogical*)mxGetData(prhs[1]), mxGetM(prhs[1]), mxGetN(prhs[1]) ) );
    // Create and write the resulting segmentation
    mwSize dims[2] = {(mwSize)r.rows(), (mwSize)r.cols()};
    plhs[0] = mxCreateNumericArray( 2, dims, mxINT32_CLASS, mxREAL );
    MatrixXi::Map( (int*)mxGetData(plhs[0]), r.rows(), r.cols() ) = r;
}
static void freeImageOverSegmentation( MEX_ARGS ) {
    if( nrhs == 0 ) {
        mexErrMsgTxt("Expected to get something to free");
        return;
    }
    for( int i=0; i<nrhs; i++ )
        destroyObject<ImageOverSegmentation>( prhs[i] );
}
/////////////// Proposals ///////////////
std::shared_ptr<UnaryFactory> createUnaryFromString( const std::string & s ) {
    size_t p = s.find('(');
    std::string unary_name = s.substr(0, p );
    std::string args       = s.substr(p+1, s.rfind(')')-p-1 );
    if( unary_name == "seedUnary" )
        return seedUnary();
    else if( unary_name == "zeroUnary" )
        return zeroUnary();
    else if( unary_name == "backgroundUnary" ) {
        std::vector<int> types;
        for( size_t i=1; i+1<args.length(); i++ ) {
            size_t n = args.find( ',', i );
            if( n==args.npos )
                n = args.length()-1;
            types.push_back( std::stoi( args.substr( i, n-i ) ) );
            i = n;
        }
        return backgroundUnary( types );
    }
    else if( unary_name == "binaryLearnedUnary" ) {
        p = args.find('"');
        std::string fn = args.substr(p+1, args.rfind('"')-p-1 );
        try {
            return binaryLearnedUnary( fn );
        }
        catch (...) {
            mexWarnMsgTxt(("Unary term '"+s+"' not found!").c_str());
        }
    }
    return zeroUnary();
}
std::shared_ptr<SeedFunction> createSeed( const std::string & s ) {
    if( s == "RegularSeed"  ) return std::make_shared<RegularSeed>();
    if( s == "SaliencySeed" ) return std::make_shared<SaliencySeed>();
    if( s == "GeodesicSeed" ) return std::make_shared<GeodesicSeed>();
    if( s == "RandomSeed"   ) return std::make_shared<RandomSeed>();
    if( s == "SegmentationSeed" ) return std::make_shared<SegmentationSeed>();
    std::shared_ptr<LearnedSeed> seed = std::make_shared<LearnedSeed>();
    try {
        seed->load( s );
    }
    catch (...) {
        mexWarnMsgTxt(("Seed '"+s+"' not found! Using Geodesic!").c_str());
        return std::make_shared<GeodesicSeed>();
    }
    return seed;
}
static void newProposal( MEX_ARGS ) {
    if( nlhs == 0 ) {
        mexErrMsgTxt("newProposal expected one return argument");
        return;
    }
    // Create an empty unary
    ProposalSettings prop_settings;
    prop_settings.unaries.clear();
    
    // Add all settings
    for ( int i=0; i<nrhs; i++ ) {
        std::string c = toString( prhs[i] );
        
        // Process the command
        if (c == "max_iou") {
            if( i+1>=nrhs || !mxIsNumeric(prhs[i+1]) )
                mexErrMsgTxt("max_iou numeric argument required");
            prop_settings.max_iou = mxGetScalar(prhs[i+1]);
            i++;
        }
        else if (c == "seed") {
            if( i+1>=nrhs || !mxIsChar(prhs[i+1]) )
                mexErrMsgTxt("seed string argument required");
            prop_settings.foreground_seeds = createSeed( toString( prhs[i+1] ) );
            i++;
        }
        else if (c == "unary") {
            if( i+4 >= nrhs || !mxIsNumeric(prhs[i+1]) || !mxIsNumeric(prhs[i+2]) || !mxIsChar(prhs[i+3]) || !mxIsChar(prhs[i+4]) )
                mexErrMsgTxt("unary N_S:int N_T:int fg_unary:string bg_unary:string [min_size:float max_side:float]");
            const int N_S = mxGetScalar(prhs[i+1]), N_T = mxGetScalar(prhs[i+2]);
            std::string fg_unary = toString( prhs[i+3] ), bg_unary = toString( prhs[i+4] );
            float min_size = 0.0, max_size=0.75;
            if( i+6 <= nrhs && mxIsNumeric(prhs[i+5]) && mxIsNumeric(prhs[i+5]) ) {
                min_size = mxGetScalar( prhs[i+5] );
                max_size = mxGetScalar( prhs[i+6] );
                i += 2;
            }
            prop_settings.unaries.push_back( ProposalSettings::UnarySettings( N_S, N_T, createUnaryFromString( fg_unary ), createUnaryFromString( bg_unary ), min_size, max_size ) );
            i+=4;
        }
        else {
            mexErrMsgTxt(("Setting '"+c+"' not found").c_str());
        }
    }
    plhs[0] = ptr2Mat( std::make_shared<Proposal>( prop_settings ) );
}
static void Proposal_propose( MEX_ARGS ) {
    if( nrhs != 2 ) {
        mexErrMsgTxt("Expected to get proposal and ImageOverSegmentation");
        return;
    }
    if( nlhs != 1 ) {
        mexErrMsgTxt("Expected a single output argument");
        return;
    }
    std::shared_ptr<Proposal> p = mat2Ptr<Proposal>( prhs[0] );
    std::shared_ptr<ImageOverSegmentation> os = mat2Ptr<ImageOverSegmentation>( prhs[1] );
    
    RMatrixXb r = p->propose( *os );
    
    // Create and write the resulting segmentation
    mwSize dims[2] = {(mwSize)r.rows(), (mwSize)r.cols()};
    plhs[0] = mxCreateNumericArray( 2, dims, mxLOGICAL_CLASS, mxREAL );
    MatrixXb::Map( (mxLogical*)mxGetData(plhs[0]), r.rows(), r.cols() ) = r;
}
static void freeProposal( MEX_ARGS ) {
    if( nrhs == 0 ) {
        mexErrMsgTxt("Expected to get something to free");
        return;
    }
    for( int i=0; i<nrhs; i++ )
        destroyObject<Proposal>( prhs[i] );
}

typedef std::pair<std::string,MatlabFunction> C;
#define A( fn ) C( #fn, fn ),
C command_list[] = {
    A( setDetector )
    A( newImageOverSegmentation )
    A( newImageOverSegmentationEmpty )
    A( ImageOverSegmentation_s )
    A( ImageOverSegmentation_boundaryMap )
    A( ImageOverSegmentation_maskToBox )
    A( ImageOverSegmentation_serialize )
    A( ImageOverSegmentation_unserialize )
    A( freeImageOverSegmentation )
    A( newProposal )
    A( Proposal_propose )
    A( freeProposal )
};
std::map<std::string,MatlabFunction> commands( command_list, command_list+sizeof(command_list)/sizeof(command_list[0]) );

void mexFunction( MEX_ARGS ) {
    if (nrhs == 0) {
        mexErrMsgTxt("An API command is required");
        return;
    }
    
    // Get the command
    std::string c = toString( prhs[0] );
    
    // Execute the command
    if (commands.find( c ) == commands.end()) {
        mexErrMsgTxt( (std::string("API command not recognized '")+c+"'").c_str() );
        return;
    }
    commands[c](nlhs, plhs, nrhs-1, prhs+1);
}