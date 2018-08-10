#include "contour/sketchtokens.h"
#include "contour/structuredforest.h"
#include "proposals/proposal.h"
#include "segmentation/segmentation.h"
#include "imgproc/image.h"

int main( int argc, const char * argv[] ) {

	/* Setup the proposals settings */
	// Play with those numbers to increase the number of proposals
	// Number of seeds N_S and number of segmentations per seed N_T
	const int N_S = 140, N_T = 4;
	// Maximal overlap between any two proposals (intersection / union)
	const float max_iou = 0.8;
	
	ProposalSettings prop_settings;
	prop_settings.max_iou = max_iou;
	
	// Load the seed function
	std::shared_ptr<LearnedSeed> seed = std::make_shared<LearnedSeed>();
	seed->load( "../data/seed_final.dat" );
	prop_settings.foreground_seeds = seed;
	
	// Load the foreground/background proposals
	for( int i=0; i<3; i++ )
		prop_settings.unaries.push_back( ProposalSettings::UnarySettings( N_S, N_T, binaryLearnedUnary("../data/masks_final_"+std::to_string(i)+"_fg.dat"), binaryLearnedUnary("../data/masks_final_"+std::to_string(i)+"_bg.dat") ) );
	// Pure background proposals
	std::vector<int> allbg = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
	prop_settings.unaries.push_back( ProposalSettings::UnarySettings( 0, N_T, zeroUnary(), backgroundUnary(allbg), 0.1, 1  ) );
	
	/* Create the proposlas */
	Proposal prop( prop_settings );
	
	/* Create the boundary detector */
	// Sketch tokens
	// SketchTokens detector;
	// detector.load( "../data/st_full_c.dat" );
	
	// Structured Forests are a bit faster, but produce more proposals
	// StructuredForest detector;
	// detector.load( "../data/sf.dat" );
	
	// Muilti Scale Structured Forests generally perform best
	MultiScaleStructuredForest detector;
	detector.load( "../data/sf.dat" );

	for( int i=1; i<argc; i++ ) {
		// Load an image
		Image8u im = imread(argv[i]);
		
		// Create an over-segmentation
		std::shared_ptr<ImageOverSegmentation> s = geodesicKMeans( im, detector, 1000 );
		RMatrixXb p = prop.propose( *s );
		printf("Generated %d proposals\n", (int)p.rows() );
		
		// If you just want boxes use
		RMatrixXi boxes = s->maskToBox( p );
		
		// To use the proposals use the over segmentation s.s() and p.row(n)
		// you can get the binary segmentation mask using the following lines
		int n_prop = 0;
		
		RMatrixXb segment( s->s().rows(), s->s().cols() );
		for( int j=0; j<s->s().rows(); j++ )
			for( int i=0; i<s->s().cols(); i++ )
				segment(j,i) = p( n_prop, s->s()(j,i) );
	}

	return 0;
}