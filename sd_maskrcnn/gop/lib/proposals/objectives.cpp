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
#include "objectives.h"


VectorXf SplitLogisticObjectve::gradient ( const VectorXf & d, const VectorXb & gt, float & e ) {
	const float b = 1;
	
	ArrayXf gi = 2*gt.cast<float>().array()-1;
	// Compute the sign of the "distace" (gt=1: d  gt=0: -d)
	ArrayXf d_gt = b+gi*d.array();
	// Split the values into positive and negative values
	ArrayXf d_n = d_gt.min(0), d_p = d_gt.max(0);
	ArrayXf e_n = d_n.exp(), ne_p = (-d_p).exp();
	// Likelihood: -log(p) = log(1 + exp(db_s)) [which is numerically unstable]
	//                     = db_p + log( exp(-db_p) + exp(db_s-db_p) ) = db_p + log( exp(-db_p) + exp(db_n) ) [which is numerically stable]
	e = ( d_p + (e_n + ne_p).log() ).sum();
	// p = 1 / (1 + exp(d-b))
	ArrayXf p = ne_p / (ne_p + e_n);
	// Gradient is p - gt
	return gi*(1 - p);
}
VectorXf WeightedSplitLogisticObjectve::gradient ( const VectorXf & d, const VectorXb & gt, float & e ) {
	const float n_pos = gt.cast<float>().array().sum();
	const float w_n = (n_pos+0.05) / (gt.size()+0.1);
	const float w_p = 1 - w_n;
	const ArrayXf w = gt.cast<float>().array()*(w_p-w_n)+w_n;
	const float b = 1;
	
	ArrayXf gi = 2*gt.cast<float>().array()-1;
	// Compute the sign of the "distace" (gt=1: d  gt=0: -d)
	ArrayXf d_gt = b+gi*d.array();
	// p = 1 / (1 + exp(d-b))
	ArrayXf p = 1.0 / (1.0 + d_gt.exp());
	// Split the values into positive and negative values
	ArrayXf db_n = d_gt.array().min(0), db_p = d_gt.array().max(0);
	// Likelihood: -log(p) = log(1 + exp(db_s)) [which is numerically unstable]
	//                     = db_p + log( exp(-db_p) + exp(db_s-db_p) ) = db_p + log( exp(-db_p) + exp(db_n) ) [which is numerically stable]
	e = (w*( db_p + ((-db_p).exp() + db_n.exp()).log() )).sum();
	// Gradient is p - gt
	return w*gi*(1 - p);
}
VectorXf SplitIOUObjectve::gradient ( const VectorXf & d, const VectorXb & gt, float & e ) {
	const float n_pos = gt.cast<float>().array().sum()+1e-30;
	const float b = 1;
	
	ArrayXf gi = 2*gt.cast<float>().array()-1;
	// Compute the sign of the "distace" (gt=1: d  gt=0: -d)
	ArrayXf d_gt = b+gi*d.array();
	
	ArrayXf p = 1.0 / (1.0 + d_gt.exp());
	
	// Compute the positive and negative props
	ArrayXf p_pos = gt.cast<float>().array()*p, p_neg = 1-(1-gt.cast<float>().array())*p;
	
	// Compute the IOU energy
	float i = p_pos.sum(), u = n_pos + p_neg.sum();
	e = -i / u;
	ArrayXf d_p = (1-p)*p / u;
	// Gradient is p - gt
	return - d_p * ( i/u - (1+i/u)*gt.cast<float>().array() );
}
float LogisticObjective::optimizeB( const VectorXf & d, float n_pos ) {
	// Use binary search to find the optimal offset b
	const float v = n_pos / d.size();
	const float o = log(1-v+1e-10) - log(v+1e-10);
	float b0 = d.minCoeff()-o, b1 = d.maxCoeff()-o;
	while( b0+1e-4 < b1 ) {
		float b = (b0+b1) / 2.0;
		float v = (1.0 / (1.0 + (d.array()-b).exp())).sum();
		if( fabs(n_pos - v) < 1e-4*(fabs(n_pos)+fabs(v)) )
			break;
		if( v > n_pos )
			b1 = b;
		else
			b0 = b;
	}
	return (b0+b1)/2.0;
}
VectorXf LogisticObjective::gradient( const VectorXf & d, const VectorXb & gt, float & e ) {
	const float b = optimizeB( d, gt.cast<int>().array().sum() );
	ArrayXf db = d.array()-b;
	// p = 1 / (1 + exp(b-d))
	ArrayXf p = 1.0 / (1.0 + db.exp());
	// Compute the sign of the likelihood (gt=1: b-d  gt=0: d-b)
	ArrayXf db_s = (2*gt.cast<float>().array()-1)*db;
	// Split the values into positive and negative values
	ArrayXf db_n = db_s.array().min(0), db_p = db_s.array().max(0);
	// Likelihood: -log(p) = log(1 + exp(db_s)) [which is numerically unstable]
	//                     = db_p + log( exp(-db_p) + exp(db_s-db_p) ) = db_p + log( exp(-db_p) + exp(db_n) ) [which is numerically stable]
	e = ( db_p + ((-db_p).exp() + db_n.exp()).log() ).sum() / d.size();
	// Gradient is p - gt
	return -(p.matrix() - gt.cast<float>()) / d.size();
}
float WeightedLogisticObjective::optimizeB( const VectorXf & d, const VectorXb & gt, float w_n, float w_p ) {
	if( w_p < 0 )
		w_p = 1.0 - w_n;
	const float n_pos = gt.cast<float>().array().sum();
	const ArrayXf w = gt.cast<float>().array()*(w_p-w_n)+w_n;
	// Use binary search to find the optimal offset b
	const float w_pos = w_p * n_pos;
	const float v = w_pos / (w_pos + w_n * (d.size()-n_pos));
	const float o = log(1-v+1e-10) - log(v+1e-10);
	float b0 = d.minCoeff()-o, b1 = d.maxCoeff()-o;
	while( b0*(1+1e-5) < b1 ) {
		float b = (b0+b1) / 2.0;
		float v = (w / (1.0 + (d.array()-b).exp())).sum();
		if( fabs(w_pos - v) < 1e-4*(fabs(w_pos)+fabs(v)) )
			break;
		if( v > w_pos )
			b1 = b;
		else
			b0 = b;
	}
	return (b0+b1)/2.0;
}
VectorXf WeightedLogisticObjective::gradient( const VectorXf & d, const VectorXb & gt, float & e ) {
	const float n_pos = gt.cast<float>().array().sum();
	const float w_n = (n_pos+0.05) / (gt.size()+0.1);
	const float w_p = 1 - w_n;
	const ArrayXf w = gt.cast<float>().array()*(w_p-w_n)+w_n;
	const float b = optimizeB( d, gt, w_n, w_p );
	ArrayXf db = d.array()-b;
	// p = 1 / (1 + exp(b-d))
	ArrayXf p = 1.0 / (1.0 + db.exp());
	// Compute the sign of the likelihood (gt=1: b-d  gt=0: d-b)
	ArrayXf db_s = (2*gt.cast<float>().array()-1)*db;
	// Split the values into positive and negative values
	ArrayXf db_n = db_s.array().min(0), db_p = db_s.array().max(0);
	// Likelihood: -log(p) = log(1 + exp(db_s)) [which is numerically unstable]
	//                     = db_p + log( exp(-db_p) + exp(db_s-db_p) ) = db_p + log( exp(-db_p) + exp(db_n) ) [which is numerically stable]
	e = (w*( db_p + ((-db_p).exp() + db_n.exp()).log() )).sum();
	// Gradient is p - gt
	return -w*(p - gt.cast<float>().array());
}
float IOUObjective::optimizeB( const VectorXf & d, float n_pos ) {
	return 0;//LogisticObjective::optimizeB( d, n_pos );
}
VectorXf IOUObjective::gradient( const VectorXf & d, const VectorXb & gt, float & e ) {
	int n_pos = gt.cast<int>().array().sum();
	const float b = optimizeB( d, n_pos );
	ArrayXf db = d.array()-b;
	// p = 1 / (1 + exp(b-d))
	ArrayXf p = 1.0 / (1.0 + db.exp());
	// Compute the sign of the likelihood (gt=1: b-d  gt=0: d-b)
	ArrayXf p_pos = gt.cast<float>().array()*p, p_neg = (1-gt.cast<float>().array())*p;
	// Compute the IOU energy
	float i = p_pos.sum(), u = n_pos + p_neg.sum();
	e = -i / u;
	
	ArrayXf d_p = (1-p)*p / u;
	// Gradient is p - gt
	return - d_p * ( i/u - (1+i/u)*gt.cast<float>().array() );
}
