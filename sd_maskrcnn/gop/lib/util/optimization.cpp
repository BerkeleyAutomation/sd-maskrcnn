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
#include "optimization.h"
#include "lbfgs.h"
// #include "lbfgsb.h"
#include <cstdio>
#include <iostream>

EnergyFunction::~EnergyFunction() {
}
VectorXf EnergyFunction::optimizationInitialGuess() const {
	return initialGuess();
}
VectorXf EnergyFunction::optimizationGradient(const VectorXf &x, float &e) const {
	return gradient(x, e);
}
VectorXf EnergyFunction::optimizationTransformResult(const VectorXf &x) const {
	return x;
}
static ArrayXf inv_softmax0( const ArrayXf & x ) {
	return x + (1 - (-x).exp()).log();
}
static ArrayXf softmax0( const ArrayXf & x ) {
	return x.max(0) + ((-x).min(0).exp()+x.min(0).exp()).log();
}
static ArrayXf d_softmax0( const ArrayXf & x ) {
	ArrayXf ex0 = (-x).min(0).exp(), ex1 = x.min(0).exp();
	return ex1 / (ex0 + ex1);
}
VectorXf PositiveConstrainedEnergyFunction::optimizationInitialGuess() const {
	VectorXf x = initialGuess();
	x.head(n_positive_) = inv_softmax0(x.head(n_positive_).array().max(1e-6));
	return x;
}
VectorXf PositiveConstrainedEnergyFunction::optimizationGradient(const VectorXf &x, float &e) const {
	VectorXf xx = x;
	xx.head(n_positive_) = softmax0(x.head(n_positive_));
	VectorXf g = gradient(xx, e);
	g.head(n_positive_).array() *= d_softmax0(x.head(n_positive_));
	return g;
}
PositiveConstrainedEnergyFunction::PositiveConstrainedEnergyFunction(int n_positive) : n_positive_(n_positive) {
}
VectorXf PositiveConstrainedEnergyFunction::optimizationTransformResult(const VectorXf &x) const {
	VectorXf xx = x;
// 	xx.head(n_positive_) = x.head(n_positive_).array().exp();
	xx.head(n_positive_) = softmax0(x.head(n_positive_));
	return xx;
}

namespace optimizePrivate{
	static int _progress( void *instance, const lbfgsfloatval_t *x, const lbfgsfloatval_t *g, const lbfgsfloatval_t fx, const lbfgsfloatval_t xnorm, const lbfgsfloatval_t gnorm, const lbfgsfloatval_t step, int n, int k, int ls ) {
		printf("Iteration %d:\n", k);
		printf("  fx = %f, xnorm = %f, gnorm = %f, step = %f\n", fx, xnorm, gnorm, step );
		std::cout<<"  x = "<<reinterpret_cast<const EnergyFunction*>(instance)->optimizationTransformResult( VectorXf::Map(x,n) ).transpose()<<std::endl;
		std::cout<<"  g = "<<VectorXf::Map(g,n).transpose()<<std::endl;
		return 0;
	}
	static lbfgsfloatval_t _evaluate( void *instance, const lbfgsfloatval_t *x, lbfgsfloatval_t *g, const int n, const lbfgsfloatval_t step ) {
		float r=0;
		VectorXf::Map(g,n) = reinterpret_cast<const EnergyFunction*>(instance)->optimizationGradient( VectorXf::Map(x,n), r );
		return r;
	}
}

VectorXf minimizeLBFGS(const EnergyFunction &f, float & e, int verbose ) {
	using namespace optimizePrivate;
	
	VectorXf r = f.optimizationInitialGuess();
	const int N = r.size();
	
	lbfgsfloatval_t fx;
	lbfgsfloatval_t *m_x = lbfgs_malloc(N);
	
	std::copy( r.data(), r.data()+N, m_x );
	
	lbfgs_parameter_t param;
	lbfgs_parameter_init( &param );
	param.max_iterations = 100;
	
	int ret = lbfgs(N, m_x, &fx, _evaluate, (verbose>1)?_progress:NULL, const_cast<EnergyFunction*>(&f), &param);
	
	/* Report the result. */
	if( verbose>0 ) {
		printf("L-BFGS optimization terminated with status code = %d\n", ret);
		printf("  fx = %f\n", fx);
		std::cout<<"  x = "<<VectorXf::Map(m_x,N).transpose()<<std::endl;
	}
	// Store the result and clean up
	e = fx;
	r = VectorXf::Map( m_x, N );
	lbfgs_free( m_x );
	return f.optimizationTransformResult(r);
}
VectorXf minimizeLBFGS( const EnergyFunction & f, int verbose ) {
	float tmp;
	return minimizeLBFGS( f, tmp, verbose );
}
// namespace optimizePrivate{
// 	static double _evaluate_b( int n, const double *x, double *g, void * instance ) {
// 		float r=0;
// 		VectorXd::Map(g,n) = reinterpret_cast<const EnergyFunction*>(instance)->optimizationGradient( VectorXd::Map(x,n).cast<float>(), r ).cast<double>();
// 		return r;
// 	}
// }
// VectorXf minimizeLBFGSB( const EnergyFunction & f, const VectorXf & min, const VectorXf & max, float & e, int verbose ) {
// 	using namespace optimizePrivate;
// 	
// 	VectorXd x = f.optimizationInitialGuess().cast<double>();
// 	const int N = x.size();
// 	VectorXd l = -1e11*VectorXd::Ones(N), u = 1e11*VectorXd::Ones(N);
// 	l.head( min.size() ) = min.cast<double>();
// 	u.head( max.size() ) = max.cast<double>();
// 	printf("Minimize\n");
// 	lbfgsb( N, 5, x.data(), l.data(), u.data(), _evaluate_b, (void*)&f );
// 	return x.cast<float>();
// }
// VectorXf minimizeLBFGSB( const EnergyFunction & f, const VectorXf & min, const VectorXf & max, int verbose ) {
// 	float tmp;
// 	return minimizeLBFGSB( f, min, max, tmp, verbose );
// }
float gradCheck( const EnergyFunction &f, const VectorXf &x, int verbose ) {
	const float EPS = 1e-3;
	
	float e;
	VectorXf g = f.optimizationGradient( x, e );
	VectorXf gg = 1*g;
	for( int i=0; i<x.size(); i++ ) {
		VectorXf d = VectorXf::Zero( x.size() );
		d[i] = 1;
		
		float e1 = 0, e2 = 0;
		f.optimizationGradient( x+EPS*d, e1 );
		f.optimizationGradient( x-EPS*d, e2 );
		gg[i] = (e1-e2)/(2*EPS);
	}
	float rel_e = ((g-gg).array().abs() / (g.array().abs()+gg.array().abs()).max(1)).maxCoeff();
// 	float rel_e = (g-gg).array().abs().maxCoeff() / (g.array().abs()+gg.array().abs()+1e-3).maxCoeff();
	if( verbose ) {
		printf("Grad test  %f  %f\n", (g-gg).norm(), rel_e);
		std::cout<<"real  = "<<g.transpose()<<std::endl;
		std::cout<<"fdiff = "<<gg.transpose()<<std::endl;
		std::cout<<"rel e = "<<((g-gg).array().abs() / (g.array().abs()+gg.array().abs()+1e-3)).transpose()<<std::endl;
	}
	return rel_e;
}
float gradCheck( const EnergyFunction &f, int verbose ) {
	return gradCheck( f, f.optimizationInitialGuess(), verbose );
}



LBFGS::LBFGS( float eps, int max_iter, int max_line_search, int max_history, bool restart ):eps_(eps),max_iter_(max_iter), max_line_search_(max_line_search), max_history_(max_history), restart_(restart) {
}
void LBFGS::progress(const EnergyFunction & f, int n_it, const VectorXf & x, const VectorXf & g, float fx, float step ) const {
	printf("Iteration %d:\n", n_it);
	printf("  fx = %f, xnorm = %f, gnorm = %f, step = %f\n", fx, x.norm(), g.norm(), step );
	std::cout<<"  x = "<<f.optimizationTransformResult(x).transpose()<<std::endl;
	std::cout<<"  g = "<<g.transpose()<<std::endl;
}

float LBFGS::backtrackWolfe( const EnergyFunction & f, VectorXf & x, VectorXf & g, float & fx, const VectorXf & z, int k ) const {
	const float ftol = 1e-4, gtol = 0.9;
	// Backtracking using Wolfe's first condition
	float des = -z.dot(g);
	float step = k ? 1.0 : (1.0 / g.norm());
	bool back = 0;
	for( int ls=0; ls<max_line_search_; ls++ ) {
		const VectorXf nx = x - step * z;
		float nfx;
		VectorXf ng = f.optimizationGradient( nx, nfx );
		
		if( nfx <= fx + ftol * step * des ) { // First Wolfe condition
			if( (-z.dot(ng) >= gtol * des) || back ) { // Second Wolfe condition
				x = nx;
				g = ng;
				fx = nfx;
				return step;
			}
			else {
				step *= 2.1;
			}
		}
		else {
			step *= 0.5;
			back = 1;
		}
	}
	return -1;
}
//
//float LBFGS::backtrackmMoreThuente( const EnergyFunction & f, VectorXf & x, VectorXf & g, float & finit, const VectorXf & z, int k ) const {
//	const float ftol = 1e-4, xtol = 1e-10, gtol = 0.9;
//	
//	const float dginit = g.dot(z);
//	const float dgtest = dginit * ftol;
//	int stage1 = 1, back = 0;
//	
//	float stx=0, sty=0, fx=finit, fy=finit, dgx=dginit, dgy=dginit;
//	
//	float dgtest = ftol * dginit;
//	float width = 1e10;
//	float prev_width = 2.0 * width;
//	
//	for( int count=0;; ) {
//		// Min max step correspond to the interval of uncertenty
//		float stmin=0, stmax=0;
//		if( back ) {
//			stmin = std::min(stx,sty);
//			stmax = std::max(stx,sty);
//		}
//		else {
//			stmin = stx;
//			stmax = step + 4.0 * (step - stx);
//		}
//		// Should we just give up?
//		if( back && ((step <= stmin || stmax <= step) || max_line_search_ <= count+1 || uinfo != 0) || (stmax - stmin <= xtol * stmax) ) {
//			step = stx;
//			x -= step * z;
//			g = f.optimizationGradient( x, fx );
//			return -1;
//		}
//		
//		const VectorXf nx = x - step * z;
//		float nfx;
//		VectorXf ng = f.optimizationGradient( nx, nfx );
//		float dg = ng.dot(z);
//		float ftest = finit + step * dgtest;
//		
//		if (nfx <= ftest && fabs(dg) <= gtol * (-dginit))
//			return step;
//		
//		/*
//		 In the first stage we seek a step for which the modified
//		 function has a nonpositive value and nonnegative derivative.
//		 */
//		if( stage1 && nfx <= ftest && std::min(ftol,gtol) * dginit <= dg )
//			stage1 = 0;
//		
//		
//		/*
//		 A modified function is used to predict the step only if
//		 we have not obtained a step for which the modified
//		 function has a nonpositive function value and nonnegative
//		 derivative, and if a lower function value has been
//		 obtained but the decrease is not sufficient.
//		 */
//		if (stage1 && ftest < nfx && nfx <= fx) {
//			/* Define the modified function and derivative values. */
//			float fm = nfx - step * dgtest;
//			float fxm = fx - stx * dgtest;
//			float fym = fy - sty * dgtest;
//			float dgm = dg - dgtest;
//			float dgxm = dgx - dgtest;
//			float dgym = dgy - dgtest;
//			
//			/*
//			 Call update_trial_interval() to update the interval of
//			 uncertainty and to compute the new step.
//			 */
//			uinfo = update_trial_interval(stx, fxm, dgxm,
//										  sty, fym, dgym,
//										  step, fm, dgm,
//										  stmin, stmax, brack );
//			
//			/* Reset the function and gradient values for f. */
//			fx = fxm + stx * dgtest;
//			fy = fym + sty * dgtest;
//			dgx = dgxm + dgtest;
//			dgy = dgym + dgtest;
//		} else {
//			/*
//			 Call update_trial_interval() to update the interval of
//			 uncertainty and to compute the new step.
//			 */
//			uinfo = update_trial_interval(stx, fx, dgx,
//										  sty, fy, dgy,
//										  step, nfx, dg,
//										  stmin, stmax, brack
//										  );
//		}
//		if( back ) {
//			if (0.66 * prev_width <= fabs(sty - stx)) {
//				step = stx + 0.5 * (sty - stx);
//			}
//			prev_width = width;
//			width = fabs(sty - stx);
//		}
//	}
//	
//	/* Initialize local variables. */
//	dgtest = param->ftol * dginit;
//	width = param->max_step - param->min_step;
//	prev_width = 2.0 * width;
//	
//	/*
//	 The variables stx, fx, dgx contain the values of the step,
//	 function, and directional derivative at the best step.
//	 The variables sty, fy, dgy contain the value of the step,
//	 function, and derivative at the other endpoint of
//	 the interval of uncertainty.
//	 The variables stp, f, dg contain the values of the step,
//	 function, and derivative at the current step.
//	 */
//	stx = sty = 0.;
//	fx = fy = finit;
//	dgx = dgy = dginit;
//	
//	for (;;) {
//
//		
//		
//		/*
//		 Force a sufficient decrease in the interval of uncertainty.
//		 */
//		if (brackt) {
//			if (0.66 * prev_width <= fabs(sty - stx)) {
//				*stp = stx + 0.5 * (sty - stx);
//			}
//			prev_width = width;
//			width = fabs(sty - stx);
//		}
//	}
//	
//	return LBFGSERR_LOGICERROR;
//}

VectorXf LBFGS::minimize( const EnergyFunction & f, float & fx, int verbose ) const {
	VectorXf x = f.optimizationInitialGuess();
	VectorXf g = f.optimizationGradient( x, fx );
	if( verbose>1 )
		progress( f, 0, x, g, fx, 1 );
	
	VectorXf px, pg;
	MatrixXf hdx(x.rows(), max_history_ ), hdg(g.rows(), max_history_ );
	
	for( int i=0,k=0; i<max_iter_; i++ ) {
		const float rel_eps = eps_ * std::max(1.f, x.norm());
		
		// Check the norm of the gradient against the convergence threshold
		if( g.norm() < rel_eps ) {
			if( verbose>0 ){
				printf("LBFGS converged gnorm = %f, iter = %d\n", g.norm(), i );
				printf("  fx = %f\n", fx);
				std::cout<<"  x = "<<f.optimizationTransformResult(x).transpose()<<std::endl;
				
			}
			return x;
		}
		VectorXf z = g;
		
		if( k && max_history_ ) {
			const int h = std::min( k, max_history_ );
			const int end = (k-1) % max_history_;
			hdx.col( end ) = x - px;
			hdg.col( end ) = g - pg;
			
			// Multiply by the hessian
			VectorXf p(h), a(h);
			for( int jj=0; jj<h; jj++ ) {
				const int j = (end - jj + max_history_) % max_history_;
				p[j] = 1.0 / hdx.col(j).dot( hdg.col(j) );
				a[j] = p[j] * hdx.col(j).dot( z );
				z -= a[j] * hdg.col(j);
			}
			// Scaling
			z *= hdx.col(end).dot( hdg.col(end) ) / hdg.col(end).dot( hdg.col(end) );
			
			for( int jj=0; jj<h; jj++ ) {
				const int j = (end+jj+1)%h;
				const float b = p[j] * hdg.col(j).dot(z);
				z += hdx.col(j) * (a[j] - b );
			}
		}
		// Save the previous x and g
		px = x;
		pg = g;
		
		// Check if z is a descend direction
		if( z.dot(g) < 1e-4 * rel_eps ) {
			if( verbose>1 )
				printf("LBFGS bad hessian after %d iterations\n", i );
			if( restart_ ) {
				if( verbose>1 )
					printf("Restarting\n");
				k = 0;
				z = g;
			}
			else
				return x;
		}
		float step = backtrackWolfe( f, x, g, fx, z, k );
		if( step < 0 ) {
			if( verbose>0 ){
				printf("LBFGS linesearch failed, iter = %d\n", i );
				printf("  fx = %f\n", fx);
				std::cout<<"  x = "<<f.optimizationTransformResult(x).transpose()<<std::endl;
			}
			return x;
		}
		if( verbose )
			progress( f, i+1, x, g, fx, step );
	}
	if( verbose>0 ){
		printf("LBFGS maximal number of iterations reached, iter = %d\n", max_iter_ );
		printf("  fx = %f\n", fx);
		std::cout<<"  x = "<<f.optimizationTransformResult(x).transpose()<<std::endl;
	}
	return f.optimizationTransformResult(x);
}
VectorXf LBFGS::minimize(const EnergyFunction &f, int verbose) const {
	float tmp = 0;
	return minimize(f, tmp, verbose);
}


SGD::SGD( float alpha, int n_iter, int mini_batch_size ) : alpha_( alpha ), n_iter_(n_iter), mb_size_( mini_batch_size ) {
}
VectorXf SGD::minimize( const EnergyFunction &f, float &e, int verbose ) const {
	// For now just use GD
	VectorXf x = f.optimizationInitialGuess();
	for( int i=0; i<n_iter_; i++ ) {
		x -= alpha_ * f.optimizationGradient( x, e );
		if( verbose )
			printf("SGD[%d] = %f\n", i, e );
	}
	return f.optimizationTransformResult( x );
}
VectorXf SGD::minimize( const EnergyFunction &f, int verbose ) const {
	float e;
	return minimize( f, e, verbose );
}
