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
#include "qp.h"
#include "util.h"
#include <Eigen/Dense>

// #define VERBOSE
#ifdef VERBOSE
#include <iostream>
#endif

const int MAX_ITER = 1000;

struct ActiveSet {
	int n;       // Current size
	ArrayXb in; // in(i) : Is variable i in the active set
	ArrayXi A;  // All variables in the active set
	ActiveSet( int max_n=0 ):n(0),in(ArrayXb::Zero(max_n)),A(-ArrayXi::Ones(max_n)){}
	int add( int i ) { // Add variable i to the active set
		eassert( !in[i] );
		in[i] = true;
		A(n) = i;
		return n++;
	}
	void removeLast() { // Remove the last entry from the active set
		eassert( n > 0 );
		--n;
		eassert( in[A(n)] );
		in[A(n)] = false;
		A(n) = -1;
	}
};

template<typename T>
T dist( T a, T b ) {
	a = std::abs(a);
	b = std::abs(b);
	if (a > b) return a*std::sqrt(1 + (b/a)*(b/a));
	if (a < b) return b*std::sqrt(1 + (a/b)*(a/b));
	return a*std::sqrt(2.0);
}

template<typename T>
bool addConstraint( RMatrixX<T> & R, RMatrixX<T> & J, VectorX<T> d, ActiveSet & active_set, int l ) {
	const int n = R.rows();
	const int na = active_set.n;
	/* Givens rotation Vodoo */
	for( int j=n-1; j>na; j-- ) {
		T cc = d[j-1], ss = d[j];
		T h = dist( cc, ss );
		if( h == 0 )
			continue;
		d[j] = 0;
		ss /= h;
		cc /= h;
		if( cc < 0 ){
			cc =-cc;
			ss =-ss;
			d[j-1] = -h;
		}
		else
			d[j-1] = h;
		T xny = ss / (1. + cc);
		RowVectorX<T> t1 = J.row(j-1);
		RowVectorX<T> t2 = J.row(j);
		J.row(j-1) = t1*cc + t2*ss;
		J.row(j)   = xny * (t1 + J.row(j-1)) - t2;
	}
	active_set.add( l );
	R.col(na).head(na+1) = d.head(na+1);
	if( na > 0 && std::abs(d[na]) <= std::numeric_limits<T>::epsilon() * R.diagonal().head(na).array().abs().maxCoeff() )
		// Degenerate R
		return false;
	return true;
}

template<typename T>
void deleteConstraint( RMatrixX<T> & R, RMatrixX<T> & J, VectorX<T> & u, ActiveSet & active_set, int l ) {
	const int n = active_set.n;
	// Find the element to remove
	int qq = -1;
	for( int i=0; i<n; i++ )
		if( active_set.A[i] == l ){
			qq = i;
			break;
		}
	eassert( qq >= 0 );
	
	// Move all elements that follow down by ones and update the QR factorization
	for( int i=qq; i < n-1; i++ ) {
		u[i] = u[i+1];
		R.col(i) = R.col(i+1);
		active_set.A[i] = active_set.A[i+1];
	}
	// And remove the last element from the active set
	R.col(n-1).setZero();
	u[n-1] = 0;
	active_set.A[n-1] = l;
	active_set.removeLast();
	/* Some more givens rotation Vodoo to restore R and J */
	for( int j = qq; j < n-1; j++ ) {
		T cc = R(j,j), ss = R(j+1,j);
		T h = dist( cc, ss );
		if( h == 0 )
			continue;
		R(j+1,j) = 0;
		ss /= h;
		cc /= h;
		if( cc < 0 ){
			cc =-cc;
			ss =-ss;
			R(j,j) = -h;
		}
		else
			R(j,j) = h;
		T xny = ss / (1. + cc);
		const int nr = n-j-1;
		if( nr>0 ) {
			RowVectorX<T> t1 = R.row(j).segment(j+1,nr);
			RowVectorX<T> t2 = R.row(j+1).segment(j+1,nr);
			R.row(j).segment(j+1,nr)   = t1*cc + t2*ss;
			R.row(j+1).segment(j+1,nr) = xny * (t1 + R.row(j).segment(j+1,nr)) - t2;
		}
		{
			RowVectorX<T> t1 = J.row(j);
			RowVectorX<T> t2 = J.row(j+1);
			J.row(j)   = t1*cc + t2*ss;
			J.row(j+1) = xny * (t1 + J.row(j)) - t2;
		}
	}
}

// Solves a strictly convex QP of the form
//  minimize_x  x' Q x  +  c' x
//  subject to  A x <= b
template<typename T>
VectorX<T> qp( const RMatrixX<T> & Q, const VectorX<T> & c, const RMatrixX<T> & A, const VectorX<T> & b ) {
#ifdef VERBOSE
	printf("=================== SOLVING THE QP ===================\n" );
#endif
	const int n = c.size(), m = b.size();
	const T inf = std::numeric_limits<T>::infinity();
	
	if( Q.cols() != n || Q.rows() != n )
		throw std::invalid_argument( "Expected quadratic matrix Q of size n x n, where n=|c|!" );
	if( A.cols() != n || A.rows() != m )
		throw std::invalid_argument( "Expected inequality constraint matrix A of size |b| x n!" );
	
	RMatrixX<T> R = RMatrixX<T>::Zero(n,n), J = RMatrixX<T>::Zero(n,n);
	ActiveSet active_set(m);
	VectorX<T> x = VectorX<T>::Zero(n), s = VectorX<T>::Zero(m), u = VectorX<T>::Zero(m);
	
	/*
	 * Preprocessing phase
	 */
	
	/* decompose the matrix Q in the form LL^T */
	LLT<RMatrixX<T>,Lower> chol(Q);
	
	/* compute the inverse of the factorized matrix Q^-1, this is the initial value for J */
	// J = L^-1
	J = chol.matrixL().solve(RMatrixX<T>::Identity(n,n));

	/*
	 * Find the unconstrained minimizer x, which is a feasible point in the dual space
	 * x = -Q^-1 * c
	 */
	x = -chol.solve(c);
	for( int iter=0; iter < MAX_ITER; iter++ ) {
		
		/* step 1: choose a violated constraint */
		
		/* compute s(x) = - A * x + b for all elements of K \ A */
		s = -A * x + b;
		
#ifdef VERBOSE
		printf("Iteration %d   %f %f\n", iter, std::abs(s.array().min(0.f).sum())/m, std::numeric_limits<T>::epsilon() );
#endif
		
		/* Are there any more infeasibilities? |s|_1 < EPS*cond*100 */
		if ( -s.minCoeff() <= 100*std::numeric_limits<T>::epsilon() )
			return x;
		
		/* Step 2: check for feasibility and determine a new S-pair */
		int new_constraint = 0;
		T ss = VectorX<T>(s.array() * ((T)1-active_set.in.cast<T>())).minCoeff( &new_constraint );
		
		/* Are all constraints feasible? */
		if (ss >= 0.0) return x;
		
		/* Get the new constraint */
		VectorX<T> np = -A.row(new_constraint).transpose();
		
#ifdef VERBOSE
		printf("New constraint %d / %d\n", new_constraint, m);
#endif
		/* Step 2a: determine step direction and take partial steps */
		/* compute z = H np: the step direction in the primal space (through J, see the paper) */
		T unA = 0; // u(nA)
		while(1) {
			/* Get the number of active constraints */
			const int nA = active_set.n;
			
			VectorX<T> d = J * np;
			VectorX<T> z = J.bottomRows(d.size()-nA).transpose() * d.tail(d.size()-nA);
			
			/* compute N* np (if q > 0): the negative of the step direction in the dual space */
			VectorX<T> r = R.topLeftCorner(nA,nA).template triangularView<Upper>().solve(d.head(nA));
			
			/* Step 2b: compute step length t1 before violating dual feasibility */
			int l = -1;
			T t1 = inf;
			if( r.size() > 0 ) {
				t1 = (r.array() > 0).select( u.head(nA).array() / r.array(), inf ).minCoeff( &l );
				l = active_set.A[l];
			}
			
			/* Compute t2: full step length (minimum step in primal space such that the constraint ip becomes feasible */
			T t2 = inf;
			if (std::abs(z.dot(z)) > std::numeric_limits<T>::epsilon()) // i.e. z != 0
				t2 = -ss / z.dot(np);
			
			/* the step is chosen as the minimum of t1 and t2 */
			T t = std::min<T>(t1, t2);
			
// #ifdef VERBOSE
// 			printf("  t = %f  %f %f    l = %d\n", t, t1, t2, l );
// 			std::cout<<"  ss = "<<ss<<"  z'np = "<<z.dot(np)<<std::endl;
// 			std::cout<<"  x = "<<x.transpose()<<std::endl;
// 			std::cout<<"  s = "<<s.transpose()<<std::endl;
// 			std::cout<<"  np = "<<np.transpose()<<std::endl;
// 			std::cout<<"  u = "<<u.head(nA).transpose()<<std::endl;
// 			std::cout<<"  r = "<<r.transpose()<<std::endl;
// 			std::cout<<"  z = "<<z.transpose()<<std::endl;
// 			std::cout<<"  d = "<<d.transpose()<<std::endl;
// 			std::cout<<"  R = "<<R.topLeftCorner(nA,nA)<<std::endl;
// 			std::cout<<"  J = "<<J<<std::endl;
// #endif
			
			/* Step 2c: determine new S-pair and take step: */
			
			/* case (i): no step in primal or dual space */
			if (t >= inf) /* QPP is infeasible */
				return VectorX<T>::Constant( n, inf );
			
			// Dual step
			u.head(nA) -= t * r;
			unA += t;
			
			// Primal step
			if( t2 < inf ) {
				x += t * z;
				ss = -A.row(new_constraint) * x + b(new_constraint);
			}
			
			/* case (ii): a patial step has taken */
			if (t2 > t1) {
#ifdef VERBOSE
				printf("  * delete %d\n", l );
#endif
				/* drop constraint l from the active set A */
				deleteConstraint<T>(R, J, u, active_set, l);
			}
			else 
				break;
		}
		/* case (iii): step in primal and dual space */
		/* full step has been taken */
		/* add constraint ip to the active set*/
#ifdef VERBOSE
		printf("  * add %d\n", new_constraint );
#endif
		if (!addConstraint<T>(R, J, J*np, active_set, new_constraint))
			throw std::invalid_argument( "Degenerate R matrix, going to give up!" );
		u[active_set.n-1] = unA; // Update the lagrangian variable
	}
	return x;
}


// Solves a strictly convex sparse QP of the form
//  minimize_x  x' diag(Q) x  +  c' x
//  subject to  A x <= b
template<typename T>
VectorX<T> sparseQp( const VectorX<T> & Q, const VectorX<T> & c, const SRMatrixX<T> & A, const VectorX<T> & b ) {
	const int n = c.size(), m = b.size();
	const T inf = std::numeric_limits<T>::infinity();
	
	if( Q.size() != n )
		throw std::invalid_argument( "Expected vector Q of size n=|c|!" );
	if( A.cols() != n || A.rows() != m )
		throw std::invalid_argument( "Expected inequality constraint matrix A of size |b| x n!" );
	
	RMatrixX<T> R = RMatrixX<T>::Zero(n,n), J = RMatrixX<T>::Zero(n,n);
	ActiveSet active_set(m);
	VectorX<T> x = VectorX<T>::Zero(n), s = VectorX<T>::Zero(m), u = VectorX<T>::Zero(m);
	
	/*
	 * Preprocessing phase
	 */
	
	/* decompose the matrix Q in the form LL^T */
	const VectorX<T> L_inv = 1.0 / Q.array().sqrt();
	
	/* compute the inverse of the factorized matrix Q^-1, this is the initial value for J */
	// J = L^-1
	J = L_inv.asDiagonal();
	
	/*
	 * Find the unconstrained minimizer x, which is a feasible point in the dual space
	 * x = -Q^-1 * c
	 */
	x = -c.array() / Q.array();
	for( int iter=0; iter < MAX_ITER; iter++ ) {
		
		/* step 1: choose a violated constraint */
		
		/* compute s(x) = - A * x + b for all elements of K \ A */
		s = -A * x + b;
		
#ifdef VERBOSE
		printf("Iteration %d  %f %f\n", iter, std::abs(s.array().min(0.f).sum())/m, std::numeric_limits<T>::epsilon());
#endif
		
		/* Are there any more infeasibilities? |s|_1 < EPS*cond*100 */
		if ( -s.minCoeff() <= std::numeric_limits<T>::epsilon() )
			return x;
		
		/* Step 2: check for feasibility and determine a new S-pair */
		int new_constraint = 0;
		T ss = VectorX<T>(s.array() * ((T)1-active_set.in.cast<T>())).minCoeff( &new_constraint );
		
		/* Are all constraints feasible? */
		if (ss >= 0.0) return x;
		
		/* Get the new constraint */
		VectorX<T> np = -RowVectorX<T>(A.row(new_constraint)).transpose();
		
#ifdef VERBOSE
		printf("New constraint %d / %d\n", new_constraint, m);
#endif
		/* Step 2a: determine step direction and take partial steps */
		/* compute z = H np: the step direction in the primal space (through J, see the paper) */
		T unA = 0; // u(nA)
		while(1) {
			/* Get the number of active constraints */
			const int nA = active_set.n;
			
			VectorX<T> d = J * np;
			VectorX<T> z = J.bottomRows(d.size()-nA).transpose() * d.tail(d.size()-nA);
			
			/* compute N* np (if q > 0): the negative of the step direction in the dual space */
			VectorX<T> r = R.topLeftCorner(nA,nA).template triangularView<Upper>().solve(d.head(nA));
			
			/* Step 2b: compute step length t1 before violating dual feasibility */
			int l = -1;
			T t1 = inf;
			if( r.size() > 0 ) {
				t1 = (r.array() > 0).select( u.head(nA).array() / r.array(), inf ).minCoeff( &l );
				l = active_set.A[l];
			}
			
			/* Compute t2: full step length (minimum step in primal space such that the constraint ip becomes feasible */
			T t2 = inf;
			if (std::abs(z.dot(z)) > std::numeric_limits<T>::epsilon()) // i.e. z != 0
				t2 = -ss / z.dot(np);
			
			/* the step is chosen as the minimum of t1 and t2 */
			T t = std::min<T>(t1, t2);
			
#ifdef VERBOSE
			printf("  t = %f  %f %f    l = %d\n", t, t1, t2, l );
			std::cout<<"  ss = "<<ss<<"  z'np = "<<z.dot(np)<<std::endl;
			std::cout<<"  x = "<<x.transpose()<<std::endl;
			std::cout<<"  s = "<<s.transpose()<<std::endl;
			std::cout<<"  np = "<<np.transpose()<<std::endl;
			std::cout<<"  u = "<<u.head(nA).transpose()<<std::endl;
			std::cout<<"  r = "<<r.transpose()<<std::endl;
			std::cout<<"  z = "<<z.transpose()<<std::endl;
			std::cout<<"  d = "<<d.transpose()<<std::endl;
			std::cout<<"  R = "<<R.topLeftCorner(nA,nA)<<std::endl;
			std::cout<<"  J = "<<J<<std::endl;
#endif
			
			/* Step 2c: determine new S-pair and take step: */
			
			/* case (i): no step in primal or dual space */
			if (t >= inf) /* QPP is infeasible */
				return VectorX<T>::Constant( n, inf );
			
			// Dual step
			u.head(nA) -= t * r;
			unA += t;
			
			// Primal step
			if( t2 < inf ) {
				x += t * z;
				ss = np.dot( x ) + b(new_constraint);
			}
			
			/* case (ii): a patial step has taken */
			if (t2 > t1) {
#ifdef VERBOSE
				printf("  * delete %d\n", l );
#endif
				/* drop constraint l from the active set A */
				deleteConstraint<T>(R, J, u, active_set, l);
			}
			else
				break;
		}
		/* case (iii): step in primal and dual space */
		/* full step has been taken */
		/* add constraint ip to the active set*/
#ifdef VERBOSE
		printf("  * add %d\n", new_constraint );
#endif
		if (!addConstraint<T>(R, J, J*np, active_set, new_constraint))
			throw std::invalid_argument( "Degenerate R matrix, going to give up!" );
		u[active_set.n-1] = unA; // Update the lagrangian variable
	}
	return x;
}

VectorXf qp( const RMatrixXf & Q, const VectorXf & c, const RMatrixXf & A, const VectorXf & b ) {
	return qp<float>( Q, c, A, b );
}
VectorXd qp( const RMatrixXd & Q, const VectorXd & c, const RMatrixXd & A, const VectorXd & b ) {
	return qp<double>( Q, c, A, b );
}
VectorXf sparseQp( const VectorXf & Q, const VectorXf & c, const SRMatrixXf & A, const VectorXf & b ) {
	return sparseQp<float>( Q, c, A, b );
}
VectorXd sparseQp( const VectorXd & Q, const VectorXd & c, const SRMatrixXd & A, const VectorXd & b ) {
	return sparseQp<double>( Q, c, A, b );
}

