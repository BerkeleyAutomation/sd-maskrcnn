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
#include "util.h"
#include "util/eigen.h"
#include "gop.h"
#include "util/graph.h"
#include "util/eigen.h"
#include "util/rasterize.h"
#include "util/qp.h"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

// For numpy 1.6 define NPY_ARRAY_*
#if NPY_API_VERSION < 0x00000007
#define NPY_ARRAY_C_CONTIGUOUS NPY_C_CONTIGUOUS
#define NPY_ARRAY_ALIGNED      NPY_ALIGNED
#endif


struct Edge_pickle : pickle_suite
{
	static tuple getinitargs(const Edge& e) {
		return make_tuple(e.a,e.b);
	}
};

template <typename SCALAR>
struct NumpyEquivalentType {};

template <> struct NumpyEquivalentType<double> {enum { type_code = NPY_DOUBLE };};
template <> struct NumpyEquivalentType<float> {enum { type_code = NPY_FLOAT };};
template <> struct NumpyEquivalentType<int64_t> {enum { type_code = NPY_INT64 };};
template <> struct NumpyEquivalentType<uint64_t> {enum { type_code = NPY_UINT64 };};
template <> struct NumpyEquivalentType<int32_t> {enum { type_code = NPY_INT32 };};
template <> struct NumpyEquivalentType<uint32_t> {enum { type_code = NPY_UINT32 };};
template <> struct NumpyEquivalentType<int16_t> {enum { type_code = NPY_INT16 };};
template <> struct NumpyEquivalentType<uint16_t> {enum { type_code = NPY_UINT16 };};
template <> struct NumpyEquivalentType<int8_t> {enum { type_code = NPY_INT8 };};
template <> struct NumpyEquivalentType<uint8_t> {enum { type_code = NPY_UINT8  };};
template <> struct NumpyEquivalentType<bool> {enum { type_code = NPY_BOOL  };};

template< typename T >
void copyMat( T * dst, const T* src, int cols, int rows, bool transpose ) {
	if( !transpose )
		memcpy( dst, src, cols*rows*sizeof(T) );
	else {
		for( int j=0; j<rows; j++ )
			for( int i=0; i<cols; i++ )
				dst[ j*cols+i ] = src[ i*rows+j ];
	}
}
// Reduce the number of pointer-to-function warnings (since disabling them seems not possible)
static PyArrayObject * PyArrayObject_New( int n, npy_intp * shape, int type ) {
	return (PyArrayObject*)PyArray_SimpleNew( n, shape, type );
}
static int PyArray_SIZE2( PyArrayObject * array ) {
	return PyArray_SIZE(array);
}
template<class MatType>
struct EigenMatrixToPython {
	static PyObject* convert(const MatType& mat) {
		typedef typename MatType::Scalar T;
		PyArrayObject* python_array;
		if( MatType::ColsAtCompileTime==1 || MatType::RowsAtCompileTime==1 ) {
			npy_intp shape[1] = { mat.rows()*mat.cols() };
			python_array = PyArrayObject_New(1, shape, NumpyEquivalentType<T>::type_code);
		}
		else {
			npy_intp shape[2] = { mat.rows(), mat.cols() };
			python_array = PyArrayObject_New(2, shape, NumpyEquivalentType<T>::type_code);
		}
		copyMat( (T*)PyArray_DATA(python_array), mat.data(), mat.rows(), mat.cols(), !(MatType::Flags & RowMajor) );
		return (PyObject*)python_array;
	}
};

template<typename MatType>
struct EigenMatrixFromPython {
	typedef typename MatType::Scalar T;
	EigenMatrixFromPython() {
		converter::registry::push_back(&convertible, &construct, type_id<MatType>());
	}
	static void* convertible(PyObject* obj_ptr) {
		const int R = MatType::RowsAtCompileTime;
		const int C = MatType::ColsAtCompileTime;
		PyArrayObject *array = reinterpret_cast<PyArrayObject*>(obj_ptr);
		if (!PyArray_Check(obj_ptr) || PyArray_NDIM(array) > 2 ||  PyArray_NDIM(array) <= 0 || PyArray_TYPE(array) != NumpyEquivalentType<T>::type_code)
			return 0;
		if( R==1 || C==1 ) { // Eigen Vector
			if ( PyArray_NDIM(array)==2 && PyArray_DIMS(array)[0]>1 && PyArray_DIMS(array)[1]>1 )
				return 0;
			if ( PyArray_NDIM(array)==1 && R*C > 0 && R*C != PyArray_DIMS(array)[0] )
				return 0;
		}
		else if ( R > 1 && PyArray_DIMS(array)[0] != R )
			return 0;
		else if ( C > 1 && PyArray_NDIM(array)<2 && PyArray_DIMS(array)[1] != C )
			return 0;
		return obj_ptr;
	}
	static void construct(PyObject* obj_ptr, converter::rvalue_from_python_stage1_data* data) {
		const int R = MatType::RowsAtCompileTime;
		const int C = MatType::ColsAtCompileTime;
		
		PyArrayObject *array = reinterpret_cast<PyArrayObject*>(obj_ptr);
		int flags = PyArray_FLAGS(array);
		if (!(flags & NPY_ARRAY_C_CONTIGUOUS) || !(flags & NPY_ARRAY_ALIGNED))
			throw std::invalid_argument("Contiguous and aligned array required!");
		const int ndims = PyArray_NDIM(array);
		
		const int dtype_size = (PyArray_DESCR(array))->elsize;
		const int s1 = PyArray_STRIDE(array, 0), s2 = ndims > 1 ? PyArray_STRIDE(array, 1) : 0;
		
		int nrows=1, ncols=1;
		if( R==1 || C==1 ) { // Vector
			nrows = R==1 ? 1 : PyArray_SIZE2(array);
			ncols = C==1 ? 1 : PyArray_SIZE2(array);
		}
		else {
			nrows = (R == Dynamic) ? PyArray_DIMS(array)[0] : R;
			if ( ndims > 1 )
				ncols = (R == Dynamic) ? PyArray_DIMS(array)[1] : R;
		}
		T* raw_data = reinterpret_cast<T*>(PyArray_DATA(array));
		
		typedef Map< Matrix<T,Dynamic,Dynamic,RowMajor>,Aligned,Stride<Dynamic, Dynamic> > MapType;
		
		void* storage=((converter::rvalue_from_python_storage<MatType>*)(data))->storage.bytes;
		new (storage) MatType;
		MatType* emat = (MatType*)storage;
		*emat = MapType(raw_data, nrows, ncols,Stride<Dynamic, Dynamic>(s1/dtype_size, s2/dtype_size));
		data->convertible = storage;
	}
};

// #define EIGEN_MATRIX_CONVERTER(Type) EigenMatrixFromPython<Type>(); to_python_converter<Type, EigenMatrixToPython<Type> >();
#define EIGEN_MATRIX_CONVERTER(Type) EigenMatrixFromPython<Type>(); to_python_converter<Type, EigenMatrixToPython<Type> >();// to_python_converter<const Type &, EigenMatrixToPython<Type> >();

#define MAT_CONV( N )\
EIGEN_MATRIX_CONVERTER( N ## d );\
EIGEN_MATRIX_CONVERTER( N ## f );\
EIGEN_MATRIX_CONVERTER( N ## i );\
EIGEN_MATRIX_CONVERTER( N ## u );\
EIGEN_MATRIX_CONVERTER( N ## s );\
EIGEN_MATRIX_CONVERTER( N ## us );\
EIGEN_MATRIX_CONVERTER( N ## i8 );\
EIGEN_MATRIX_CONVERTER( N ## u8 );\
EIGEN_MATRIX_CONVERTER( N ## b )


struct VecPolygons_pickle_suite : pickle_suite {
	static object getstate(const std::vector<Polygons>& vp) {
		std::stringstream ss;
		unsigned int s = vp.size();
		ss.write( (const char*)&s, sizeof(s) );
		for( Polygons p: vp ) {
			s = p.size();
			ss.write( (const char*)&s, sizeof(s) );
			for( Polygon v: p )
				saveMatrixX( ss, v );
		}
		std::string data = ss.str();
		return object( handle<>( PyBytes_FromStringAndSize( data.data(), data.size() ) ) );
	}
	static void setstate(std::vector<Polygons> & vp, const object & state) {
		if(!PyBytes_Check(state.ptr()))
			throw std::invalid_argument("Failed to unpickle, unexpected type!");
		std::stringstream ss( std::string( PyBytes_AS_STRING(state.ptr()), PyBytes_Size(state.ptr()) ) );
		unsigned int s;
		ss.read( (char*)&s, sizeof(s) );
		vp.resize( s );
		for( Polygons & p: vp ) {
			ss.read( (char*)&s, sizeof(s) );
			p.resize( s );
			for( Polygon & v: p )
				loadMatrixX( ss, v );
		}
	}
};

#if PY_MAJOR_VERSION >= 3
int init_numpy() { import_array(); }
#else
void init_numpy() { import_array(); }
#endif

void defineUtil() {
	init_numpy();
	boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
	
	ADD_MODULE(util);
	class_<Edge>("Edge")
	.def(init<int,int>())
	.def_readonly( "a", &Edge::a )
	.def_readonly( "b", &Edge::b )
	.def_pickle(Edge_pickle());
	
	class_< Edges >("Edges")
	.def(init<Edges>())
	.def(vector_indexing_suite<Edges>());
	
	def("qp",static_cast<VectorXf(*)(const RMatrixXf &, const VectorXf &, const RMatrixXf &, const VectorXf &)>(&qp));
	def("qp",static_cast<VectorXd(*)(const RMatrixXd &, const VectorXd &, const RMatrixXd &, const VectorXd &)>(&qp));
	
	// NOTE: When overloading functions always make sure to put the array/matrix function before the vector one
	MAT_CONV( MatrixX );
	MAT_CONV( RMatrixX );
	MAT_CONV( VectorX );
	MAT_CONV( ArrayXX );
	MAT_CONV( RArrayXX );
	MAT_CONV( ArrayX );
	
	// Datastructures
	class_< std::vector<int> >("VecInt").def( vector_indexing_suite< std::vector<int> >() );
	class_< std::vector<float> >("VecFloat").def( vector_indexing_suite< std::vector<float> >() );
	
	class_<Polygons>("Polygons").def( vector_indexing_suite<Polygons,true >() );
	class_< std::vector<Polygons> >("VecPolygons").def( vector_indexing_suite< std::vector<Polygons> >() )
	.def_pickle(VecPolygons_pickle_suite());
	
	// Rasterize
	def("rasterize",static_cast<RMatrixXf(*)(const Polygon &)>(&rasterize));
	def("rasterize",static_cast<RMatrixXf(*)(const Polygons &)>(&rasterize));
}
