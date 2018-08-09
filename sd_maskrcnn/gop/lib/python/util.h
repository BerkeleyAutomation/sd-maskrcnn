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
#include <boost/version.hpp>
#include <boost/python.hpp>
#include <boost/numpy.hpp>
#include <stdexcept>
#include <memory>
using namespace boost::python;
namespace np = boost::numpy;
#include "util/util.h"

// Make older boost versions happy
#if BOOST_VERSION < 105300
template<typename T> T* get_pointer(const std::shared_ptr<T>& p) { return p.get(); }
#endif

#define checkArray( f, T, min_dim, max_dim, cont ) {\
	if( f.get_dtype() != np::dtype::get_builtin<T>() )\
		throw std::invalid_argument( ((std::string)_xstr(f) + " @ " + FILE_AND_LINE)+"  :  "+(std::string)"Only '"+typeid(T).name()+"' array supported!" );\
	if( min_dim > f.get_nd() || f.get_nd() > max_dim )\
		throw std::invalid_argument( ((std::string)_xstr(f) + " @ " + FILE_AND_LINE)+"  :  "+std::to_string(min_dim)+"d "+(max_dim>min_dim?"- "+std::to_string(max_dim)+"d ":"")+"array required" );\
	if( cont && !(f.get_flags() & np::ndarray::C_CONTIGUOUS) )\
		throw std::invalid_argument( ((std::string)_xstr(f) + " @ " + FILE_AND_LINE)+"  :  "+"Contiguous array required!" );\
}

template<typename OBJ>
struct SaveLoad_pickle_suite : pickle_suite {
	static object getstate(const OBJ& obj) {
		std::stringstream ss;
		obj.save( ss );
		std::string data = ss.str();
		return object( handle<>( PyBytes_FromStringAndSize( data.data(), data.size() ) ) );
	}
	
	static void setstate(OBJ& obj, const object & state) {
		if(!PyBytes_Check(state.ptr()))
			throw std::invalid_argument("Failed to unpickle, unexpected type!");
		std::stringstream ss( std::string( PyBytes_AS_STRING(state.ptr()), PyBytes_Size(state.ptr()) ) );
		obj.load( ss );
	}
};

template<typename OBJ>
struct SaveLoad_pickle_suite_shared_ptr : pickle_suite {
	static object getstate(const std::shared_ptr<OBJ>& obj) {
		std::stringstream ss;
		obj->save( ss );
		std::string data = ss.str();
		return object( handle<>( PyBytes_FromStringAndSize( data.data(), data.size() ) ) );
	}
	
	static void setstate(std::shared_ptr<OBJ> obj, const object & state) {
		if(!PyBytes_Check(state.ptr()))
			throw std::invalid_argument("Failed to unpickle, unexpected type!");
		std::stringstream ss( std::string( PyBytes_AS_STRING(state.ptr()), PyBytes_Size(state.ptr()) ) );
		obj->load( ss );
	}
};

template<typename OBJ>
struct VectorSaveLoad_pickle_suite_shared_ptr : pickle_suite {
	static object getstate(const std::vector< std::shared_ptr<OBJ> > & obj) {
		std::stringstream ss;
		const int nobj = obj.size();
		ss.write( (const char*)&nobj, sizeof(nobj) );
		for( int i=0; i<nobj; i++ )
			obj[i]->save( ss );
		std::string data = ss.str();
		return object( handle<>( PyBytes_FromStringAndSize( data.data(), data.size() ) ) );
	}
	
	static void setstate(std::vector< std::shared_ptr<OBJ> > & obj, const object & state) {
		if(!PyBytes_Check(state.ptr()))
			throw std::invalid_argument("Failed to unpickle, unexpected type!");
		std::stringstream ss( std::string( PyBytes_AS_STRING(state.ptr()), PyBytes_Size(state.ptr()) ) );
		int nobj = 0;
		ss.read( (char*)&nobj, sizeof(nobj) );
		obj.resize( nobj );
		for( int i=0; i<nobj; i++ ) {
			obj[i] = std::make_shared<OBJ>();
			obj[i]->load( ss );
		}
	}
};

void defineUtil();
