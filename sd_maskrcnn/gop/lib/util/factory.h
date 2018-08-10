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
#include "util.h"
#include <tuple>
#include <memory>

template<typename T, typename A>
class Factory {
public:
	typedef T Type;
	typedef A Args;
protected:
	virtual std::shared_ptr<Type> createFromTuple( const Args & args ) const = 0;
public:
	virtual ~Factory(){
	}
template<typename ...ARGS>
	std::shared_ptr<T> create( ARGS... args ) {
		return createFromTuple( std::make_tuple( args... ) );
	}
template<typename TT, typename ...PARAMS>
	static std::shared_ptr< Factory<T,A> > make( PARAMS... params );
template<typename TT>
	static std::shared_ptr< Factory<T,A> > make();
};

namespace FactoryPrivate {
	template<typename F, typename T, typename ...PARAMS>
	class TypedFactoryParam: public F {
	protected:
		std::tuple<PARAMS...> params_;
		virtual std::shared_ptr<typename F::Type> createFromTuple( const typename F::Args & args ) const {
			return make_shared_from_tuple<T>( std::tuple_cat( args, params_ ) );
		}
	public:
		TypedFactoryParam( PARAMS... params ):params_(std::make_tuple( params...)) {}
	};
	template<typename F, typename T>
	class TypedFactory: public F {
	protected:
		virtual std::shared_ptr<typename F::Type> createFromTuple( const typename F::Args & args ) const {
			return make_shared_from_tuple<T>( args );
		}
	};
}
template<typename T, typename A>
template<typename TT, typename ...PARAMS>
std::shared_ptr< Factory< T, A > > Factory<T,A>::make( PARAMS... params ) {
	using namespace FactoryPrivate;
	return std::make_shared< TypedFactoryParam< Factory< T, A >, TT, PARAMS... > >( params... );
}
template<typename T, typename A>
template<typename TT>
std::shared_ptr< Factory<T,A> > Factory<T,A>::make() {
	using namespace FactoryPrivate;
	return std::make_shared< TypedFactory< Factory< T, A >, TT > > ();
}
