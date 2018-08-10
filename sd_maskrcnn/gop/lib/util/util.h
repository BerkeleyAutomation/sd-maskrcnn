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
#include <unordered_map>
#include <string>
#include <atomic>
#include <memory>
#include <stdexcept>

double getTime();
void tic();
double toc();
double toc( const char * str );
struct Clock{
	double tic_time;
	Clock();
	virtual ~Clock(){}
	virtual void tic();
	virtual double toc();
};
struct Timer: public Clock{
	bool print_on_exit_;
	std::unordered_map<std::string,double> time_;
	Timer();
	virtual ~Timer();
	virtual void toc(std::string name="other");
	virtual void print() const;
};
class ProgressPrint {
protected:
	std::string msg_;
	float start_, range_;
public:
	ProgressPrint( const std::string & msg, float end );
	ProgressPrint( const std::string & msg, float start, float end );
	~ProgressPrint();
	void update( float n ) const;
};
class UpdateProgressPrint : protected ProgressPrint {
protected:
	std::string msg_;
	float start_, range_;
	std::atomic<float> p_;
public:
template<typename ...ARGS>
	UpdateProgressPrint( ARGS ... args ):ProgressPrint( args ... ),p_(0){
	}
	void updateDelta( float delta_n );
};
class MemoryPool {
	MemoryPool( const MemoryPool & o ) = delete;
	char * data_;
public:
	MemoryPool( int size );
	~MemoryPool();
	char * data() const;
};
// Assert with exception handling
class AssertException: public std::logic_error {
public:
	AssertException( const std::string & assertion, const std::string & location ): logic_error("Assertion \""+assertion+"\" failed in "+location) {}
};
#define eassert( x ) {if (!(x)) throw AssertException( _str(x), FILE_AND_LINE );}
#define _xstr(s) _str(s)
#define _str(s) #s
#define LINE_STRING _xstr(__LINE__)
#define FILE_AND_LINE ((std::string)__FILE__ + (std::string)":" + LINE_STRING)

// Create a shared_ptr given a type and a tuple of arguments
template<typename T, int N> struct MakeSharedFromTuple {
	template<typename ...A1, typename ...A2>
	static std::shared_ptr<T> make( const std::tuple< A1... > & params, A2 ... args ) {
		return MakeSharedFromTuple<T,N-1>::make( params, std::get<N-1>(params), args... );
	}
};
template<typename T> struct MakeSharedFromTuple<T,0> {
	template<typename ...A1, typename ...A2>
	static std::shared_ptr<T> make( const std::tuple< A1... > & params, A2 ... args ) {
		return std::make_shared<T>( args... );
	}
};
template<typename T,typename ...A1>
std::shared_ptr<T> make_shared_from_tuple( const std::tuple< A1... > & params ) {
	return MakeSharedFromTuple<T,sizeof...(A1)>::make( params );
}

// Create a shared_ptr given a type and a tuple of arguments
template<typename T, typename F, int N> struct CallTuple {
	template<typename ...A1, typename ...A2>
	static T call( const F & f, const std::tuple< A1... > & params, A2 ... args ) {
		return CallTuple<T,F,N-1>::call( f, params, std::get<N-1>(params), args... );
	}
};
template<typename T, typename F> struct CallTuple<T,F,0> {
	template<typename ...A1, typename ...A2>
	static T call( const F & f, const std::tuple< A1... > & params, A2 ... args ) {
		return f( args... );
	}
};
template<typename T, typename F, typename ...A1>
T call_tuple( const F & f, const std::tuple< A1... > & params ) {
	return CallTuple<T,F,sizeof...(A1)>::call( f, params );
}
